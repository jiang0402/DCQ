"""Rotated-box geometry, matching, training losses, and evaluation."""

import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F


def wrap_angle(angle):
    """Map an unoriented axis angle to [-pi/2, pi/2)."""
    return (angle + math.pi / 2).remainder(math.pi) - math.pi / 2


def canonicalize_boxes(boxes):
    """Canonical xywha with the long edge first, in a square coordinate frame."""
    center, sides, angle = boxes.split((2, 2, 1), dim=-1)
    swap = sides[..., 1:2] > sides[..., :1]
    sides = torch.where(swap, sides.flip(-1), sides)
    return torch.cat((center, sides, wrap_angle(angle + swap * (math.pi / 2))), dim=-1)


def rotate_offsets(offsets, angle):
    """Rotate xy offsets in image coordinates, where y increases downwards."""
    x, y = offsets.unbind(-1)
    cos, sin = angle.cos(), angle.sin()
    return torch.stack((cos * x - sin * y, sin * x + cos * y), dim=-1)


def obb_corners(boxes):
    width, height = (boxes[..., 2:4] * 0.5).unbind(-1)
    x = torch.stack((-width, width, width, -width), dim=-1)
    y = torch.stack((-height, -height, height, height), dim=-1)
    offsets = torch.stack((x, y), dim=-1)
    return boxes[..., None, :2] + rotate_offsets(offsets, boxes[..., 4, None])


def sample_grid(values, positions):
    # Use float32 coordinates in BF16 mode to retain subpixel positions.
    if values.dtype == torch.bfloat16:
        values, positions = values.float(), positions.float()
    return F.grid_sample(values, positions.to(values.dtype), align_corners=False)


def sample_box_regions(maps, boxes):
    """Average 9 interior samples per box. Maps are BCHW and boxes are BQ5."""
    dtype = torch.float32 if maps.dtype == torch.bfloat16 else maps.dtype
    grid = torch.arange(-1, 2, device=maps.device, dtype=dtype) / 3
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    offsets = torch.stack((x, y), dim=-1).reshape(1, 1, 9, 2)
    positions = boxes[..., None, :2] + rotate_offsets(
        offsets * boxes[..., None, 2:4], boxes[..., 4, None]
    )
    sampled = sample_grid(maps, positions * 2 - 1)
    return sampled.mean(dim=-1).transpose(1, 2)


def _gaussian_cost(pred, target):
    """Bounded Gaussian KL distance for normalized rotated rectangles."""
    pred_sides = (pred[..., 2:4] * 0.5).clamp_min(1e-5)
    target_sides = (target[..., 2:4] * 0.5).clamp_min(1e-5)
    delta = rotate_offsets(pred[..., :2] - target[..., :2], -target[..., 4])
    position = (delta / target_sides).square().sum(-1)

    relative = pred[..., 4] - target[..., 4]
    cos2, sin2 = relative.cos().square(), relative.sin().square()
    w2, h2 = pred_sides.square().unbind(-1)
    diagonal = torch.stack((w2 * cos2 + h2 * sin2, w2 * sin2 + h2 * cos2), -1)
    trace = (diagonal / target_sides.square()).sum(-1)
    logdet = 2 * (target_sides.log() - pred_sides.log()).sum(-1)
    distance = (0.5 * (trace + position + logdet - 2)).clamp_min(0).log1p()
    return distance / (1 + distance)


def _hausdorff(distances):
    return torch.maximum(
        distances.amin(dim=-1).amax(dim=-1),
        distances.amin(dim=-2).amax(dim=-1),
    )


@torch.no_grad()
def _match_stages(logits, boxes, targets, valid):
    """Compute all layer costs before one host transfer per minibatch."""
    stages, batch, queries, _ = logits.shape
    empty = np.empty(0, dtype=np.int64)
    result = [[(empty, empty) for _ in range(batch)] for _ in range(stages)]
    if not queries or not any(target["labels"].numel() for target in targets):
        return result

    p = logits.sigmoid()
    pos = 0.25 * (1 - p).square() * F.softplus(-logits)
    neg = 0.75 * p.square() * F.softplus(logits)
    focal = pos - neg
    corners = obb_corners(boxes)
    matrices = []
    for image, target in enumerate(targets):
        gt, labels = target["boxes"].float(), target["labels"]
        if not labels.numel():
            continue
        class_cost = focal[:, image].index_select(-1, labels)
        pred_corners, gt_corners = corners[:, image], obb_corners(gt)
        distances = torch.linalg.vector_norm(
            pred_corners[:, :, None, :, None] - gt_corners[None, None, :, None, :],
            dim=-1,
        )
        geometry = _hausdorff(distances)
        gaussian = _gaussian_cost(boxes[:, image, :, None], gt[None, None])
        matrices.append((image, 2 * class_cost + 5 * geometry + 5 * gaussian))

    packed = torch.cat((valid.flatten().float(), *(cost.flatten() for _, cost in matrices))).cpu().numpy()
    cursor = valid.numel()
    usable = packed[:cursor].reshape(stages, batch, queries).astype(bool)
    for image, cost in matrices:
        size = cost.numel()
        values = packed[cursor:cursor + size].reshape(cost.shape)
        cursor += size
        for stage, matrix in enumerate(values):
            valid_rows = np.flatnonzero(usable[stage, image])
            rows, cols = linear_sum_assignment(matrix[valid_rows])
            result[stage][image] = (valid_rows[rows], cols)
    return result


@torch.no_grad()
def match_queries(logits, boxes, targets, valid=None):
    """Single-layer Hungarian assignments in the original query coordinates."""
    if valid is None:
        valid = torch.ones_like(logits[..., 0], dtype=torch.bool)
    matches = _match_stages(logits[None].float(), boxes[None].float(), targets, valid[None])[0]
    return [tuple(torch.as_tensor(index, device=boxes.device) for index in pair) for pair in matches]


def _focal_loss(logits, target):
    prob = logits.sigmoid()
    error_prob = torch.where(target.bool(), 1 - prob, prob)
    alpha = torch.where(target.bool(), 0.25, 0.75)
    return alpha * error_prob.square() * F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )


def _box_loss(pred, target):
    center_size = (pred[..., :4] - target[..., :4]).abs().sum(-1)
    corners = torch.linalg.vector_norm(
        obb_corners(pred)[..., :, None, :] - obb_corners(target)[..., None, :, :],
        dim=-1,
    )
    return 5 * center_size + 5 * _gaussian_cost(pred, target) + _hausdorff(corners)


class DetectionLoss(nn.Module):
    def __init__(self, defect_weight=0.5):
        super().__init__()
        self.defect_weight = defect_weight

    @staticmethod
    def _detection(outputs, targets):
        max_queries = max(stage["logits"].size(1) for stage in outputs)
        all_valid = outputs[0]["logits"].new_ones(
            (outputs[0]["logits"].size(0), max_queries), dtype=torch.bool
        )
        score_layers, box_layers, valid_layers = [], [], []
        for stage in outputs:
            scores, boxes = stage["logits"], stage["boxes"]
            mask = stage.get("valid")
            if mask is None:
                mask = all_valid[:, :scores.size(1)]
            gap = max_queries - scores.size(1)
            if gap:
                scores, boxes = (F.pad(values, (0, 0, 0, gap)) for values in (scores, boxes))
                mask = F.pad(mask, (0, gap))
            score_layers.append(scores)
            box_layers.append(boxes)
            valid_layers.append(mask)
        logits = torch.stack(score_layers).float()
        boxes = torch.stack(box_layers).float()
        valid = torch.stack(valid_layers)
        matches = _match_stages(logits.detach(), boxes.detach(), targets, valid)
        lengths = [len(target["labels"]) for target in targets]
        offsets = np.cumsum([0, *lengths])
        packed = [
            np.stack((np.full_like(rows, layer), np.full_like(rows, image), rows, cols + offsets[image]))
            for layer, assignments in enumerate(matches)
            for image, (rows, cols) in enumerate(assignments)
            if rows.size
        ]
        indices = np.concatenate(packed, axis=1) if packed else np.empty((4, 0), dtype=np.int64)
        layer, image, row, target_index = torch.from_numpy(indices).to(boxes.device, non_blocking=True)
        labels = torch.cat([target["labels"] for target in targets])
        ground_truth = torch.cat([target["boxes"] for target in targets]).float()
        class_target = torch.zeros_like(logits)
        class_target[layer, image, row, labels[target_index]] = 1
        cls = (_focal_loss(logits, class_target) * valid[..., None]).flatten(1).sum(1)
        matched_loss = _box_loss(boxes[layer, image, row], ground_truth[target_index])
        reg = boxes.new_zeros(len(outputs)).scatter_add(0, layer, matched_loss)
        normalizer = max(1, sum(lengths))
        return 2 * cls / normalizer, reg / normalizer

    @staticmethod
    def _denoising(outputs, metadata):
        labels, target = metadata["labels"], metadata["boxes"]
        valid, positive = metadata["valid"], metadata["positive"]
        normalizer = positive.sum().clamp_min(1)
        logits = torch.stack([stage["logits"] for stage in outputs]).float()
        boxes = torch.stack([stage["boxes"] for stage in outputs]).float()
        class_target = F.one_hot(labels, logits.shape[-1]).to(logits.dtype) * positive[..., None]
        cls = (_focal_loss(logits, class_target.expand_as(logits)) * valid[..., None]).sum()
        reg = (_box_loss(boxes, target) * positive).sum()
        return (2 * cls + reg) / (len(outputs) * normalizer)

    def forward(self, output, targets, defect_masks, valid):
        stages = [output, *output["auxiliary"], *output["encoder"]]
        cls, box = self._detection(stages, targets)
        losses = {"classification": cls[0], "rotation_box": box[0]}
        start = 1
        for name, weight in (("auxiliary", 1.0), ("encoder", 0.5)):
            if output[name]:
                end = start + len(output[name])
                losses[name] = weight * (cls[start:end] + box[start:end]).mean()
                start = end
        if output["denoising"]:
            losses["denoising"] = self._denoising(output["denoising"], output["dn_metadata"])

        defect_logits = output["defect_logits"].float()
        mask = F.interpolate(defect_masks, defect_logits.shape[-2:], mode="area")
        area = F.interpolate(valid[:, None].float(), defect_logits.shape[-2:], mode="area")
        weights = area * (1 + 3 * mask)
        defect = F.binary_cross_entropy_with_logits(defect_logits, mask, reduction="none")
        losses["defect"] = self.defect_weight * (weights * defect).sum() / weights.sum().clamp_min(1)
        return sum(losses.values()), losses


@torch.no_grad()
def rotated_iou(boxes_a, boxes_b):
    """Pairwise rotated IoU on CPU, with an AABB prefilter for polygon clipping."""
    a, b = boxes_a.detach().float().cpu(), boxes_b.detach().float().cpu()
    iou = np.zeros((len(a), len(b)), dtype=np.float32)
    if not len(a) or not len(b):
        return iou
    # Pixel-sized coordinates are more stable for OpenCV polygon intersections.
    scale = 1024
    pa, pb = (obb_corners(a) * scale).numpy(), (obb_corners(b) * scale).numpy()
    aa = a[:, 2:4].prod(-1).numpy() * scale**2
    ab = b[:, 2:4].prod(-1).numpy() * scale**2
    lower = np.maximum(pa.min(axis=1)[:, None], pb.min(axis=1)[None])
    upper = np.minimum(pa.max(axis=1)[:, None], pb.max(axis=1)[None])
    candidates = np.argwhere((upper > lower).all(axis=-1))
    for row, col in candidates:
        area, _ = cv2.intersectConvexConvex(pa[row], pb[col])
        area = max(0.0, min(float(area), aa[row], ab[col]))
        iou[row, col] = area / max(aa[row] + ab[col] - area, 1e-10)
    return iou


class RotatedEvaluator:
    """Rotated AP with 101 recall points and one-to-one matching at each IoU."""
    thresholds = np.linspace(0.50, 0.95, 10)

    def __init__(self, class_names, max_detections=300):
        self.class_names = class_names
        self.max_detections = max_detections
        self.target_counts = np.zeros(len(class_names), dtype=np.int64)
        self.records = []
        self.proposal_hits = 0
        self.affected_targets = self.affected_proposals = self.affected_matches = 0
        self.measure_affected = False

    @torch.no_grad()
    def update(self, output, targets, masks=None):
        self.measure_affected = self.measure_affected or masks is not None
        scores, labels = output["logits"].float().sigmoid().max(-1)
        decoded = torch.cat((scores[..., None], labels[..., None].float(), output["boxes"].float()), dim=-1).cpu()
        initial = output["initial_boxes"].float().cpu()
        for index, target in enumerate(targets):
            order = decoded[index, :, 0].argsort(descending=True, stable=True)[:self.max_detections]
            predictions = decoded[index, order]
            score = predictions[:, 0].numpy()
            label = predictions[:, 1].long().numpy()
            gt_labels = target["labels"].cpu().numpy()
            self.target_counts += np.bincount(gt_labels, minlength=len(self.class_names))
            iou = rotated_iou(predictions[:, 2:], target["boxes"])
            hits = np.zeros((len(order), len(self.thresholds)), dtype=bool)
            if gt_labels.size:
                proposals = rotated_iou(initial[index], target["boxes"])
                covered = proposals.max(axis=0) >= 0.5
                self.proposal_hits += int(covered.sum())
                if masks is not None:
                    box = target["boxes"].float().to(masks.device)
                    severity = sample_box_regions(masks[index:index + 1], box[None])[0].amax(-1)
                    affected = severity.cpu().numpy() >= 0.5
                    self.affected_targets += int(affected.sum())
                    self.affected_proposals += int((covered & affected).sum())
                iou[label[:, None] != gt_labels[None, :]] = 0
                for col, threshold in enumerate(self.thresholds):
                    used = np.zeros(len(gt_labels), dtype=bool)
                    for row in np.flatnonzero(iou.max(axis=1) >= threshold):
                        available = np.where(used, -1, iou[row])
                        match = available.argmax()
                        if available[match] >= threshold:
                            hits[row, col] = used[match] = True
                    if col == 0 and masks is not None:
                        self.affected_matches += int((used & affected).sum())
            self.records.append((score, label, hits))

    def compute(self):
        scores, classes, hits = (np.concatenate(chunk) for chunk in zip(*self.records))
        ap_by_class = {}
        for cls, (name, count) in enumerate(zip(self.class_names, self.target_counts)):
            if not count:
                continue
            rows = np.flatnonzero(classes == cls)
            rows = rows[np.argsort(-scores[rows], kind="stable")]
            tp = np.cumsum(hits[rows], axis=0)
            precision = tp / np.arange(1, len(rows) + 1)[:, None]
            precision = np.maximum.accumulate(precision[::-1], axis=0)[::-1]
            precision = np.concatenate((precision, np.zeros((1, len(self.thresholds)))))
            recall = tp / count
            indices = np.stack([
                np.searchsorted(col, np.linspace(0, 1, 101), side="left") for col in recall.T
            ])
            ap_by_class[name] = precision[indices, np.arange(len(self.thresholds))[:, None]].mean(-1)
        mean_ap = np.mean(list(ap_by_class.values()), axis=0) if ap_by_class else np.zeros(10)
        metrics = {
            "AP": float(mean_ap.mean()),
            "AP50": float(mean_ap[0]),
            "AP75": float(mean_ap[5]),
            "proposal_recall50": self.proposal_hits / max(1, int(self.target_counts.sum())),
            "AP50_by_class": {name: float(ap[0]) for name, ap in ap_by_class.items()},
        }
        if self.measure_affected:
            n = self.affected_targets
            metrics["affected_targets"] = n
            metrics["affected_proposal_recall50"] = self.affected_proposals / n if n else None
            metrics["affected_detection_recall50"] = self.affected_matches / n if n else None
        return metrics
