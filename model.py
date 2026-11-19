"""DCQ-DETR with separate optical and SAR memories and a shared rotated decoder."""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops.misc import FrozenBatchNorm2d

from functions import canonicalize_boxes, rotate_offsets, sample_box_regions, sample_grid


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 3
    sar_channels: int = 3
    hidden_dim: int = 256
    num_queries: int = 300
    encoder_layers: int = 3
    decoder_layers: int = 6
    compensation_layers: int = 3
    heads: int = 8
    points: int = 4
    dropout: float = 0.1
    denoising_queries: int = 100
    proposal: str = "quota"
    sampling: str = "full"
    backbone_norm: str = "group"

    def __post_init__(self):
        positive = (self.num_classes, self.sar_channels, self.num_queries, self.encoder_layers,
                    self.decoder_layers, self.heads, self.points)
        if min(positive) < 1 or self.hidden_dim < 16 or self.hidden_dim % 4 or self.hidden_dim % self.heads:
            raise ValueError("Layer counts must be positive and hidden_dim divisible by 4 and heads")
        if not 0 <= self.compensation_layers <= self.decoder_layers or self.denoising_queries < 0:
            raise ValueError("Invalid number of compensation layers or denoising queries")
        if self.proposal not in ("quota", "global") or self.sampling not in ("base", "gate", "weights", "full"):
            raise ValueError("Invalid proposal or sampling mode")
        if not 0 <= self.dropout < 1 or self.backbone_norm not in ("group", "frozen"):
            raise ValueError("Invalid dropout or backbone normalization")


def _group_norm(channels):
    return nn.GroupNorm(math.gcd(32, channels // 2), channels)


def _box_head(dim):
    head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 5))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def _class_head(dim, classes):
    head = nn.Linear(dim, classes)
    nn.init.constant_(head.bias, math.log(0.01 / 0.99))
    return head


def _refine_boxes(references, delta):
    references, delta = references.float(), delta.float()
    center = (torch.logit(references[..., :2], eps=1e-4) + delta[..., :2]).sigmoid()
    sides = 2 * (torch.logit(references[..., 2:4] / 2, eps=1e-4) + delta[..., 2:4]).sigmoid()
    angle = references[..., 4:] + (math.pi / 2) * delta[..., 4:].tanh()
    return canonicalize_boxes(torch.cat((center, sides.clamp_min(1e-5), angle), dim=-1))


def _feature_maps(tokens, shapes):
    chunks = tokens.split([height * width for height, width in shapes], dim=1)
    return [
        chunk.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], height, width)
        for chunk, (height, width) in zip(chunks, shapes)
    ]


class RotatedDeformableAttention(nn.Module):
    levels = 3

    def __init__(self, cfg):
        super().__init__()
        self.heads, self.points = cfg.heads, cfg.points
        self.head_dim = cfg.hidden_dim // cfg.heads
        taps = self.heads * self.levels * self.points
        self.value_projection = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.output_projection = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.offsets = nn.Linear(cfg.hidden_dim, taps * 2)
        self.attention_weights = nn.Linear(cfg.hidden_dim, taps)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        theta = torch.arange(self.heads) * (2 * math.pi / self.heads)
        directions = torch.stack((theta.cos(), theta.sin()), dim=-1)
        directions = directions / directions.abs().amax(dim=-1, keepdim=True)
        radii = torch.arange(1, self.points + 1).reshape(1, 1, -1, 1)
        bias = (directions[:, None, None, :] * radii).expand(-1, self.levels, -1, -1)
        with torch.no_grad():
            self.offsets.bias.copy_(bias.reshape(-1))

    def project_values(self, tokens, shapes, valid):
        projected = self.value_projection(tokens).masked_fill(~valid[..., None], 0)
        if projected.dtype == torch.bfloat16:
            projected = projected.float()
        return [
            feature.reshape(-1, self.head_dim, height, width)
            for feature, (height, width) in zip(_feature_maps(projected, shapes), shapes)
        ]

    def forward(self, query, references, values, weight_query=None):
        batch, length, _ = query.shape
        offsets = self.offsets(query).float().view(batch, length, self.heads, self.levels, self.points, 2)
        logits = self.attention_weights(query if weight_query is None else weight_query)
        weights = logits.view(batch, length, self.heads, self.levels * self.points)
        weights = weights.softmax(dim=-1, dtype=torch.float32).view(
            batch, length, self.heads, self.levels, self.points
        )

        center = references[:, :, None, None, None, :2]
        if references.shape[-1] == 5:
            sides = references[:, :, None, None, None, 2:4]
            angle = references[:, :, None, None, None, 4]
            positions = center + rotate_offsets(offsets * sides / (2 * self.points), angle)
        else:
            normalizer = offsets.new_tensor([(value.shape[-1], value.shape[-2]) for value in values])
            positions = center + offsets / normalizer[None, None, None, :, None, :]
        # grid_sample uses (x, y) in [-1, 1]; positions use the padded image frame.
        positions = positions.mul(2).sub(1)
        result = 0
        for level, value in enumerate(values):
            grid = positions[:, :, :, level].permute(0, 2, 1, 3, 4)
            grid = grid.reshape(batch * self.heads, length, self.points, 2)
            sampled = sample_grid(value, grid)
            attention = weights[:, :, :, level].permute(0, 2, 1, 3)
            attention = attention.reshape(batch * self.heads, 1, length, self.points)
            result = result + (sampled * attention.to(sampled.dtype)).sum(-1)
        result = result.view(batch, self.heads, self.head_dim, length)
        result = result.permute(0, 3, 1, 2).reshape(batch, length, -1)
        return self.output_projection(result)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.hidden_dim
        self.network = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        return self.norm(tokens + self.dropout(self.network(tokens)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = RotatedDeformableAttention(cfg)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = FeedForward(cfg)

    def forward(self, tokens, positions, anchors, shapes, valid):
        values = self.attention.project_values(tokens, shapes, valid)
        evidence = self.attention(tokens + positions, anchors, values)
        tokens = self.ffn(self.norm(tokens + self.dropout(evidence)))
        return tokens.masked_fill(~valid[..., None], 0)


class ModalityEncoder(nn.Module):
    def __init__(self, cfg, channels, imagenet):
        super().__init__()
        norm = FrozenBatchNorm2d if cfg.backbone_norm == "frozen" else _group_norm
        weights = ResNet50_Weights.DEFAULT if imagenet else None
        self.backbone = resnet50(weights=weights, norm_layer=norm)
        if channels != 3:
            original = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False)
            if imagenet:
                with torch.no_grad():
                    weight = original.weight.mean(dim=1, keepdim=True).expand(-1, channels, -1, -1)
                    self.backbone.conv1.weight.copy_(weight * (3 / channels))
        self.backbone.fc = nn.Identity()
        dim = cfg.hidden_dim
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, dim, 1), _group_norm(dim)) for channels in (512, 1024, 2048)
        )
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.encoder_layers))
        self.level_embeddings = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.defect_head = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(dim // 2, 1, 1))
        nn.init.constant_(self.defect_head[-1].bias, -2)
        self.proposal_class = _class_head(dim, cfg.num_classes)
        self.proposal_box = _box_head(dim)

    def forward(self, image, valid):
        backbone = self.backbone
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(image))))
        p3 = backbone.layer2(backbone.layer1(x))
        p4 = backbone.layer3(p3)
        p5 = backbone.layer4(p4)
        features = [project(feature) for project, feature in zip(self.projections, (p3, p4, p5))]
        shapes = [feature.shape[-2:] for feature in features]
        masks = [
            F.adaptive_max_pool2d(valid[:, None].float(), shape).flatten(1).bool() for shape in shapes
        ]
        valid_tokens = torch.cat(masks, dim=1)
        tokens = torch.cat([feature.flatten(2).transpose(1, 2) for feature in features], dim=1)
        tokens = tokens.masked_fill(~valid_tokens[..., None], 0)

        centers, sizes, positions = [], [], []
        frequency = torch.arange(tokens.shape[-1] // 4, device=image.device, dtype=torch.float32)
        frequency = 2 * math.pi / (10000 ** (frequency / len(frequency)))
        for level, (height, width) in enumerate(shapes):
            y = (torch.arange(height, device=image.device, dtype=torch.float32) + 0.5) / height
            x = (torch.arange(width, device=image.device, dtype=torch.float32) + 0.5) / width
            y, x = torch.meshgrid(y, x, indexing="ij")
            center = torch.stack((x, y), dim=-1).reshape(-1, 2)
            phase = center[..., None] * frequency
            encoding = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(1)
            centers.append(center)
            sizes.append(center.new_full(center.shape, 0.05 * 2**level))
            positions.append(encoding + self.level_embeddings[level])
        anchors = torch.cat(centers).unsqueeze(0).expand(image.shape[0], -1, -1)
        positions = torch.cat(positions).to(tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens, positions, anchors, shapes, valid_tokens)
        sizes = torch.cat(sizes).unsqueeze(0).expand_as(anchors)
        priors = torch.cat((anchors, sizes, torch.zeros_like(anchors[..., :1])), dim=-1)
        defects = self.defect_head(features[0])
        return (
            (tokens, shapes, valid_tokens), defects, self.proposal_class(tokens),
            _refine_boxes(priors, self.proposal_box(tokens)),
        )


class DecoderLayer(nn.Module):
    def __init__(self, cfg, compensate):
        super().__init__()
        dim = cfg.hidden_dim
        self.self_attention = nn.MultiheadAttention(dim, cfg.heads, cfg.dropout, batch_first=True)
        self.self_norm, self.cross_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross_attention = nn.ModuleList(RotatedDeformableAttention(cfg) for _ in range(2))
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = FeedForward(cfg)
        self.compensate = compensate
        if compensate:
            self.evidence_query = nn.ModuleList(
                nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) for _ in range(2)
            )
            self.defect_query = nn.ModuleList(nn.Linear(2, dim, bias=False) for _ in range(2))
            self.gates = nn.Sequential(nn.Linear(3 * dim + 2, dim), nn.GELU(), nn.Linear(dim, 4))
            for layer in self.evidence_query:
                nn.init.zeros_(layer[-1].weight)
                nn.init.zeros_(layer[-1].bias)
            for layer in self.defect_query:
                nn.init.zeros_(layer.weight)
            with torch.no_grad():
                self.gates[-1].bias.copy_(torch.tensor([1.0, 1.0, -2.0, -2.0]))
        self.defects_in_offsets = float(cfg.sampling == "full")
        self.defects_in_weights = float(cfg.sampling in ("full", "weights"))

    def forward(self, query, positions, boxes, memories, maps, attention_mask, padding_mask):
        positioned = query + positions
        attended = self.self_attention(
            positioned, positioned, query, attn_mask=attention_mask,
            key_padding_mask=padding_mask, need_weights=False,
        )[0]
        query = self.self_norm(query + self.dropout(attended))
        positioned = query + positions
        values = [attention.project_values(*memory) for attention, memory in zip(self.cross_attention, memories)]
        evidence = [attention(positioned, boxes, value) for attention, value in zip(self.cross_attention, values)]
        if self.compensate:
            defects = sample_box_regions(maps, boxes).to(query.dtype)
            gates = self.gates(torch.cat((query, *evidence, defects), dim=-1))
            alpha = torch.cat((gates[..., :2], torch.zeros_like(gates[..., :1])), dim=-1).softmax(-1)
            beta = gates[..., 2:].sigmoid()
            fused = 0
            for source, attention in enumerate(self.cross_attention):
                receiver = 1 - source
                content = positioned + self.evidence_query[source](evidence[receiver])
                context = self.defect_query[source](defects[..., [receiver, source]])
                conditioned = attention(
                    content + self.defects_in_offsets * context, boxes, values[source],
                    weight_query=content + self.defects_in_weights * context,
                )
                fused = fused + alpha[..., source:source + 1] * (
                    evidence[source] + beta[..., source:source + 1] * (conditioned - evidence[source])
                )
        else:
            fused = (evidence[0] + evidence[1]) * 0.5
        return self.ffn(self.cross_norm(query + self.dropout(fused)))


def _select(logits, boxes, valid, count):
    scores = logits.detach().amax(dim=-1).masked_fill(~valid, -torch.inf)
    indices = scores.topk(min(count, scores.shape[1]), dim=1).indices
    real = valid.gather(1, indices)
    if count:
        top = indices[:, :1]
        indices = torch.where(real, indices, top)
        extra = count - indices.size(1)
        if extra:
            indices = torch.cat((indices, top.expand(-1, extra)), dim=1)
            real = F.pad(real, (0, extra), value=False)
    return {
        "logits": logits.gather(1, indices[..., None].expand(-1, -1, logits.shape[-1])),
        "boxes": boxes.gather(1, indices[..., None].expand(-1, -1, 5)),
        "valid": real,
    }


class DCQDETR(nn.Module):
    """Predict normalized xywha boxes in the shared, padded tile coordinates."""

    def __init__(self, cfg, imagenet=False):
        super().__init__()
        if imagenet and cfg.backbone_norm != "frozen":
            raise ValueError("ImageNet initialization uses FrozenBatchNorm2d")
        self.cfg = cfg
        dim = cfg.hidden_dim
        self.optical = ModalityEncoder(cfg, 3, imagenet)
        self.sar = ModalityEncoder(cfg, cfg.sar_channels, imagenet)
        self.query_embeddings = nn.Embedding(cfg.num_queries, dim)
        self.position = nn.Sequential(nn.Linear(6, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.label_embeddings = nn.Embedding(cfg.num_classes, dim)
        start = cfg.decoder_layers - cfg.compensation_layers
        self.decoder = nn.ModuleList(
            DecoderLayer(cfg, layer >= start and cfg.sampling != "base") for layer in range(cfg.decoder_layers)
        )
        self.class_head = _class_head(dim, cfg.num_classes)
        self.box_heads = nn.ModuleList(_box_head(dim) for _ in range(cfg.decoder_layers))

    def _noisy_queries(self, targets):
        if not self.training or not targets or self.cfg.denoising_queries < 2:
            return None
        slots = min(max(len(target["labels"]) for target in targets), self.cfg.denoising_queries // 2)
        if slots == 0:
            return None
        groups = self.cfg.denoising_queries // (2 * slots)
        batch = len(targets)
        device = targets[0]["labels"].device
        labels = torch.zeros((batch, slots), device=device, dtype=torch.long)
        base_boxes = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.0], device=device).repeat(batch, slots, 1)
        present = torch.zeros((batch, slots), device=device, dtype=torch.bool)
        for index, target in enumerate(targets):
            size = min(slots, len(target["labels"]))
            indices = torch.randperm(len(target["labels"]), device=device)[:size]
            labels[index, :size] = target["labels"][indices]
            base_boxes[index, :size] = target["boxes"][indices]
            present[index, :size] = True

        labels = labels[:, None, None].expand(-1, groups, 2, -1)
        boxes = base_boxes[:, None, None].expand(-1, groups, 2, -1, -1)
        valid = present[:, None, None].expand_as(labels)
        positive = valid & torch.tensor([True, False], device=device)[None, None, :, None]
        draw = torch.rand_like(boxes)
        magnitude = torch.where(positive[..., None], 0.25 * draw, 0.5 + 0.5 * draw)
        noise = magnitude * torch.where(torch.rand_like(boxes) < 0.5, -1, 1)
        displacement = rotate_offsets(noise[..., :2] * boxes[..., 2:4], boxes[..., 4])
        center = (boxes[..., :2] + displacement).clamp(1e-4, 1 - 1e-4)
        sides = (boxes[..., 2:4] * noise[..., 2:4].exp()).clamp(1e-4, 1.99)
        angle = boxes[..., 4:] + noise[..., 4:] * (math.pi / 3)
        noisy_boxes = canonicalize_boxes(torch.cat((center, sides, angle), dim=-1)).reshape(batch, -1, 5)
        alternative = torch.randint(self.cfg.num_classes, labels.shape, device=device)
        noisy_labels = torch.where(torch.rand(labels.shape, device=device) < 0.2, alternative, labels)
        noisy_contents = self.label_embeddings(noisy_labels.reshape(batch, -1))
        meta = {
            "labels": labels.reshape(batch, -1), "boxes": boxes.reshape(batch, -1, 5),
            "valid": valid.reshape(batch, -1), "positive": positive.reshape(batch, -1),
        }
        count = noisy_boxes.shape[1]
        total = count + self.cfg.num_queries
        attention_mask = torch.zeros((total, total), device=device, dtype=torch.bool)
        group_id = torch.arange(count, device=device) // (2 * slots)
        attention_mask[:count, :count] = group_id[:, None] != group_id[None, :]
        attention_mask[count:, :count] = True
        padding = F.pad(~meta["valid"], (0, self.cfg.num_queries), value=False)
        return noisy_contents, noisy_boxes, meta, attention_mask, padding

    def forward(self, optical, sar, valid=None, targets=None):
        if (
            optical.ndim != 4 or sar.ndim != 4 or optical.shape[1] != 3
            or sar.shape[1] != self.cfg.sar_channels or optical.shape[0] != sar.shape[0]
            or optical.shape[-2:] != sar.shape[-2:] or optical.shape[-2] != optical.shape[-1]
        ):
            raise ValueError("Expected matching square BCHW optical and SAR tiles with configured channels")
        if valid is None:
            valid = torch.ones_like(optical[:, 0], dtype=torch.bool)
        elif valid.dtype != torch.bool or valid.shape != optical[:, 0].shape:
            raise ValueError("valid must be a boolean mask of shape BHW")
        opt_memory, opt_defect, opt_logits, opt_boxes = self.optical(optical, valid)
        sar_memory, sar_defect, sar_logits, sar_boxes = self.sar(sar, valid)
        memories = opt_memory, sar_memory
        defects = torch.cat((opt_defect, sar_defect), dim=1)
        maps = defects.sigmoid()
        count = self.cfg.num_queries
        if self.cfg.proposal == "quota":
            optical_count = (count + 1) // 2
            first = _select(opt_logits, opt_boxes, opt_memory[2], optical_count)
            second = _select(sar_logits, sar_boxes, sar_memory[2], count - optical_count)
            initial = torch.cat((first["boxes"], second["boxes"]), dim=1)
        else:
            initial = _select(
                torch.cat((opt_logits, sar_logits), dim=1), torch.cat((opt_boxes, sar_boxes), dim=1),
                torch.cat((opt_memory[2], sar_memory[2]), dim=1), count,
            )["boxes"]
        references = initial.detach()
        query = self.query_embeddings.weight.unsqueeze(0).expand(optical.shape[0], -1, -1)

        denoising = self._noisy_queries(targets)
        dn_count, meta, attention_mask, padding = 0, None, None, None
        if denoising is not None:
            noisy_contents, noisy_boxes, meta, attention_mask, padding = denoising
            dn_count = noisy_boxes.shape[1]
            query = torch.cat((noisy_contents, query), dim=1)
            references = torch.cat((noisy_boxes, references), dim=1)

        stages, dn_stages = [], []
        for layer, predict_box in zip(self.decoder, self.box_heads):
            angle = 2 * references[..., 4:]
            positions = self.position(torch.cat((references[..., :4], angle.sin(), angle.cos()), dim=-1))
            query = layer(query, positions, references, memories, maps, attention_mask, padding)
            logits = self.class_head(query)
            boxes = _refine_boxes(references, predict_box(query))
            stages.append({"logits": logits[:, dn_count:], "boxes": boxes[:, dn_count:]})
            if dn_count:
                dn_stages.append({"logits": logits[:, :dn_count], "boxes": boxes[:, :dn_count]})
            references = boxes.detach()

        encoder_predictions = []
        if self.training:
            encoder_predictions = [
                _select(opt_logits, opt_boxes, opt_memory[2], count),
                _select(sar_logits, sar_boxes, sar_memory[2], count),
            ]
        return {
            **stages[-1], "initial_boxes": initial.detach(), "defect_logits": defects,
            "auxiliary": stages[:-1] if self.training else [], "encoder": encoder_predictions,
            "denoising": dn_stages, "dn_metadata": meta,
        }
