"""Paired OGSOD-2.0 and M4-SAR tiles with shared geometry and local corruption."""

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from functions import canonicalize_boxes


CLASS_NAMES = {
    "ogsod2": ("bridge", "harbor", "storage_tank"),
    "m4sar": ("bridge", "harbor", "oil_tank", "playground", "airport", "wind_turbine"),
}
IMAGE_SIZES = {"ogsod2": 256, "m4sar": 512}
CONDITIONS = ("clean", "optical", "sar")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _image_index(directory):
    index = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        if path.stem in index:
            raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
        index[path.stem] = path
    return index


def _read_image(path, channels, optical=False):
    with Image.open(path) as source:
        if optical or source.mode in {"P", "CMYK", "RGBX"}:
            source = source.convert("RGB")
        array = np.array(source)
    if array.ndim == 2:
        array = array[..., None]
    if array.shape[-1] == 4 and channels == 3:
        array = array[..., :3]
    if array.shape[-1] == 1 and channels == 3:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != channels:
        raise ValueError(
            f"{path} has {array.shape[-1]} channels, expected {channels}. "
            "Set --sar-channels to the number of bands in the release."
        )

    if np.issubdtype(array.dtype, np.unsignedinteger):
        scale = np.iinfo(array.dtype).max
    elif array.dtype.kind in "bf" and np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1:
        scale = 1
    else:
        raise ValueError(f"Expected unsigned image data or meaningful float values in [0, 1]: {path}")
    image = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).contiguous()
    return image / scale


def _read_boxes(path, dataset, names, original_width, original_height, new_width, new_height, size):
    boxes, labels = [], []
    ids = {name.lower().replace("-", "_"): idx for idx, name in enumerate(names)}
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path is not None else ()
    for number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields or fields[0].lower().startswith(("imagesource:", "gsd:")):
            continue
        try:
            if dataset == "ogsod2" and len(fields) in (9, 10):
                label = ids[fields[8].lower().replace("-", "_")]
                points = np.asarray(fields[:8], dtype=np.float32).reshape(4, 2)
                points *= (new_width / original_width, new_height / original_height)
            elif dataset == "m4sar" and len(fields) == 9:
                label = int(fields[0])
                points = np.asarray(fields[1:], dtype=np.float32).reshape(4, 2)
                points *= (new_width, new_height)
            else:
                raise ValueError("Expected a DOTA or YOLO-OBB annotation row")
            if not 0 <= label < len(names) or not np.isfinite(points).all():
                raise ValueError("Invalid class ID or vertex coordinate")
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path}:{number}: {line}") from exc

        (cx, cy), (width, height), angle = cv2.minAreaRect(points)
        # Clipping polygons at a tile boundary can produce a zero-area annotation.
        if min(width, height) <= 1e-5:
            continue
        boxes.append((cx / size, cy / size, width / size, height / size, math.radians(angle)))
        labels.append(label)

    return {
        "boxes": canonicalize_boxes(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _shared_geometry(optical, sar, valid, boxes, generator=None):
    if torch.rand((), generator=generator) < 0.5:
        optical, sar, valid = (value.flip(-1) for value in (optical, sar, valid))
        boxes[:, 0] = 1 - boxes[:, 0]
        boxes[:, 4] = -boxes[:, 4]
    turns = int(torch.randint(4, (), generator=generator))
    if turns:
        optical, sar, valid = (value.rot90(turns, (-2, -1)) for value in (optical, sar, valid))
        for _ in range(turns):
            x, y = boxes[:, 0].clone(), boxes[:, 1].clone()
            boxes[:, 0], boxes[:, 1] = y, 1 - x
        boxes[:, 4] -= turns * math.pi / 2
    return optical, sar, valid, canonicalize_boxes(boxes)


def _local_mask(valid, generator):
    height, width = valid.shape
    y_valid = valid.any(dim=1).nonzero(as_tuple=True)[0]
    x_valid = valid.any(dim=0).nonzero(as_tuple=True)[0]
    params = torch.rand(5, generator=generator)
    x = (torch.arange(width) - x_valid[0] + 0.5) / len(x_valid)
    y = (torch.arange(height) - y_valid[0] + 0.5) / len(y_valid)
    radius_x, radius_y = 0.18 + 0.30 * params[2:4]
    ellipse = ((x[None, :] - params[0]) / radius_x).square()
    ellipse = ellipse + ((y[:, None] - params[1]) / radius_y).square()
    variation = torch.rand((1, 1, 6, 6), generator=generator)
    variation = F.interpolate(variation, size=(height, width), mode="bilinear", align_corners=False)
    edge = 1 + (variation[0, 0] - 0.5) * 0.45
    return ((edge - ellipse) / 0.25).clamp(0, 1) * (0.55 + 0.45 * params[4]) * valid


def _corrupt(optical, sar, valid, condition, generator):
    masks = torch.zeros((2, *valid.shape), dtype=optical.dtype)
    if condition == "optical":
        masks[0] = _local_mask(valid, generator)
        optical = optical.lerp(torch.ones_like(optical), masks[0])
    elif condition == "sar":
        masks[1] = _local_mask(valid, generator)
        low = F.avg_pool2d(sar[None], 4)
        blurred = F.interpolate(low, size=valid.shape, mode="bilinear", align_corners=False)[0]
        sar = sar.lerp(blurred, masks[1])
    return optical, sar, masks


class PairedTiles(Dataset):
    optical_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    optical_std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(
        self, root, dataset, split, image_size, sar_channels=3, class_names=None,
        training=False, condition="clean", optical_probability=0.25, sar_probability=0.25,
        seed=42, limit=None, epoch_seeded=False,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown evaluation condition: {condition}")
        if image_size < 32 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if min(optical_probability, sar_probability) < 0 or optical_probability + sar_probability > 1:
            raise ValueError("Corruption probabilities must be nonnegative and sum to at most 1")
        root = Path(root)
        if dataset == "ogsod2":
            optical_dir, sar_dir = (root / modality / split / "images" for modality in ("rgb", "sar"))
            label_dir = root / "rgb" / split / "labelTxt"
        elif dataset == "m4sar":
            optical_dir, sar_dir = (root / modality / "images" / split for modality in ("optical", "sar"))
            label_dir = root / "optical" / "labels" / split
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.class_names = tuple(class_names or CLASS_NAMES[dataset])
        self.dataset = dataset
        self.image_size = image_size
        self.sar_channels = sar_channels
        self.training = training
        self.condition = condition
        self.optical_probability = optical_probability
        self.sar_probability = sar_probability
        self.seed = seed
        self._epoch = torch.zeros((), dtype=torch.long).share_memory_() if training and epoch_seeded else None

        optical, sar = _image_index(optical_dir), _image_index(sar_dir)
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Could not find the label directory: {label_dir}")
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        if optical.keys() != sar.keys():
            missing = sorted(optical.keys() ^ sar.keys())
            raise ValueError(f"{split} has {len(missing)} unpaired image stems, including {missing[:5]}")
        missing = sorted(optical.keys() - labels.keys())
        if missing and dataset == "ogsod2":
            raise ValueError(f"{split} has {len(missing)} missing label files, including {missing[:5]}")
        ids = sorted(optical)
        if limit is not None and len(ids) > limit:
            ids = [ids[index] for index in np.linspace(0, len(ids) - 1, limit, dtype=int)]
        if not ids:
            raise ValueError(f"No paired tiles found under {root} for split {split}")
        self.records = [(optical[name], sar[name], labels.get(name)) for name in ids]

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        if self._epoch is not None:
            self._epoch.fill_(epoch)

    def __getitem__(self, index):
        optical_path, sar_path, label_path = self.records[index]
        optical = _read_image(optical_path, 3, optical=True)
        sar = _read_image(sar_path, self.sar_channels)
        height, width = optical.shape[-2:]
        if sar.shape[-2:] != (height, width):
            raise ValueError(f"The paired tiles have different raster sizes: {optical_path} and {sar_path}")

        size = self.image_size
        scale = size / max(height, width)
        new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
        pair = torch.cat((optical, sar), dim=0)
        if (height, width) != (new_height, new_width):
            pair = F.interpolate(
                pair[None], (new_height, new_width), mode="bilinear", align_corners=False, antialias=True
            )[0]
        pair = F.pad(pair, (0, size - new_width, 0, size - new_height))
        optical, sar = pair.split((3, self.sar_channels), dim=0)
        valid = torch.zeros((size, size), dtype=torch.bool)
        valid[:new_height, :new_width] = True
        target = _read_boxes(label_path, self.dataset, self.class_names, width, height, new_width, new_height, size)

        condition = self.condition
        generator = None
        if self.training:
            if self._epoch is not None:
                seed = self.seed + int(self._epoch) * len(self.records) + index
                generator = torch.Generator().manual_seed(seed % (1 << 64))
            optical, sar, valid, target["boxes"] = _shared_geometry(optical, sar, valid, target["boxes"], generator)
            draw = float(torch.rand((), generator=generator))
            if draw < self.optical_probability:
                condition = "optical"
            elif draw < self.optical_probability + self.sar_probability:
                condition = "sar"
            else:
                condition = "clean"
        else:
            generator = torch.Generator().manual_seed(self.seed + index)
        optical, sar, masks = _corrupt(optical, sar, valid, condition, generator)
        optical = ((optical - self.optical_mean) / self.optical_std).masked_fill(~valid, 0)
        sar = ((sar - 0.5) / 0.25).masked_fill(~valid, 0)
        return {"optical": optical, "sar": sar, "valid": valid, "masks": masks, "target": target}


def collate_pairs(items):
    batch = {key: torch.stack([item[key] for item in items]) for key in ("optical", "sar", "valid", "masks")}
    batch["targets"] = [item["target"] for item in items]
    return batch
