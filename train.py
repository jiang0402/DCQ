"""Train and evaluate DCQ-DETR from the command line."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import torch
from torch.utils.data import DataLoader, RandomSampler

from data import CLASS_NAMES, CONDITIONS, IMAGE_SIZES, PairedTiles, collate_pairs
from functions import DetectionLoss, RotatedEvaluator
from model import DCQDETR, ModelConfig


MODEL_OPTIONS = (
    "sar_channels", "hidden_dim", "num_queries", "encoder_layers", "decoder_layers",
    "compensation_layers", "heads", "points", "dropout", "denoising_queries",
    "proposal", "sampling",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=CLASS_NAMES)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--class-names", nargs="+")
    parser.add_argument("--sar-channels", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Training checkpoint, or model weights with --eval-only")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--eval-conditions", nargs="+", choices=CONDITIONS, default=["clean"])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--preset", choices=("default", "rtx5090"), help="Runtime defaults for local or larger CUDA runs")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"))
    parser.add_argument("--imagenet", action="store_true", help="Initialize with torchvision ImageNet weights")
    parser.add_argument("--epochs", type=int, default=36, help="Total epoch count, including resumed epochs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch-factor", type=int, help="Number of queued batches per worker")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, help="Reuse the training workers between epochs")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=[27, 33])
    parser.add_argument("--clip-grad", type=float, default=0.1)
    parser.add_argument("--defect-weight", type=float, default=0.5)
    parser.add_argument("--optical-probability", type=float, default=0.25)
    parser.add_argument("--sar-probability", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    for option in (
        "hidden-dim", "num-queries", "encoder-layers", "decoder-layers",
        "compensation-layers", "heads", "points", "denoising-queries",
    ):
        parser.add_argument(f"--{option}", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--proposal", choices=("quota", "global"))
    parser.add_argument("--sampling", choices=("base", "gate", "weights", "full"))
    args = parser.parse_args()

    positive = (args.epochs, args.cpu_threads, args.eval_every, args.log_every)
    counts = (args.batch_size, args.prefetch_factor, args.train_limit, args.eval_limit)
    if (
        min(positive) < 1 or (args.workers is not None and args.workers < 0)
        or any(count is not None and count < 1 for count in counts)
    ):
        parser.error("Expected positive counts and a nonnegative number of workers")
    if min(args.lr, args.backbone_lr, args.clip_grad) <= 0 or min(args.weight_decay, args.defect_weight) < 0:
        parser.error("Invalid learning rate, gradient clipping, or loss weight")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    if not args.eval_only and (args.eval_split != "val" or "clean" not in args.eval_conditions):
        parser.error("Training selects best.pt on the clean validation split")
    if args.resume and args.imagenet:
        parser.error("Choose checkpoint loading or ImageNet initialization")
    return args


def get_configuration(args, checkpoint):
    saved_model = checkpoint["model_config"] if checkpoint else {}
    saved_data = checkpoint["data_config"] if checkpoint else {}
    dataset = args.dataset or saved_data.get("dataset", "ogsod2")
    overrides = {key: getattr(args, key) for key in MODEL_OPTIONS if getattr(args, key) is not None}
    names = args.class_names or (
        saved_data.get("class_names") if saved_data.get("dataset") == dataset else None
    ) or CLASS_NAMES[dataset]
    names = tuple(names)
    if len({name.lower().replace("-", "_") for name in names}) != len(names):
        raise ValueError("Class names must be unique")
    settings = dict(saved_model, **overrides)
    if not saved_model:
        settings["backbone_norm"] = "frozen" if args.imagenet else "group"
        settings["sar_channels"] = overrides.get("sar_channels", 3)
        if "decoder_layers" in settings and "compensation_layers" not in settings:
            settings["compensation_layers"] = min(3, settings["decoder_layers"])
    settings["num_classes"] = len(names)
    model = ModelConfig(**settings)
    size = args.image_size if args.image_size is not None else saved_data.get("image_size", IMAGE_SIZES[dataset])
    data = {"dataset": dataset, "image_size": size, "class_names": names, "sar_channels": model.sar_channels}
    if checkpoint and (asdict(model) != saved_model or names != tuple(saved_data["class_names"])):
        raise ValueError("Network dimensions, ablation settings, and class order must match the checkpoint")
    if checkpoint and not args.eval_only and data != saved_data:
        raise ValueError("Resumed training requires the dataset layout and tile size stored in the checkpoint")
    return model, data


def _server_batch(device, image_size):
    """Start from 32 tiles at 256 pixels with at least 26 GiB available."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    available = torch.cuda.mem_get_info(device)[0] / 2**30
    area = max(1, (image_size / 256) ** 2)
    batch = int(32 * min(1, available / 26) / area)
    return batch // 4 * 4 if batch >= 4 else max(1, batch)


def configure_runtime(args, device, image_size, saved=None):
    saved = saved or {}
    inherited = saved if args.preset is None and saved.get("device_type") == device.type else {}
    preset = args.preset or saved.get("preset", "default")
    cuda = device.type == "cuda"
    server = cuda and preset == "rtx5090"

    batch = args.batch_size if args.batch_size is not None else inherited.get("batch_size")
    if batch is None:
        batch = _server_batch(device, image_size) if server else 2
    workers = args.workers if args.workers is not None else inherited.get("workers")
    if workers is None:
        workers = 0
        if server:
            try:
                cpus = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                cpus = os.cpu_count() or 1
            workers = min(8, cpus // 2)
    prefetch = args.prefetch_factor or inherited.get("prefetch_factor", 2)
    persistent = args.persistent_workers
    if persistent is None:
        persistent = inherited.get("persistent_workers", server)

    amp = (args.amp if args.amp is not None else inherited.get("amp", True)) and cuda
    dtype = args.amp_dtype or inherited.get("amp_dtype")
    if amp and (dtype == "bf16" or (dtype is None and server)):
        with torch.cuda.device(device):
            supported = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
        if dtype == "bf16" and not supported:
            raise ValueError("The selected CUDA device lacks native BF16 support. Use --amp-dtype fp16.")
        dtype = "bf16" if supported else "fp16"

    runtime = {
        "preset": preset, "device_type": device.type, "batch_size": batch, "workers": workers,
        "prefetch_factor": prefetch, "persistent_workers": bool(persistent and workers and not args.eval_only),
        "amp": amp, "amp_dtype": (dtype or "fp16") if amp else None,
        "seed": args.seed if args.seed is not None else saved.get("seed", 42),
    }
    vars(args).update(runtime)
    return runtime


def make_loader(dataset, args, device, generator, shuffle=False):
    options = {}
    if args.workers:
        options["prefetch_factor"] = getattr(args, "prefetch_factor", 2)
        if shuffle and getattr(args, "persistent_workers", False):
            # Seed sampling independently of worker launches, including on resume.
            options["sampler"] = RandomSampler(dataset, generator=generator)
            options["persistent_workers"] = True
            generator = torch.Generator().manual_seed(generator.initial_seed())
            shuffle = False
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_pairs,
        num_workers=args.workers, pin_memory=device.type == "cuda", generator=generator, **options,
    )


def move_batch(batch, device):
    batch["targets"] = [
        {key: value.to(device, non_blocking=True) for key, value in target.items()} for target in batch["targets"]
    ]
    for key in ("optical", "sar", "valid", "masks"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def make_optimizer(model, args, device):
    backbone, other = [], []
    for name, param in model.named_parameters():
        group = backbone if ".backbone." in name else other
        group.append(param)
    fused = device.type == "cuda"
    return torch.optim.AdamW(
        [{"params": backbone, "lr": args.backbone_lr}, {"params": other, "lr": args.lr}],
        weight_decay=args.weight_decay, fused=fused,
    )


def train_epoch(model, criterion, loader, optimizer, scaler, device, amp, args, epoch, amp_dtype=torch.float16):
    model.train()
    count = 0
    totals = {}
    scaling = scaler.is_enabled()
    if scaling:
        scale_probe = torch.ones((), device=device)
        amp_updates = torch.zeros((), dtype=torch.long, device=device)
    for iteration, batch in enumerate(loader, 1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            output = model(batch["optical"], batch["sar"], batch["valid"], batch["targets"])
        loss, terms = criterion(output, batch["targets"], batch["masks"], batch["valid"])
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss in epoch {epoch}, batch {iteration}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=not scaling)
        if scaling:
            # Keep update accounting on the device until the epoch ends.
            old_scale = scaler.scale(scale_probe)
        scaler.step(optimizer)
        scaler.update()
        if scaling:
            amp_updates += scaler.scale(scale_probe) >= old_scale
        size = len(batch["targets"])
        count += size
        for key, value in {"total": loss, **terms}.items():
            totals[key] = totals.get(key, 0) + value.detach() * size
        if iteration % args.log_every == 0 or iteration == len(loader):
            print(f"epoch {epoch:03d} | step {iteration:05d}/{len(loader):05d} | loss {float(loss.detach()):.4f}", flush=True)
    averages = (torch.stack(list(totals.values())) / count).cpu().tolist()
    updates = int(amp_updates) if scaling else len(loader)
    return {**dict(zip(totals, averages)), "optimizer_updates": updates}


@torch.inference_mode()
def evaluate(model, loader, device, amp, amp_dtype=torch.float16):
    model.eval()
    metric = RotatedEvaluator(loader.dataset.class_names)
    for batch in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            output = model(*(batch[key].to(device, non_blocking=True) for key in ("optical", "sar", "valid")))
        masks = batch["masks"] if loader.dataset.condition != "clean" else None
        metric.update(output, batch["targets"], masks)
    return metric.compute()


def save_checkpoint(payload, path):
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            torch.save(payload, handle)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    cv2.setNumThreads(0)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Choose a CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This Python environment does not provide usable CUDA")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    model_config, data_config = get_configuration(args, checkpoint)
    runtime = configure_runtime(args, device, data_config["image_size"], checkpoint.get("runtime") if checkpoint else None)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    model = DCQDETR(model_config, imagenet=args.imagenet).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    print(f"device={device} runtime={runtime} model_config={asdict(model_config)}", flush=True)

    eval_loaders = {}
    for condition in dict.fromkeys(args.eval_conditions):
        data = PairedTiles(
            args.data_root, split=args.eval_split, condition=condition,
            seed=args.seed, limit=args.eval_limit, **data_config,
        )
        eval_loaders[condition] = make_loader(data, args, device, torch.Generator().manual_seed(args.seed))
    print(f"evaluation pairs={len(data)} split={args.eval_split} classes={data.class_names}", flush=True)

    if args.eval_only:
        results = {name: evaluate(model, loader, device, args.amp, amp_dtype) for name, loader in eval_loaders.items()}
        print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
        return

    output = args.output
    if output is None:
        output = args.resume.parent if args.resume else Path("runs") / datetime.now().strftime("dcq_detr_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    if last.exists() and (not args.resume or not os.path.samefile(last, args.resume)):
        raise FileExistsError(f"{last} already exists. Use --resume with this checkpoint or choose a new --output.")

    train_set = PairedTiles(
        args.data_root, split="train", training=True, optical_probability=args.optical_probability,
        sar_probability=args.sar_probability, seed=args.seed, limit=args.train_limit,
        epoch_seeded=args.persistent_workers, **data_config,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(train_set, args, device, generator, shuffle=True)
    print(f"training pairs={len(train_set)} output={output}", flush=True)
    if checkpoint and "optimizer" not in checkpoint:
        raise ValueError("Use last.pt to resume optimizer state. best.pt is for evaluation.")
    optimizer = make_optimizer(model, args, device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=args.amp and amp_dtype == torch.float16)
    criterion = DetectionLoss(defect_weight=args.defect_weight)
    first_epoch, best = 1, -1.0
    if checkpoint:
        state = checkpoint["optimizer"]
        # AdamW moves saved step counters to the parameter device when loading fused groups.
        groups = [dict(group, fused=device.type == "cuda", foreach=None) for group in state["param_groups"]]
        optimizer.load_state_dict({**state, "param_groups": groups})
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler.is_enabled() and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        cuda_rng = checkpoint.get("cuda_rng")
        if device.type == "cuda" and cuda_rng and len(cuda_rng) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_rng)
        first_epoch, best = checkpoint["epoch"] + 1, checkpoint["best_ap"]
        if not (output / "best.pt").is_file():
            best = -1.0
        print(f"resumed epoch={first_epoch - 1} best_AP={best:.4f}", flush=True)
        del checkpoint

    if first_epoch > args.epochs:
        print(f"Checkpoint has already reached epoch {args.epochs}.")
        return

    for epoch in range(first_epoch, args.epochs + 1):
        train_set.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        lrs = [group["lr"] for group in optimizer.param_groups]
        loss = train_epoch(model, criterion, train_loader, optimizer, scaler, device, args.amp, args, epoch, amp_dtype)
        train_seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        scheduler.step()
        validation = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = {name: evaluate(model, loader, device, args.amp, amp_dtype) for name, loader in eval_loaders.items()}
        improved = bool(validation) and validation["clean"]["AP"] > best
        if improved:
            best = validation["clean"]["AP"]
        record = {
            "epoch": epoch, "loss": loss, "validation": validation, "best_ap": best,
            "learning_rates": lrs, "seconds": round(time.perf_counter() - start, 2),
            "train_imgs_per_sec": round(len(train_set) / train_seconds, 2),
        }
        if peak is not None:
            record["train_peak_gpu_gib"] = round(peak, 2)
        model_payload = {
            "epoch": epoch, "model": model.state_dict(), "model_config": asdict(model_config),
            "data_config": data_config, "runtime": runtime, "best_ap": best, "metrics": record,
        }
        if improved:
            save_checkpoint(model_payload, output / "best.pt")
        save_checkpoint({
            **model_payload, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, last)
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        print(encoded, flush=True)


if __name__ == "__main__":
    main()
