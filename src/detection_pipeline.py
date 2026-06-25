from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

from common import (
    AnnotationRecord,
    ensure_dir,
    grouped_holdout_query_names,
    image_prefix,
    load_detection_records,
    safe_open_image,
    split_records_by_prefix,
)


@dataclass
class PredictionMetrics:
    precision: float
    recall: float
    f1: float
    mean_iou: float
    matched: int
    predicted: int
    targets: int


class DetectionDataset(Dataset):
    def __init__(self, records: list[AnnotationRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        # 读图失败时返回 None，后面的 collate 会统一过滤掉损坏样本。
        image = safe_open_image(record.image_path)
        if image is None:
            return None
        tensor = TF.to_tensor(image)
        boxes = torch.tensor(record.boxes, dtype=torch.float32)
        # 实验里把所有文字都合并成单一的 text 类别，避免多类别标注带来额外复杂度。
        labels = torch.ones((len(record.boxes),), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(record.boxes),), dtype=torch.int64),
            "path": str(record.image_path),
        }
        return tensor, target


    # 跳过损坏样本，保证批处理训练和评估能够继续进行。
def collate_detection(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return [], []
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def build_model(num_classes: int) -> torch.nn.Module:
    # 用预训练 Faster R-CNN 作为初始化，并把分类头替换成实验需要的类别数。
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    # 仅在 CUDA 上启用混合精度，CPU 环境下则按普通精度训练。
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    losses = []
    for images, targets in loader:
        if not images:
            continue
        images = [image.to(device) for image in images]
        # 把目标框和标签搬到同一设备上，才能送入检测模型计算损失。
        moved_targets = []
        for target in targets:
            moved_targets.append(
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                    "image_id": target["image_id"].to(device),
                    "area": target["area"].to(device),
                    "iscrowd": target["iscrowd"].to(device),
                }
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss_dict = model(images, moved_targets)
            loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def evaluate_predictions(
    model: torch.nn.Module,
    records: list[AnnotationRecord],
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[PredictionMetrics, dict[str, dict[str, list[list[float]] | list[float]]]]:
    model.eval()
    prediction_store = {}
    matched = 0
    total_pred = 0
    total_gt = 0
    matched_ious: list[float] = []

    with torch.inference_mode():
        for record in records:
            image = safe_open_image(record.image_path)
            if image is None:
                continue
            tensor = TF.to_tensor(image).to(device)
            outputs = model([tensor])[0]
            scores = outputs["scores"].detach().cpu().numpy()
            boxes = outputs["boxes"].detach().cpu().numpy()
            keep = scores >= score_threshold
            pred_boxes = boxes[keep].astype(np.float32)
            pred_scores = scores[keep].astype(np.float32)
            gt_boxes = np.array(record.boxes, dtype=np.float32)
            total_pred += len(pred_boxes)
            total_gt += len(gt_boxes)

            # 采用贪心方式按 IoU 从大到小配对，统计匹配到的预测框和真实框。
            ious = box_iou_numpy(pred_boxes, gt_boxes)
            used_pred = set()
            used_gt = set()
            candidates = []
            for pred_index in range(ious.shape[0]):
                for gt_index in range(ious.shape[1]):
                    candidates.append((float(ious[pred_index, gt_index]), pred_index, gt_index))
            candidates.sort(reverse=True)
            for iou, pred_index, gt_index in candidates:
                if iou < iou_threshold:
                    break
                if pred_index in used_pred or gt_index in used_gt:
                    continue
                used_pred.add(pred_index)
                used_gt.add(gt_index)
                matched += 1
                matched_ious.append(iou)

            # 这里把每张图的预测与真值都保存下来，后面画可视化面板会直接复用。
            prediction_store[record.image_path.name] = {
                "pred_boxes": pred_boxes.tolist(),
                "pred_scores": pred_scores.tolist(),
                "gt_boxes": gt_boxes.tolist(),
            }

    precision = matched / total_pred if total_pred else 0.0
    recall = matched / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return (
        PredictionMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=mean_iou,
            matched=matched,
            predicted=total_pred,
            targets=total_gt,
        ),
        prediction_store,
    )


    # 左轴画训练损失，右轴画验证 F1，方便同时观察收敛和泛化情况。
def plot_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_f1 = [entry["val_f1"] for entry in history]

    figure, axis_left = plt.subplots(figsize=(8, 5))
    axis_right = axis_left.twinx()
    axis_left.plot(epochs, train_losses, color="#8c2d04", marker="o", linewidth=2, label="Train loss")
    axis_right.plot(epochs, val_f1, color="#08519c", marker="s", linewidth=2, label="Val F1")
    axis_left.set_xlabel("Epoch")
    axis_left.set_ylabel("Train loss")
    axis_right.set_ylabel("Validation F1")
    axis_left.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def resize_for_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(image, size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def draw_boxes(image: Image.Image, boxes: list[list[float]], color: str, width: int = 4) -> Image.Image:
    result = image.copy()
    drawer = ImageDraw.Draw(result)
    for x1, y1, x2, y2 in boxes:
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return result


    # 统一版式：上方查询图，中间检测结果，右侧 Top-5 检索结果。
def compose_visual_panel(
    query_image: Image.Image,
    detected_image: Image.Image,
    retrieved_images: list[tuple[Image.Image, bool, str]],
    title: str,
) -> Image.Image:
    canvas = Image.new("RGB", (1700, 980), "#f5f3ef")
    drawer = ImageDraw.Draw(canvas)
    drawer.text((30, 20), title, fill="black")
    drawer.text((30, 70), "Query", fill="black")
    drawer.text((30, 500), "Predicted Detection", fill="black")
    drawer.text((830, 70), "Top-5 Retrievals", fill="black")

    query_tile = resize_for_tile(query_image, (720, 380))
    det_tile = resize_for_tile(detected_image, (720, 380))
    canvas.paste(query_tile, (30, 100))
    canvas.paste(det_tile, (30, 530))

    tile_w, tile_h = 240, 180
    start_x, start_y = 830, 110
    gap_x, gap_y = 24, 26
    for index, (image, is_correct, label) in enumerate(retrieved_images):
        x = start_x + (index % 2) * (tile_w + gap_x)
        y = start_y + (index // 2) * (tile_h + 60 + gap_y)
        tile = resize_for_tile(image, (tile_w, tile_h))
        border = "#198754" if is_correct else "#c92a2a"
        tile = ImageOps.expand(tile, border=6, fill=border)
        canvas.paste(tile, (x, y))
        drawer.text((x, y + tile_h + 18), label, fill="black")
    return canvas


def build_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
    cols = 2
    tile_size = (850, 490)
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_size[0], rows * tile_size[1]), "#eae6df")
    for index, image in enumerate(images):
        tile = resize_for_tile(image, tile_size)
        x = (index % cols) * tile_size[0]
        y = (index // cols) * tile_size[1]
        canvas.paste(tile, (x, y))
    canvas.save(output_path, quality=92)


def load_rankings(rankings_path: Path) -> dict[str, list[str]]:
    if not rankings_path.exists():
        return {}
    return json.loads(rankings_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Text detection baseline for the BJTU course experiment.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detection"))
    parser.add_argument("--retrieval-rankings", type=Path, default=Path("outputs/retrieval/top60_rankings.json"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    detection_root = args.dataset_root / "object_detection" / "data"
    query_dir = args.dataset_root / "image_retrieval" / "query"
    retrieval_base_dir = args.dataset_root / "image_retrieval" / "base" / "BJTU"

    # 每个 landmark 留 2 张图做最终展示样例，保证 24 张面板覆盖所有类别。
    holdout_names = set(grouped_holdout_query_names(query_dir, count_per_prefix=2))
    all_records = load_detection_records(detection_root)
    # 训练集和验证集按 landmark 分层切分，避免同一类样本泄漏到验证侧。
    train_records, val_records, holdout_records = split_records_by_prefix(
        records=all_records,
        holdout_names=holdout_names,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(
        json.dumps(
            {
                "train_records": len(train_records),
                "val_records": len(val_records),
                "holdout_records": len(holdout_records),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2).to(device)

    train_loader = DataLoader(
        DetectionDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_detection,
    )

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 2, 1), gamma=0.2)

    history = []
    best_state = None
    best_val_f1 = -1.0
    best_predictions = {}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_metrics, _ = evaluate_predictions(
            model=model,
            records=val_records,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )
        holdout_metrics, holdout_predictions = evaluate_predictions(
            model=model,
            records=holdout_records,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall,
                "val_f1": val_metrics.f1,
                "val_mean_iou": val_metrics.mean_iou,
                "holdout_precision": holdout_metrics.precision,
                "holdout_recall": holdout_metrics.recall,
                "holdout_f1": holdout_metrics.f1,
                "holdout_mean_iou": holdout_metrics.mean_iou,
            }
        )
        # 用验证集 F1 选择最佳权重，避免只看训练损失导致过拟合。
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            best_predictions = holdout_predictions

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_f1": val_metrics.f1,
                    "holdout_f1": holdout_metrics.f1,
                },
                ensure_ascii=False,
            )
        )

    if best_state is None:
        raise RuntimeError("No valid detection checkpoint was produced.")
    model.load_state_dict(best_state)

    # 训练结束后，用最佳权重重新在验证集和展示集上评估一次，保证最终结果一致。
    final_val_metrics, _ = evaluate_predictions(
        model=model,
        records=val_records,
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
    )
    final_holdout_metrics, holdout_predictions = evaluate_predictions(
        model=model,
        records=holdout_records,
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
    )

    torch.save(best_state, output_dir / "best_detector.pt")
    plot_training_curve(history, output_dir / "training_curve.png")

    metrics_payload = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "val_ratio": args.val_ratio,
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
        },
        "splits": {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "holdout_records": len(holdout_records),
        },
        "history": history,
        "best_val_f1": best_val_f1,
        "final_val_metrics": vars(final_val_metrics),
        "final_holdout_metrics": vars(final_holdout_metrics),
    }
    (output_dir / "detection_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "holdout_predictions.json").write_text(
        json.dumps(holdout_predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 将检测框和检索结果拼成统一面板，方便报告和展示直接使用。
    rankings = load_rankings(args.retrieval_rankings)
    visual_dir = ensure_dir(output_dir / "visualizations")
    panel_paths = []
    for record in holdout_records:
        query_image = safe_open_image(record.image_path)
        if query_image is None:
            continue
        prediction = holdout_predictions.get(record.image_path.name, {})
        pred_boxes = prediction.get("pred_boxes", [])
        detected_image = draw_boxes(query_image, pred_boxes, color="#d94801", width=5)

        retrieved_images = []
        for rank, retrieved_name in enumerate(rankings.get(record.image_path.name, [])[:5], start=1):
            image = safe_open_image(retrieval_base_dir / retrieved_name)
            if image is None:
                continue
            is_correct = image_prefix(retrieved_name) == record.prefix
            retrieved_images.append((image, is_correct, f"#{rank} {retrieved_name}"))

        title = f"{record.prefix.upper()} | {record.image_path.name}"
        panel = compose_visual_panel(query_image, detected_image, retrieved_images, title)
        panel_path = visual_dir / f"{record.prefix}_{record.image_path.stem}.jpg"
        panel.save(panel_path, quality=92)
        panel_paths.append(panel_path)

    build_contact_sheet(sorted(panel_paths), output_dir / "visualizations_contact_sheet.jpg")

    print(json.dumps(metrics_payload["final_holdout_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
