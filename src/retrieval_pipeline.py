from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from common import ensure_dir, image_prefix, list_images, safe_open_image


class RetrievalDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = safe_open_image(path)
        # 读图失败时返回 None，后面的 collate 会统一过滤掉损坏样本。
        if image is None:
            return None
        return self.transform(image), path.name


def collate_valid(batch):
    # DataLoader 里把 None 样本剔除，避免单张损坏图片中断整批推理。
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    images = torch.stack([item[0] for item in batch], dim=0)
    names = [item[1] for item in batch]
    return images, names


def extract_features(
    model: nn.Module,
    paths: list[Path],
    transform,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[str]]:
    dataset = RetrievalDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_valid,
    )

    features = []
    names: list[str] = []
    corrupt: list[str] = []
    seen = set()
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            images, batch_names = batch
            images = images.to(device)
            # 只做前向推理，不计算梯度；并在特征上做 L2 归一化，方便后面算余弦相似度。
            batch_features = F.normalize(model(images), dim=1)
            features.append(batch_features.cpu())
            names.extend(batch_names)
            seen.update(batch_names)

    # 没有在“成功读取样本集合”里出现的文件名，视为坏图并记录下来。
    corrupt.extend(path.name for path in paths if path.name not in seen)
    return torch.cat(features, dim=0), names, corrupt


def plot_precision_curves(curves: dict[str, list[float]], output_path: Path) -> None:
    classes = sorted(curves)
    cols = 3
    rows = (len(classes) + cols - 1) // cols
    figure, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    ks = np.arange(1, 61)
    for axis in axes.flat:
        axis.axis("off")

    for axis, prefix in zip(axes.flat, classes):
        axis.axis("on")
        axis.plot(ks, curves[prefix], color="#0a5c8b", linewidth=2)
        axis.scatter([20, 40, 60], [curves[prefix][19], curves[prefix][39], curves[prefix][59]], color="#d94801", s=28)
        axis.set_title(prefix.upper())
        axis.set_xlim(1, 60)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.set_xlabel("K")
        axis.set_ylabel("Precision")

    figure.suptitle("Precision@K Curves by Landmark", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Image retrieval baseline for the BJTU course experiment.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    retrieval_root = args.dataset_root / "image_retrieval"
    base_dir = retrieval_root / "base" / "BJTU"
    query_dir = retrieval_root / "query"

    base_paths = list_images(base_dir)
    query_paths = list_images(query_dir)

    weights = ResNet18_Weights.DEFAULT
    # 直接使用 ImageNet 预训练 ResNet18 做全局特征提取，不再额外训练检索模型。
    model = resnet18(weights=weights)
    model.fc = nn.Identity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = weights.transforms()

    base_features, base_names, base_corrupt = extract_features(
        model=model,
        paths=base_paths,
        transform=transform,
        batch_size=args.batch_size,
        device=device,
    )
    query_features, query_names, query_corrupt = extract_features(
        model=model,
        paths=query_paths,
        transform=transform,
        batch_size=args.batch_size,
        device=device,
    )

    similarities = query_features @ base_features.T
    # 余弦相似度矩阵按行排序，就得到每个查询图对应的 Top-K 检索结果。
    ranked_indices = similarities.argsort(dim=1, descending=True)

    ks = (20, 40, 60)
    per_class_scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: {k: [] for k in ks})
    precision_curves: dict[str, list[list[float]]] = defaultdict(list)
    top60_rankings: dict[str, list[str]] = {}

    for row, query_name in enumerate(query_names):
        prefix = image_prefix(query_name)
        indices = ranked_indices[row, :60].tolist()
        ranked_names = [base_names[index] for index in indices]
        top60_rankings[query_name] = ranked_names
        hits = np.array([1.0 if image_prefix(name) == prefix else 0.0 for name in ranked_names], dtype=np.float32)
        cumulative = np.cumsum(hits) / np.arange(1, len(hits) + 1, dtype=np.float32)
        precision_curves[prefix].append(cumulative.tolist())
        for k in ks:
            per_class_scores[prefix][k].append(float(hits[:k].mean()))

    averaged_curves = {
        prefix: np.mean(np.array(curves, dtype=np.float32), axis=0).tolist()
        for prefix, curves in precision_curves.items()
    }

    summary_rows = []
    overall_scores = {k: [] for k in ks}
    for prefix in sorted(per_class_scores):
        row = {"prefix": prefix}
        for k in ks:
            score = float(np.mean(per_class_scores[prefix][k]))
            row[f"P@{k}"] = score
            overall_scores[k].extend(per_class_scores[prefix][k])
        summary_rows.append(row)

    metrics = {
        "model": "resnet18-pretrained",
        "usable_base_images": len(base_names),
        "usable_query_images": len(query_names),
        "skipped_base_images": base_corrupt,
        "skipped_query_images": query_corrupt,
        "overall": {f"P@{k}": float(np.mean(overall_scores[k])) for k in ks},
        "per_class": summary_rows,
    }

    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "top60_rankings.json").write_text(
        json.dumps(top60_rankings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "corrupt_images.json").write_text(
        json.dumps({"base": base_corrupt, "query": query_corrupt}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (output_dir / "per_class_precision.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prefix", "P@20", "P@40", "P@60"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "query_rankings.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_name", "query_prefix", "rank", "retrieved_name", "retrieved_prefix", "similarity", "is_relevant"])
        for row, query_name in enumerate(query_names):
            prefix = image_prefix(query_name)
            for rank in range(60):
                base_index = ranked_indices[row, rank].item()
                retrieved_name = base_names[base_index]
                writer.writerow(
                    [
                        query_name,
                        prefix,
                        rank + 1,
                        retrieved_name,
                        image_prefix(retrieved_name),
                        float(similarities[row, base_index].item()),
                        int(image_prefix(retrieved_name) == prefix),
                    ]
                )

    # 把指标、排名和坏图记录分别落盘，方便 Notebook 和 Word 报告直接复用。
    plot_precision_curves(averaged_curves, output_dir / "precision_curves.png")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
