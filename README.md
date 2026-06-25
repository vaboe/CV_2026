# 计算机视觉基础实验二说明

本文件夹对应课程实验二，内容围绕两个任务展开：

1. 图像检索
2. 文字检测

目录中已经包含数据集、源码、Notebook 和运行结果，可以直接在 Anaconda / Jupyter 环境中打开并运行。

## 目录说明

- `dataset/`：实验二所需数据集（文件过大因此未放入，运行时需创建该文件夹并将数据放入）
- `src/`：核心 Python 源码
- `experiment2_full_pipeline.ipynb`：完整运行版 Notebook，可从数据检查、训练、评估一直运行到结果生成
- `outputs/`：实验输出结果目录
- `README.md`：本说明文件

## 环境说明

建议使用本机已有的 Anaconda Python 环境运行。

已使用的核心依赖包括：

- Python 3.13.5
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- Jupyter Notebook

如果你的环境与当前机器一致，可直接使用：

```powershell
& 'D:\Anaconda\python.exe' --version
```

## 运行方式

### 方式一：运行完整 Notebook

如果你需要的是“从训练到测试都在一个 Notebook 里完成”的版本，推荐直接打开：

- `experiment2_full_pipeline.ipynb`

使用步骤：

1. 打开 Anaconda Prompt 或 Jupyter。
2. 打开 `experiment2_full_pipeline.ipynb`。
3. 从上到下依次运行所有单元。

这个 Notebook 会完成：

- 数据检查
- 图像检索特征提取与评估
- 文字检测训练
- 验证集与展示集评估
- 可视化结果生成

结果默认保存到：

- `outputs/notebook_full_pipeline/`

### 方式二：直接运行 Python 脚本

如果希望不通过 Notebook，而是在命令行中分别运行两个任务，可以在 `实验二` 目录下执行：

```powershell
& 'D:\Anaconda\python.exe' .\src\retrieval_pipeline.py
& 'D:\Anaconda\python.exe' .\src\detection_pipeline.py
```

说明：

- 第一个脚本负责图像检索
- 第二个脚本负责文字检测
- 文字检测脚本会读取检索结果文件，用于后续联合可视化展示

## 输出结果说明

### 1. 图像检索结果

目录：`outputs/retrieval/`

主要文件：

- `retrieval_metrics.json`：整体与各类别的 `P@20 / P@40 / P@60`
- `per_class_precision.csv`：各类 landmark 的精度统计表
- `query_rankings.csv`：每张查询图对应的 Top-60 检索结果
- `top60_rankings.json`：检索结果的 JSON 版本
- `precision_curves.png`：12 个 landmark 的 Precision 曲线图
- `corrupt_images.json`：运行时自动跳过的坏图记录

### 2. 文字检测结果

目录：`outputs/detection/`

主要文件：

- `detection_metrics.json`：训练配置、训练历史、验证集指标、展示集指标
- `best_detector.pt`：保存的最佳检测模型参数
- `training_curve.png`：训练损失与验证 F1 曲线
- `holdout_predictions.json`：展示样例的预测结果
- `visualizations/`：单张“检索 + 检测”面板图
- `visualizations_contact_sheet.jpg`：24 组结果的总览拼图

### 3. 完整 Notebook 输出

目录：`outputs/notebook_full_pipeline/`

这个目录保存的是 `experiment2_full_pipeline.ipynb` 从头完整运行后得到的一套单独结果，便于和已有结果区分。

## 代码说明

`src/` 中的两个主文件分别为：

- `retrieval_pipeline.py`：图像检索主程序，基于预训练 `ResNet18` 提取全局特征，并使用余弦相似度排序
- `detection_pipeline.py`：文字检测主程序，基于预训练 `Faster R-CNN MobileNetV3 320 FPN` 进行微调、评估和联合可视化

## 注意事项

- 如果重新训练文字检测模型，运行时间会明显长于图像检索部分。
- 如果某些图片损坏，程序会自动跳过，并记录到结果文件中，不影响整体运行。
