# SealScriptOCR

SealScriptOCR 是一个基于 PyTorch 的单字图像分类项目，面向小篆、古文字或其他单字符图像识别任务。项目提供从数据集检查、标签映射生成、模型训练、评估曲线保存到单张图片推理的完整最小闭环。

当前实现聚焦“单张图像 -> 单个类别标签”的分类问题，不包含文本检测、整行 OCR、多字符序列识别或图形界面。

## Highlights

- **自定义字符数据集**：使用 `torchvision.datasets.ImageFolder` 读取 `train/val/test` 目录。
- **古文字预处理**：灰度化、自动对比度、简单中值滤波、等比例缩放并居中填充。
- **两种模型骨干**：轻量 `SimpleCNN` 与单通道输入版 `ResNet18`。
- **训练闭环完整**：训练、验证、最佳权重保存、Loss/Accuracy 曲线输出。
- **标签可追踪**：训练时自动生成 `label_map.json`，推理时显示真实类别名。
- **脚本简单透明**：核心参数集中在 `src/config.py`，适合教学、实验和二次开发。

## Project Structure

```text
SealScriptOCR/
├── data/                       # 数据集根目录，需包含 train/val/test
├── checkpoints/                # 训练权重输出目录
├── outputs/                    # 训练曲线等输出文件
├── scripts/
│   ├── check_dataset.py        # 检查各类别样本数量
│   └── gen_label_map.py        # 根据 train 目录生成标签映射
├── src/
│   ├── config.py               # 训练与推理配置
│   ├── train.py                # 训练入口
│   ├── infer.py                # 单图推理入口
│   ├── datasets/
│   │   └── mnist_dataset.py    # MNIST 与自定义字符数据加载
│   ├── models/
│   │   ├── simple_cnn.py
│   │   └── resnet18_mnist.py
│   └── utils/
│       ├── seed.py
│       └── train_eval.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- Pillow
- matplotlib
- numpy
- scikit-learn
- tqdm

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要 GPU，请根据你的 CUDA 版本先安装匹配的 PyTorch，再安装其余依赖。详见 PyTorch 官方安装说明。

## Dataset

自定义数据集采用 ImageFolder 目录结构。目录名即类别名：

```text
data/
├── train/
│   ├── 日/
│   │   ├── 0001.png
│   │   └── 0002.png
│   └── 月/
│       └── 0001.png
├── val/
│   ├── 日/
│   └── 月/
└── test/
    ├── 日/
    └── 月/
```

支持的常见图片格式包括 `png`、`jpg`、`jpeg`、`bmp`、`webp`。

建议：

- 每个 split 中保持类别集合一致。
- 每类尽量准备足够样本，少样本类别容易过拟合。
- 单张图片最好只包含一个字符主体。
- 背景、尺寸和笔画颜色越稳定，训练越容易收敛。

检查数据集数量：

```bash
python scripts/check_dataset.py
```

手动生成标签映射：

```bash
python scripts/gen_label_map.py
```

训练脚本也会自动在 `data/label_map.json` 写入标签映射。

## Configuration

主要参数位于 `src/config.py`：

```python
class Config:
    seed = 42
    batch_size = 512
    lr = 1e-3
    epochs = 100
    data_root = "data"
    num_classes = 4
    image_size = 28
    model_name = "simple_cnn"  # simple_cnn or resnet18
    num_workers = 0
    save_path = "checkpoints/best_model.pth"
```

常用修改项：

- `data_root`：数据集根目录，默认读取 `data/train`、`data/val`、`data/test`。
- `model_name`：选择 `simple_cnn` 或 `resnet18`。
- `image_size`：输入图片尺寸。
- `batch_size` / `epochs` / `lr`：训练超参数。
- `save_path`：最佳模型权重保存位置。

注意：当前 `SimpleCNN` 的全连接层按 `image_size = 28` 设计。如果需要使用其他输入尺寸，建议选择 `resnet18`，或同步调整 `src/models/simple_cnn.py` 中的分类头尺寸。

## Training

确认 `data/train`、`data/val`、`data/test` 准备完成后运行：

```bash
python src/train.py
```

训练过程会：

1. 读取 `src/config.py` 配置。
2. 加载自定义字符数据集。
3. 根据 `train` 目录生成 `data/label_map.json`。
4. 按验证集准确率保存最佳模型到 `checkpoints/best_model.pth`。
5. 输出训练曲线到 `outputs/training_plot.png`。

训练完成后，你应该看到类似输出：

```text
Training finished. Best val acc: 0.9500. Model saved to checkpoints/best_model.pth
```

## Inference

使用训练好的权重对单张图片推理：

```bash
python src/infer.py --image path/to/image.png
```

推理脚本会读取：

- 模型结构：`Config.model_name`
- 权重路径：`Config.save_path`
- 标签映射：`Config.data_root/label_map.json`

输出包含预测类别和每个类别的概率。

如果不传入 `--image`，脚本会从配置的数据集测试集中取第一个样本做演示推理。

## Model Notes

### SimpleCNN

轻量卷积网络，适合快速验证流程和小尺寸输入。默认输入为单通道 `28 x 28` 图片。

### ResNet18

基于 torchvision ResNet18，第一层卷积已改为支持单通道输入，分类头会根据数据集类别数重建。适合更复杂或更大尺寸的字符图像实验。

## Outputs

默认输出文件：

```text
checkpoints/best_model.pth     # 验证集准确率最高的模型权重
outputs/training_plot.png      # Loss 与 Accuracy 曲线
data/label_map.json            # 类别名到类别索引的映射
```

`label_map.json` 示例：

```json
{
  "日": 0,
  "月": 1,
  "山": 2,
  "水": 3
}
```

## Development Roadmap

- 增加命令行参数，减少对 `src/config.py` 的手动修改。
- 增加测试集评估入口与混淆矩阵输出。
- 增加数据增强策略，如随机旋转、平移、缩放和轻微噪声。
- 支持批量推理与 CSV 结果导出。
- 为不同输入尺寸提供更稳健的 CNN 分类头。

## Troubleshooting

### 未找到 `data/train`

请确认 `Config.data_root` 指向的数据目录包含 `train`、`val`、`test` 三个子目录。

### 推理类别显示为数字

通常是缺少 `label_map.json`。先运行训练脚本，或执行：

```bash
python scripts/gen_label_map.py
```

### 修改 `image_size` 后 `SimpleCNN` 报维度错误

`SimpleCNN` 当前按 `28 x 28` 输入设计。请将 `image_size` 改回 `28`，或改用 `model_name = "resnet18"`。

### 训练集准确率高但验证集低

常见原因是样本过少、类别不平衡、训练轮数过多或训练/验证分布不一致。优先检查每类样本数量，并增加数据增强或补充样本。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
