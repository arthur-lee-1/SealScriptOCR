# 训练与推理指南

本文档给出从秦简牍动物字体数据到训练分类器、再到单图预测的最短可靠流程。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果需要 CUDA 训练，请先安装与你的 GPU 匹配的 PyTorch 版本。

## 2. 准备数据

使用 ImageFolder 目录结构：

```text
data/
├── train/
│   ├── 马/
│   ├── 牛/
│   └── 犬/
├── val/
│   ├── 马/
│   ├── 牛/
│   └── 犬/
└── test/
    ├── 马/
    ├── 牛/
    └── 犬/
```

每张图片应包含一个裁剪好的动物字形或动物相关字符。

检查数据集：

```bash
python scripts/check_dataset.py
```

如需手动生成标签映射：

```bash
python scripts/gen_label_map.py
```

训练脚本也会自动写入 `data/label_map.json`。

## 3. 修改配置

编辑 `src/config.py`：

```python
class Config:
    seed = 42
    batch_size = 512
    lr = 1e-3
    epochs = 100
    data_root = "data"
    image_size = 28
    model_name = "simple_cnn"
    num_workers = 0
    save_path = "checkpoints/best_model.pth"
```

推荐设置：

| 场景 | 设置 |
| --- | --- |
| 快速冒烟测试 | `simple_cnn`，`image_size = 28` |
| 较大字形裁剪图 | `resnet18`，`image_size = 64` 或 `128` |
| 仅 CPU 训练 | 降低 `batch_size` |
| 样本较少 | 减少 `epochs`，观察过拟合 |

注意：`SimpleCNN` 当前假设 `image_size = 28`。其他尺寸请使用 `resnet18`，或同步修改 `SimpleCNN` 的分类头。

## 4. 训练

```bash
python src/train.py
```

脚本会执行：

1. 加载 `data/train`、`data/val`、`data/test`。
2. 从 `train` 目录推断类别数。
3. 保存 `data/label_map.json`。
4. 训练指定模型。
5. 按验证集准确率保存最优权重。
6. 写入训练曲线。

预期输出：

```text
checkpoints/best_model.pth
outputs/training_plot.png
data/label_map.json
```

## 5. 推理

```bash
python src/infer.py --image path/to/image.png
```

脚本会使用：

- `Config.model_name`
- `Config.save_path`
- `Config.data_root`
- `label_map.json`

如果不传入 `--image`，脚本会从配置的数据集测试集中取第一张样本做演示推理。

## 6. 结果判断

良好信号：

- 训练损失下降。
- 验证集准确率上升后趋稳。
- 推理输出可读标签，而不是只有数字。
- 明显样本可以预测正确。

异常信号：

| 现象 | 可能原因 |
| --- | --- |
| loss 不下降 | 标签错误、学习率不合适、预处理异常 |
| 训练准确率高但验证准确率低 | 过拟合或数据分布不一致 |
| 所有图片都预测为同一类 | 类别不平衡或标签映射错误 |
| 推理加载模型失败 | 配置与保存的权重不匹配 |
| 标签显示为数字 | 缺少 `label_map.json` |

## 7. 实验记录建议

每次运行记录一行：

| 字段 | 示例 |
| --- | --- |
| 数据集 | `qin_animals_v1` |
| 模型 | `resnet18` |
| 输入尺寸 | `64` |
| batch_size | `128` |
| 学习率 | `1e-3` |
| 训练轮数 | `50` |
| 最佳验证准确率 | `0.842` |
| 权重路径 | `checkpoints/best_model.pth` |
| 备注 | `删除重复的牛类样本` |
