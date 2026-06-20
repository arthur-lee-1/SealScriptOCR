# 系统设计

SealScriptOCR 采用小型、模块化的 PyTorch 工程设计。每个模块只负责一件事：加载数据、预处理图像、构建模型、训练、评估或推理。

## 架构

```text
ImageFolder 数据集
      |
      v
AncientCharPreprocess
      |
      v
DataLoader
      |
      v
SimpleCNN / ResNet18
      |
      v
train_one_epoch / evaluate
      |
      v
checkpoint + training_plot + label_map
      |
      v
单图推理
```

## 仓库结构

```text
src/
├── config.py
├── train.py
├── infer.py
├── datasets/
│   └── mnist_dataset.py
├── models/
│   ├── simple_cnn.py
│   └── resnet18_mnist.py
└── utils/
    ├── seed.py
    └── train_eval.py
```

## 模块说明

### 配置模块

文件：`src/config.py`

存放训练和推理共用的实验参数：

- 数据集路径
- 输入图像尺寸
- 模型名称
- batch size
- 学习率
- 训练轮数
- 权重保存路径
- 运行设备

### 数据集模块

文件：`src/datasets/mnist_dataset.py`

主要入口：

```python
get_custom_chars_loaders(data_root, batch_size, image_size, num_workers)
```

返回内容：

- `train_loader`
- `val_loader`
- `test_loader`
- `label_map`

该加载器会将 `label_map.json` 写入 `data_root`，方便推理时把类别 ID 映射回可读标签。

### 预处理模块

类：`AncientCharPreprocess`

流程：

1. 转为灰度图。
2. 自动对比度增强。
3. 中值滤波。
4. 保持长宽比缩放。
5. 填充到正方形画布。
6. 再次自动对比度增强。
7. 转为张量并归一化到约 `[-1, 1]`。

该流程刻意保持保守。对于秦简牍动物字形，强数据增强或二值化应在人工检查图像后再加入。

### 模型模块

文件：

- `src/models/simple_cnn.py`
- `src/models/resnet18_mnist.py`

`SimpleCNN` 是最快的基线模型。它当前假设输入为 `28 x 28`，因为分类头中包含固定的 `64 * 7 * 7` 线性层输入尺寸。

`ResNet18` 将第一层卷积替换为单通道输入层，并按数据集类别数重建最后的分类层。

### 训练与评估模块

文件：`src/utils/train_eval.py`

函数：

- `train_one_epoch`
- `evaluate`
- `save_model`
- `plot_metrics`

训练使用：

- `torch.nn.functional.cross_entropy`
- Adam 优化器
- 验证集准确率作为最优权重保存依据

### 推理模块

文件：`src/infer.py`

当前命令：

```bash
python src/infer.py --image path/to/image.png
```

推理会读取：

- `Config.model_name` 中的模型结构
- `Config.save_path` 中的权重
- `Config.data_root/label_map.json` 中的标签映射

## 输出产物

| 路径 | 含义 |
| --- | --- |
| `checkpoints/best_model.pth` | 最优模型权重 |
| `outputs/training_plot.png` | 损失和准确率曲线 |
| `data/label_map.json` | 类别名到索引的映射 |

## 扩展点

推荐的后续改动：

- 用 argparse 替代纯配置文件工作流。
- 增加 `evaluate.py` 输出测试集指标。
- 增加混淆矩阵和每类准确率。
- 增加批量推理。
- 为每次实验增加元数据日志。
