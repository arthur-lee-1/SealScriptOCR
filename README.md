# 基于深度学习的小篆汉字图像识别系统（CNN分类）

本项目面向“小篆汉字图像识别”任务，采用卷积神经网络（CNN）将小篆字符识别转化为**多类别图像分类**问题，提供从数据集组织、训练评估到模型保存与推理的完整流程，并支持命令行进行单图识别（可扩展简单界面）。

---

## 1. 功能特性

- 支持不少于 20 类小篆汉字的图像分类（类别数可扩展）
- 数据预处理：尺寸统一、灰度化/归一化（训练阶段自动完成）
- 数据增强：随机旋转、平移、缩放、噪声等（可配置）
- 模型训练：交叉熵损失（CrossEntropyLoss）、Accuracy评估
- 结果分析：支持输出混淆矩阵、训练/验证曲线（可选）
- 部署推理：命令行输入图片路径进行识别，输出Top-1/Top-k结果

---

## 2. 环境要求

- Python >= 3.9（建议 3.10/3.11）
- PyTorch >= 2.0
- torchvision
- numpy, pandas
- opencv-python 或 pillow
- matplotlib（可选：画曲线）
- scikit-learn（可选：混淆矩阵）

---

## 3. 安装与运行

### 3.1 克隆项目

```bash
git clone <your-repo-url>.git
cd seal-script-recognition
```

### 3.2 创建虚拟环境（推荐）

```bash
conda create -n sealscript python=3.10 -y
conda activate sealscript
```

### 3.3 安装依赖

方式A：使用 requirements.txt（推荐）

```bash
pip install -r requirements.txt
```

方式B：手动安装

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib scikit-learn
```

---

## 4. 数据集准备（重要）

### 4.1 数据规模要求（项目要求）

- 类别数：不少于 **20 类**
- 每类样本：不少于 **20 张**
- 图片可来自碑刻、拓片、字典扫描、截图等
- 允许存在一定噪声，但建议尽量清晰可辨

### 4.2 数据组织结构（推荐：ImageFolder风格）

将数据按类别文件夹存放（文件夹名=类别标签），示例：

```
data/
  sealscript_20/
    train/
      王/
        0001.png
        0002.png
      山/
        0001.png
    val/
      王/
      山/
    test/
      王/
      山/
```

> 说明

- 若你没有单独的 `val/test`，代码也可支持从 `train` 中按比例划分（取决于你的实现）。
- 类别名可以是汉字（如“山”“王”），也可以是编码（如 `shan`、`wang`），但需保持一致。

### 4.3 图片规范建议

- 格式：png/jpg均可
- 建议背景尽量统一（白底黑字或黑底白字）
- 若底色不一致，可在预处理里做二值化/反色（可选）

---

## 5. 项目结构（示例）

你可以按以下结构组织代码（README可直接配合此结构使用）：

```
.
├── data/
│   └── sealscript_20/
│       ├── train/
│       ├── val/
│       └── test/
├── checkpoints/
│   └── best.pth
├── outputs/
│   ├── logs/
│   ├── curves.png
│   └── confusion_matrix.png
├── src/
│   ├── train.py
│   ├── infer.py
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

> 如果你当前项目结构不同，只要将下面的命令中的路径改为你的实际路径即可。

---

## 6. 训练（Training）

### 6.1 基本训练命令

```bash
python src/train.py \
  --data_dir data/sealscript_20 \
  --img_size 128 \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-3 \
  --num_workers 4 \
  --model resnet18 \
  --save_dir checkpoints
```

### 6.2 常用参数说明

- `--data_dir`：数据集根目录（包含 train/val/test）
- `--img_size`：输入图片尺寸（统一缩放到 img_size × img_size）
- `--model`：模型类型（如 `lenet` / `simple_cnn` / `resnet18`）
- `--epochs`：训练轮数
- `--lr`：学习率
- `--save_dir`：模型保存目录（保存 best.pth / last.pth 等）
- `--augment`：开启数据增强（若实现支持）
- `--seed`：随机种子，保证可复现

---

## 7. 评估（Evaluation）

### 7.1 在测试集上评估准确率

```bash
python src/train.py \
  --data_dir data/sealscript_20 \
  --eval_only \
  --ckpt checkpoints/best.pth
```

### 7.2 混淆矩阵（可选）

如果代码支持：

```bash
python src/train.py \
  --data_dir data/sealscript_20 \
  --eval_only \
  --ckpt checkpoints/best.pth \
  --plot_cm \
  --cm_out outputs/confusion_matrix.png
```

---

## 8. 推理（Inference / Deploy）

### 8.1 单张图片识别（命令行）

```bash
python src/infer.py \
  --ckpt checkpoints/best.pth \
  --img_path demo/unknown.png \
  --img_size 128 \
  --topk 5
```

输出示例：

```
Top-1: 山 (0.83)
Top-5:
  山 0.83
  川 0.07
  水 0.03
  ...
```

### 8.2 批量图片识别（可选）

```bash
python src/infer.py \
  --ckpt checkpoints/best.pth \
  --img_dir demo/images \
  --out_csv outputs/predictions.csv
```

---

## 9. 结果分析建议（写论文/报告可用）

常见错误来源：

1. **类间字形相似**：如某些部件/笔画结构高度相似，模型易混淆
2. **样本不足导致过拟合**：每类样本过少时，训练集准确率高但测试集下降
3. **噪声与形变**：拓片裂纹、墨迹不均、倾斜拉伸、背景纹理
4. **数据分布不一致**：训练集与测试集来源不同（不同书写风格/扫描质量）

改进方向：

- 扩充数据集、做更强的数据增强
- 使用更强的骨干网络（ResNet34/50、EfficientNet等）
- 引入度量学习/对比学习以提高细粒度区分能力
- 做更专业的预处理（反色、二值化、去噪、倾斜校正、形态学操作）

---

## 10. 常见问题（FAQ）

### Q1：为什么训练准确率很高但测试准确率低？

A：数据量小、增强不足或训练轮数过多导致过拟合。建议：

- 增加数据增强
- 减少模型复杂度或使用正则化（Dropout/Weight Decay）
- 使用早停（Early Stopping）
- 增加每类样本数量

### Q2：类别文件夹是中文会不会有问题？

A：一般不会。若在Windows环境出现编码/路径问题，可改为拼音或数字标签。

### Q3：必须灰度图吗？

A：不必须。小篆多为黑白图，灰度可降低冗余信息；若使用ResNet等预训练模型，常见做法是复制灰度到3通道或直接读RGB并统一处理。

---

## 11. 参考实现建议（你可以在代码里体现）

- 损失函数：`torch.nn.CrossEntropyLoss`
- 指标：Accuracy（Top-1），可选 Top-k
- 数据增强（训练集）：RandomRotation / RandomAffine / ColorJitter（谨慎）/ RandomInvert（视数据而定）
- 学习率调度：StepLR / CosineAnnealingLR（可选）
- 保存最优模型：按 val accuracy 最高保存 `best.pth`

---

## 12. License

MIT License

---

## 13. 致谢

- PyTorch / torchvision
- OpenCV / PIL
- 相关古文字资料来源与整理者（按你的数据来源补充）
