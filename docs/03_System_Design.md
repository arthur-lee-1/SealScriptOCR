# 系统设计

## 1. 总体架构

本系统采用模块化的深度学习图像分类工程结构。

系统整体流程如下：

    数据准备
        ↓
    数据加载
        ↓
    模型构建
        ↓
    模型训练
        ↓
    模型评估
        ↓
    模型保存
        ↓
    单图推理
        ↓
    结果可视化

---

## 2. 项目结构

    single_char_recognition/
    ├── data/
    ├── checkpoints/
    ├── outputs/
    ├── docs/
    ├── src/
    │   ├── datasets/
    │   │   └── mnist_dataset.py
    │   ├── models/
    │   │   ├── simple_cnn.py
    │   │   └── resnet18_mnist.py
    │   ├── utils/
    │   │   ├── train_eval.py
    │   │   ├── metrics.py
    │   │   ├── visualize.py
    │   │   └── seed.py
    │   ├── config.py
    │   ├── train.py
    │   ├── infer.py
    │   └── main.py
    ├── requirements.txt
    └── README.md

---

## 3. 模块设计

### 3.1 配置模块

**文件：** `src/config.py`

该模块用于存放项目中的统一配置项，例如：

- batch size
- 学习率
- epoch 数
- 运行设备
- 图像尺寸
- 类别数
- 模型保存路径
- 随机种子

**作用：**

- 集中管理超参数
- 避免硬编码
- 简化后续数据集替换

---

### 3.2 数据集模块

**文件：** `src/datasets/mnist_dataset.py`

该模块负责：

- 下载 MNIST
- 应用图像变换
- 构建训练和测试 DataLoader

**当前预处理流程：**

- `ToTensor()`
- `Normalize((0.1307,), (0.3081,))`

**未来扩展：**

后续可以替换为：

- `ImageFolder`
- 面向小篆单字图像的自定义 Dataset 类

---

### 3.3 模型模块

#### 3.3.1 SimpleCNN

**文件：** `src/models/simple_cnn.py`

这是一个轻量级 CNN，用于快速验证训练流程。

推荐结构：

    Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Fully Connected -> Output

**作用：**

- 结构简单，便于理解
- 易于调试
- 适合初始实验

#### 3.3.2 面向 MNIST 的 ResNet18

**文件：** `src/models/resnet18_mnist.py`

这是一个适配 MNIST 的 ResNet18 模型。

需要进行的修改：

- 将第一层卷积输入通道从 3 改为 1
- 将最后全连接层输出维度改为 10

**作用：**

- 提供更强的基线模型
- 为后续迁移到更复杂的单字数据集做准备

---

### 3.4 训练与评估模块

**文件：** `src/utils/train_eval.py`

该模块应包含可复用的训练与评估函数。

典型函数包括：

- `train_one_epoch()`
- `evaluate()`

**职责：**

- 计算训练损失
- 更新模型参数
- 在验证集或测试集上进行推理
- 统计评估指标

---

### 3.5 指标模块

**文件：** `src/utils/metrics.py`

该模块用于计算模型评估指标。

当前指标：

- Accuracy（准确率）

未来可选扩展指标：

- Precision（精确率）
- Recall（召回率）
- F1-score
- Confusion Matrix（混淆矩阵）

---

### 3.6 可视化模块

**文件：** `src/utils/visualize.py`

该模块用于生成以下图形化结果：

- 训练损失曲线
- 验证准确率曲线
- 混淆矩阵图

生成的图像建议保存在 `outputs/` 目录下。

---

### 3.7 推理模块

**文件：** `src/infer.py`

该模块用于加载训练好的模型权重，并对单张图像进行预测。

期望命令格式：

    python src/infer.py --img path/to/image.png --weights checkpoints/best_model.pth

期望输出内容：

- 预测标签
- 置信度
- 可选 top-k 结果

---

## 4. 数据流设计

训练过程中的数据流如下：

    原始 MNIST 图像
        ↓
    图像变换
        ↓
    Tensor 批数据
        ↓
    模型前向传播
        ↓
    损失计算
        ↓
    反向传播
        ↓
    优化器更新

推理过程中的数据流如下：

    单张输入图像
        ↓
    预处理
        ↓
    模型前向传播
        ↓
    Softmax / 概率计算
        ↓
    预测标签

---

## 5. 设计原则

本项目遵循以下设计原则：

1. **模块化**
   不同功能拆分为不同模块。

2. **清晰性**
   项目结构应易于阅读和理解。

3. **可扩展性**
   框架应支持后续替换数据集和模型。

4. **可复用性**
   训练、评估等核心逻辑应可复用。

5. **渐进式开发**
   从简单模型和简单数据集开始，逐步演进。

---

## 6. 面向未来迁移的设计

当前系统有意按可迁移架构设计，以便后续扩展到小篆单字识别任务。

从 MNIST 迁移到自定义单字数据集时，主要需要修改以下部分：

- 数据加载逻辑
- 类别数
- 标签映射
- 图像预处理
- 模型输出维度

而以下部分可基本保持不变：

- 训练循环
- 模型保存
- 模型评估
- 推理逻辑

