# 开发流程

本文档约定项目的日常开发方式：每次改动应尽量小、可复现、易验证。

## 当前开发主线

当前开发主线是：

> 秦简牍动物字体单图分类基线。

现阶段不要优先做完整 OCR。先把单字符分类器做稳定。

## 标准流程

1. 准备或更新数据集。
2. 运行数据集检查。
3. 更新 `src/config.py`。
4. 训练一个基线模型。
5. 检查训练曲线和验证集准确率。
6. 运行单图推理。
7. 记录结果和失败样本。
8. 做下一处最小改进。

## 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux 或 macOS：

```bash
source .venv/bin/activate
```

## 数据检查

```bash
python scripts/check_dataset.py
python scripts/gen_label_map.py
```

训练前确认：

- `data/train` 存在。
- `data/val` 存在。
- `data/test` 存在。
- 类别文件夹一致。
- 每类样本数量足够。
- `label_map.json` 与预期标签一致。

## 基线训练

编辑 `src/config.py`：

```python
data_root = "data"
model_name = "simple_cnn"
image_size = 28
save_path = "checkpoints/best_model.pth"
```

运行：

```bash
python src/train.py
```

预期输出：

- `checkpoints/best_model.pth`
- `outputs/training_plot.png`
- `data/label_map.json`

## 冒烟测试

使用一张已知测试图片：

```bash
python src/infer.py --image path/to/image.png
```

输出应包含：

- 预测标签
- 每个类别的概率

## 改动策略

改进项目时推荐按以下顺序：

1. 修正数据和标签。
2. 完善评估。
3. 加入保守的数据增强。
4. 对比模型。
5. 增加自动化。

避免一次改动太多部分。如果准确率变化，应能判断原因来自数据、预处理、模型还是超参数。

## 合并前检查

提交前确认：

- 仍能在项目根目录运行 `python src/train.py`。
- 仍能运行 `python src/infer.py --image ...`。
- README 和 docs 与真实命令接口一致。
- 新输出写入 `outputs/` 或 `checkpoints/`。
- 除非明确需要，不提交生成的数据集或模型权重。

## 编码约定

- 数据集逻辑放在 `src/datasets/`。
- 模型定义放在 `src/models/`。
- 可复用训练工具放在 `src/utils/`。
- 脚本保持小而直接。
- 注释只解释不明显的行为。
- 优先保持基线清晰，而不是过早抽象。
