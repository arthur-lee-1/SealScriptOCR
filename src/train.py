import os
import torch
from config import Config
from utils.seed import set_seed
from datasets.mnist_dataset import get_custom_chars_loaders, get_mnist_loaders
from models.simple_cnn import SimpleCNN
from models.resnet18_mnist import resnet18_single_channel
from utils.train_eval import train_one_epoch, evaluate, save_model, plot_metrics


def main():
	cfg = Config
	set_seed(cfg.seed)
	device = torch.device(cfg.device)
	# 强制使用自定义古汉字数据集（ImageFolder 结构）
	if not (os.path.isdir(cfg.data_root) and os.path.isdir(os.path.join(cfg.data_root, 'train'))):
		print(f"错误：未找到自定义数据集目录: {cfg.data_root}/train")
		print("请按项目约定准备数据：data_root/train/<class>/*.png, data_root/val/<class>/*.png, data_root/test/<class>/*.png")
		return

	train_loader, val_loader, test_loader, label_map = get_custom_chars_loaders(
		data_root=cfg.data_root,
		batch_size=cfg.batch_size,
		image_size=cfg.image_size,
		num_workers=cfg.num_workers,
	)
	cfg.num_classes = len(label_map)
	eval_loader = val_loader

	model = SimpleCNN(num_classes=cfg.num_classes).to(device)
	# 选择模型
	if cfg.model_name == "resnet18":
		print(f"使用模型: ResNet18，类别数: {cfg.num_classes}")
		model = resnet18_single_channel(num_classes=cfg.num_classes, pretrained=False).to(device)
	else:
		print(f"使用模型: SimpleCNN，类别数: {cfg.num_classes}")
		model = SimpleCNN(num_classes=cfg.num_classes).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

	print(f"开始训练 — 数据模式: {cfg.dataset_mode}，设备: {cfg.device}，批大小: {cfg.batch_size}，轮数: {cfg.epochs}")

	history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
	best_acc = 0.0

	for epoch in range(1, cfg.epochs + 1):
		print(f"第 {epoch}/{cfg.epochs} 轮开始...")
		train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
		val_loss, val_acc = evaluate(model, eval_loader, device)

		history['train_loss'].append(train_loss)
		history['val_loss'].append(val_loss)
		history['train_acc'].append(train_acc)
		history['val_acc'].append(val_acc)

		print(f"第{epoch}轮结果 — 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

		if val_acc > best_acc:
			best_acc = val_acc
			save_model(model, cfg.save_path)
			print(f"已保存最佳模型，验证准确率: {best_acc:.4f} -> {cfg.save_path}")

	plot_metrics(history)
	print(f"Training finished. Best val acc: {best_acc:.4f}. Model saved to {cfg.save_path}")


if __name__ == '__main__':
	main()

