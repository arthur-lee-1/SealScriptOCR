import argparse
import os

import torch

from config import Config
from datasets.mnist_dataset import get_custom_chars_loaders
from models.resnet18_mnist import resnet18_single_channel
from models.simple_cnn import SimpleCNN
from utils.seed import set_seed
from utils.train_eval import evaluate, plot_metrics, save_model, train_one_epoch


def build_model(model_name, num_classes, device):
	if model_name == "resnet18":
		print(f"使用模型：ResNet18，类别数：{num_classes}")
		return resnet18_single_channel(num_classes=num_classes, pretrained=False).to(device)

	if model_name == "simple_cnn":
		print(f"使用模型：SimpleCNN，类别数：{num_classes}")
		return SimpleCNN(num_classes=num_classes).to(device)

	options = ", ".join(Config.model_save_paths)
	raise ValueError(f"未知模型名称：{model_name}。可选项：{options}")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--model",
		choices=("simple_cnn", "resnet18"),
		default=Config.model_name,
		help="要训练的模型，会覆盖 Config.model_name。",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	cfg = Config
	model_name = args.model
	save_path = cfg.get_save_path(model_name)
	plot_path = f"outputs/{model_name}_training_plot.png"

	set_seed(cfg.seed)
	device = torch.device(cfg.device)

	train_dir = os.path.join(cfg.data_root, "train")
	if not (os.path.isdir(cfg.data_root) and os.path.isdir(train_dir)):
		print(f"错误：未找到自定义数据集目录：{train_dir}")
		print("期望结构：data_root/train/<class>/*.png, data_root/val/<class>/*.png, data_root/test/<class>/*.png")
		return

	train_loader, val_loader, _test_loader, label_map = get_custom_chars_loaders(
		data_root=cfg.data_root,
		batch_size=cfg.batch_size,
		image_size=cfg.image_size,
		num_workers=cfg.num_workers,
	)
	num_classes = len(label_map)

	model = build_model(model_name, num_classes, device)
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

	print(
		f"开始训练：模型={model_name}，设备={cfg.device}，"
		f"批大小={cfg.batch_size}，轮数={cfg.epochs}，权重保存路径={save_path}"
	)

	history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
	best_acc = 0.0

	for epoch in range(1, cfg.epochs + 1):
		print(f"第 {epoch}/{cfg.epochs} 轮")
		train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, device)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["train_acc"].append(train_acc)
		history["val_acc"].append(val_acc)

		print(
			f"结果：训练损失={train_loss:.4f}，训练准确率={train_acc:.4f}，"
			f"验证损失={val_loss:.4f}，验证准确率={val_acc:.4f}"
		)

		if val_acc > best_acc:
			best_acc = val_acc
			save_model(model, save_path)
			print(f"已保存最佳 {model_name} 模型：验证准确率={best_acc:.4f} -> {save_path}")

	plot_metrics(history, plot_path)
	print(f"训练完成。最佳验证准确率：{best_acc:.4f}。模型已保存到：{save_path}")
	print(f"训练曲线已保存到：{plot_path}")


if __name__ == "__main__":
	main()
