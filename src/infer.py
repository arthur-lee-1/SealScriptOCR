import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms

from config import Config
from datasets.mnist_dataset import AncientCharPreprocess, get_custom_chars_loaders
from models.resnet18_mnist import resnet18_single_channel
from models.simple_cnn import SimpleCNN


def build_model(model_name, num_classes, device):
	if model_name == "resnet18":
		return resnet18_single_channel(num_classes=num_classes, pretrained=False).to(device)

	if model_name == "simple_cnn":
		return SimpleCNN(num_classes=num_classes).to(device)

	options = ", ".join(Config.model_save_paths)
	raise ValueError(f"未知模型名称：{model_name}。可选项：{options}")


def load_image(path, image_size):
	img = Image.open(path)
	transform = transforms.Compose([
		AncientCharPreprocess(image_size=image_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
	return transform(img).unsqueeze(0)


def load_idx_to_class(data_root):
	label_map_path = os.path.join(data_root, "label_map.json")
	if os.path.exists(label_map_path):
		with open(label_map_path, "r", encoding="utf-8") as f:
			class_to_idx = json.load(f)
		return {int(v): k for k, v in class_to_idx.items()}
	return None


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--image", type=str, help="输入图片路径")
	parser.add_argument(
		"--model",
		choices=("simple_cnn", "resnet18"),
		default=Config.model_name,
		help="推理使用的模型，会覆盖 Config.model_name。",
	)
	return parser.parse_args()


def print_probs(probs, idx_to_class):
	for i, p in enumerate(probs.squeeze().tolist()):
		label = idx_to_class.get(i, str(i)) if idx_to_class else str(i)
		print(f"  {label}: {p:.4f}")


def main():
	args = parse_args()
	cfg = Config
	model_name = args.model
	save_path = cfg.get_save_path(model_name)
	device = torch.device(cfg.device)

	idx_to_class = load_idx_to_class(cfg.data_root)
	if idx_to_class is None:
		print(f"警告：未在 {cfg.data_root} 找到 label_map.json。请先运行训练脚本或 scripts/gen_label_map.py。")
		num_classes = cfg.num_classes
	else:
		num_classes = len(idx_to_class)

	if not os.path.exists(save_path):
		print(f"错误：未找到权重文件：{save_path}")
		print(f"请先训练该模型：python src/train.py --model {model_name}")
		return

	print(f"正在加载模型：{model_name}，权重路径：{save_path}")
	model = build_model(model_name, num_classes, device)
	model.load_state_dict(torch.load(save_path, map_location=device))
	model.eval()

	if args.image:
		print(f"正在推理图片：{args.image}")
		x = load_image(args.image, cfg.image_size).to(device)
		with torch.no_grad():
			logits = model(x)
			probs = torch.softmax(logits, dim=1)
			pred = probs.argmax(dim=1).item()

		pred_label = idx_to_class.get(pred, str(pred)) if idx_to_class else str(pred)
		print(f"预测结果：{pred_label}")
		print("各类别概率：")
		print_probs(probs, idx_to_class)
		return

	_, _, test_loader, _ = get_custom_chars_loaders(
		data_root=cfg.data_root,
		batch_size=1,
		image_size=cfg.image_size,
	)
	x, y = next(iter(test_loader))
	x = x.to(device)
	with torch.no_grad():
		logits = model(x)
		probs = torch.softmax(logits, dim=1)
		pred = probs.argmax(dim=1).item()

	true_label = idx_to_class.get(int(y.item()), str(int(y.item()))) if idx_to_class else str(int(y.item()))
	pred_label = idx_to_class.get(pred, str(pred)) if idx_to_class else str(pred)
	print(f"示例样本：真实标签={true_label}，预测结果={pred_label}")
	print("各类别概率：")
	print_probs(probs, idx_to_class)


if __name__ == "__main__":
	main()
