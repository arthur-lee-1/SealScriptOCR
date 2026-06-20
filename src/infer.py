import argparse
import json
import os
import torch
from PIL import Image
from torchvision import transforms
from config import Config
from models.simple_cnn import SimpleCNN
from models.resnet18_mnist import resnet18_single_channel
from datasets.mnist_dataset import AncientCharPreprocess, get_custom_chars_loaders, get_mnist_loaders


def load_image(path, image_size):
	img = Image.open(path)
	transform = transforms.Compose([
		AncientCharPreprocess(image_size=image_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])
	return transform(img).unsqueeze(0)


def load_idx_to_class(data_root):
	label_map_path = os.path.join(data_root, "label_map.json")
	if os.path.exists(label_map_path):
		with open(label_map_path, "r", encoding="utf-8") as f:
			class_to_idx = json.load(f)
		return {int(v): k for k, v in class_to_idx.items()}
	return None


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', type=str, help='Path to input image')
	args = parser.parse_args()
	cfg = Config
	device = torch.device(cfg.device)
	# 中文说明
	print("\n=== 推理说明 ===")
	print("这是推理脚本 (infer.py)。")
	print("模式：当前只支持自定义古汉字数据集 (custom_chars)。")
	print("输入：--image 指定单张图片路径；若不指定，会从 data_root 的 test 集中取第一个样本演示。")
	print("输出：打印预测标签（若为自定义字符则显示中文类名）及各类别概率分布。模型从配置的保存路径加载。\n")

	# 选择模型
	if cfg.model_name == "resnet18":
		print(f"加载模型: ResNet18 (单通道)。权重路径: {cfg.save_path}")
		model = resnet18_single_channel(num_classes=cfg.num_classes, pretrained=False).to(device)
	else:
		print(f"加载模型: SimpleCNN。权重路径: {cfg.save_path}")
		model = SimpleCNN(num_classes=cfg.num_classes).to(device)
	model.load_state_dict(torch.load(cfg.save_path, map_location=device))
	model.eval()
	# 强制使用自定义数据集映射（如果存在 label_map.json 则加载）
	idx_to_class = load_idx_to_class(cfg.data_root)
	if idx_to_class is None:
		print(f"警告：未在 {cfg.data_root} 找到 label_map.json；若要显示中文类名请先用训练脚本生成或手动提供映射。")

	if args.image:
		print(f"正在对图片进行推理：{args.image}")
		x = load_image(args.image, cfg.image_size).to(device)
		with torch.no_grad():
			logits = model(x)
			probs = torch.softmax(logits, dim=1)
			pred = probs.argmax(dim=1).item()
		pred_label = idx_to_class.get(pred, str(pred)) if idx_to_class else str(pred)
		print(f"预测结果：{pred_label}")
		print("各类别概率（前10）:")
		probs_list = probs.squeeze().tolist()
		for i, p in enumerate(probs_list):
			label = idx_to_class.get(i, str(i)) if idx_to_class else str(i)
			print(f"  {label}: {p:.4f}")
	else:
		# run on first sample from the configured test set
		if cfg.dataset_mode == "custom_chars":
			_, _, test_loader, _ = get_custom_chars_loaders(
				data_root=cfg.data_root,
				batch_size=1,
				image_size=cfg.image_size,
			)
		else:
			_, test_loader = get_mnist_loaders(batch_size=1, image_size=cfg.image_size)
		x, y = next(iter(test_loader))
		x = x.to(device)
		with torch.no_grad():
			logits = model(x)
			probs = torch.softmax(logits, dim=1)
			pred = probs.argmax(dim=1).item()
		true_label = (idx_to_class.get(int(y.item()), str(int(y.item()))) if idx_to_class else str(int(y.item())))
		pred_label = idx_to_class.get(pred, str(pred)) if idx_to_class else str(pred)
		print(f"示例样本 — 真实标签: {true_label}，预测: {pred_label}")
		print("预测概率:")
		probs_list = probs.squeeze().tolist()
		for i, p in enumerate(probs_list):
			label = idx_to_class.get(i, str(i)) if idx_to_class else str(i)
			print(f"  {label}: {p:.4f}")


if __name__ == '__main__':
	main()

