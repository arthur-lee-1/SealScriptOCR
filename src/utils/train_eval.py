import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_one_epoch(model, loader, optimizer, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for x, y in tqdm(loader, desc="Train", leave=False):
		x = x.to(device)
		y = y.to(device)
		optimizer.zero_grad()
		logits = model(x)
		loss = F.cross_entropy(logits, y)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * x.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == y).sum().item()
		total += x.size(0)

	return running_loss / total, correct / total


def evaluate(model, loader, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for x, y in tqdm(loader, desc="Eval", leave=False):
			x = x.to(device)
			y = y.to(device)
			logits = model(x)
			loss = F.cross_entropy(logits, y)
			running_loss += loss.item() * x.size(0)
			preds = logits.argmax(dim=1)
			correct += (preds == y).sum().item()
			total += x.size(0)

	return running_loss / total, correct / total


def save_model(model, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(model.state_dict(), path)


def plot_metrics(history, out_path="outputs/training_plot.png"):
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	epochs = range(1, len(history['train_loss']) + 1)
	plt.figure(figsize=(8, 4))
	plt.subplot(1, 2, 1)
	plt.plot(epochs, history['train_loss'], label='train_loss')
	plt.plot(epochs, history['val_loss'], label='val_loss')
	plt.legend()
	plt.title('Loss')

	plt.subplot(1, 2, 2)
	plt.plot(epochs, history['train_acc'], label='train_acc')
	plt.plot(epochs, history['val_acc'], label='val_acc')
	plt.legend()
	plt.title('Accuracy')

	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()

