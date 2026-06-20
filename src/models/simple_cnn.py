import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
	def __init__(self, num_classes=10):
		super(SimpleCNN, self).__init__()
		self.feature = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 7 * 7, 128),
			nn.ReLU(inplace=True), 
			nn.Dropout(0.5),
			nn.Linear(128, num_classes)
		)

	def forward(self, x):
		x = self.feature(x)
		x = self.classifier(x)
		return x

