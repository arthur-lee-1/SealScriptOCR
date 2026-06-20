import json
import os
from PIL import Image, ImageOps, ImageFilter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def _resize_with_padding(image, image_size, fill=255):
  width, height = image.size
  if width == 0 or height == 0:
    return Image.new("L", (image_size, image_size), color=fill)

  scale = min(image_size / width, image_size / height)
  new_width = max(1, int(round(width * scale)))
  new_height = max(1, int(round(height * scale)))
  resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

  canvas = Image.new("L", (image_size, image_size), color=fill)
  left = (image_size - new_width) // 2
  top = (image_size - new_height) // 2
  canvas.paste(resized, (left, top))
  return canvas


class AncientCharPreprocess:
  def __init__(self, image_size=28, pad_fill=255):
    self.image_size = image_size
    self.pad_fill = pad_fill

  def __call__(self, image):
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = _resize_with_padding(image, self.image_size, fill=self.pad_fill)
    image = ImageOps.autocontrast(image)
    return image


def _build_mnist_transform(image_size):
  return transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
  ])


def _build_custom_char_transform(image_size):
  return transforms.Compose([
    AncientCharPreprocess(image_size=image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
  ])


def get_mnist_loaders(batch_size=64, image_size=28, download=True):
  """Return train and test DataLoader for MNIST.

  Args:
    batch_size (int): batch size for loaders
    image_size (int): target image size (height and width)
    download (bool): whether to download the dataset

  Returns:
    (DataLoader, DataLoader): train_loader, test_loader
  """

  transform = _build_mnist_transform(image_size)

  train_ds = datasets.MNIST(root="./data", train=True, download=download, transform=transform)
  test_ds = datasets.MNIST(root="./data", train=False, download=download, transform=transform)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
  test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

  return train_loader, test_loader


def get_custom_chars_loaders(data_root, batch_size=64, image_size=28, num_workers=0):
  """Load a custom ancient-character dataset stored in ImageFolder layout.

  Expected structure:
    data_root/
      train/<class_name>/*.png|jpg
      val/<class_name>/*.png|jpg
      test/<class_name>/*.png|jpg
  """

  transform = _build_custom_char_transform(image_size)

  train_ds = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=transform)
  val_ds = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=transform)
  test_ds = datasets.ImageFolder(root=os.path.join(data_root, "test"), transform=transform)

  label_map = train_ds.class_to_idx
  with open(os.path.join(data_root, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  return train_loader, val_loader, test_loader, label_map
