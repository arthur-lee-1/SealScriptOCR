class Config:
    seed = 42
    batch_size = 512
    lr = 1e-3
    epochs = 100
    dataset_mode = "mnist"
    data_root = "data"  # 修改这里：你的古汉字数据根目录（应包含 train/val/test）
    num_classes = 4
    image_size = 28
    model_name = "simple_cnn"  # options: simple_cnn, resnet18
    num_workers = 0
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    save_path = "checkpoints/best_model.pth"  # 修改这里：模型保存路径
