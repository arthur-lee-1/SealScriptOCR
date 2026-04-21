class Config:
    seed = 42
    batch_size = 64
    lr = 1e-3
    eoochs = 10
    num_classes = 10 # 替换
    image_size = 28 # 替换
    device = "cuda"
    save_path = "checkpoints/best_model.pth" # 替换