class Config:
    seed = 42
    batch_size = 1024
    lr = 1e-3
    epochs = 100
    dataset_mode = "custom_chars"
    data_root = "data"  # Dataset root, should include train/val/test.
    num_classes = 10
    image_size = 28
    model_name = "resnet18"  # options: simple_cnn, resnet18
    num_workers = 0
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model_save_paths = {
        "simple_cnn": "checkpoints/simple_cnn_best_model.pth",
        "resnet18": "checkpoints/resnet18_best_model.pth",
    }
    save_path = model_save_paths[model_name]

    @classmethod
    def get_save_path(cls, model_name=None):
        model_name = model_name or cls.model_name
        if model_name not in cls.model_save_paths:
            options = ", ".join(cls.model_save_paths)
            raise ValueError(f"Unknown model_name: {model_name}. Options: {options}")
        return cls.model_save_paths[model_name]
