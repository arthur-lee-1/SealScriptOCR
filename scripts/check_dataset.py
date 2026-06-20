import os


DATA_ROOT = os.path.join("data")  # 修改这里：你的古汉字数据目录
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def count_images(folder_path):
    return sum(
        1 for filename in os.listdir(folder_path)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    )


def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"错误：未找到数据根目录 {DATA_ROOT}")
        return

    for split in SPLITS:
        split_root = os.path.join(DATA_ROOT, split)
        print(f"\n== {split} ==")
        if not os.path.isdir(split_root):
            print(f"  缺少目录：{split_root}")
            continue

        class_dirs = sorted(
            d for d in os.listdir(split_root)
            if os.path.isdir(os.path.join(split_root, d))
        )
        if not class_dirs:
            print("  没有找到类别文件夹")
            continue

        for class_name in class_dirs:
            class_path = os.path.join(split_root, class_name)
            image_count = count_images(class_path)
            note = ""
            if image_count < 50:
                note = "（警告：样本过少）"
            elif image_count < 200:
                note = "（建议：增加到至少 200 张）"
            print(f"  {class_name}: {image_count} 张 {note}")


if __name__ == "__main__":
    main()
