import json
import os


DATA_ROOT = os.path.join("data")  # 修改这里：你的古汉字数据目录
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")


def main():
    if not os.path.isdir(TRAIN_ROOT):
        print(f"错误：未找到训练目录 {TRAIN_ROOT}")
        return

    classes = sorted(
        d for d in os.listdir(TRAIN_ROOT)
        if os.path.isdir(os.path.join(TRAIN_ROOT, d))
    )
    if not classes:
        print(f"错误：{TRAIN_ROOT} 下没有任何类别文件夹")
        return

    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    label_map_path = os.path.join(DATA_ROOT, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    print("已生成标签映射：")
    print(json.dumps(class_to_idx, ensure_ascii=False, indent=2))
    print(f"保存位置：{label_map_path}")


if __name__ == "__main__":
    main()
