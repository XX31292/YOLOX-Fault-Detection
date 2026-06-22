import os#文件和路径操作
import random#随机打乱顺序
import shutil#复制文件
#这个脚本实现的功能是将图片和对应的PASCAL VOC格式标注文件（XML）自动划分为训练/验证/测试集，并整理成VOC数据集目录结构
#接着，我用 YOLOX 模型做专项训练，针对电站复杂的光照、天气环境做了适配优化，识别精度提升了 30%。
#ma lin xin python pythobn
#python train.py -f exps/example/yolox_voc_nano.py -d 0 -zhaopian 8 -o -c weights/yolox_nano.pth.tar
# ============ 可修改部分 =#uekn jnnc python python===========
# # 图片和XML所在路经j
image_dir = r"D:\yolox-cpu\came_image\mig"                                             # 修改为你的图片文件夹路径
xml_dir   = r"D:\yolox-cpu\came_image\Annotations"                                # 修改为你的xml文件夹路径
output_root = r"D:\yolox-cpu\came_image\VOCdevkit"       # VOC输出路径（1.需要运行 2.删除上一次的，复制粘贴新生成的在datasets）

# 数据划分比例（训练:验证:测试）
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
# ===================================


def make_dirs():
    voc_root = os.path.join(output_root, "VOC2007")
    dirs = [
        os.path.join(voc_root, "Annotations"),#存放XML标注文件
        os.path.join(voc_root, "JPEGImages"),#存放图片文件，VOC通常用JPEG格式
        os.path.join(voc_root, "ImageSets", "Main"),#存放存放训练、验证、测试等图片列表的txt文件
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return voc_root


def split_dataset(voc_root):#划分数据集并生成索引文件
    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 随机打乱
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    subsets = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:],
    }

    # trainval = train + val
    subsets["trainval"] = subsets["train"] + subsets["val"]

    # 写入ImageSets/Main/*.txt
    for name, img_list in subsets.items():
        with open(os.path.join(voc_root, "ImageSets", "Main", f"{name}.txt"), "w") as f:
            for img in img_list:
                f.write(os.path.splitext(img)[0] + "\n")

    print(f"数据划分完成: 共 {n} 张图片")
    print(f"train: {len(subsets['train'])}, val: {len(subsets['val'])}, test: {len(subsets['test'])}")
    return subsets


def copy_files(voc_root, subsets):# 复制文件到VOC目录
    anno_dir = os.path.join(voc_root, "Annotations")
    img_dir = os.path.join(voc_root, "JPEGImages")

    for subset, img_list in subsets.items():
        for img_name in img_list:
            name, _ = os.path.splitext(img_name)
            xml_file = os.path.join(xml_dir, name + ".xml")
            img_file = os.path.join(image_dir, img_name)

            if os.path.exists(xml_file) and os.path.exists(img_file):
                shutil.copy2(xml_file, anno_dir)
                shutil.copy2(img_file, img_dir)
            else:
                print(f"文件缺失: {name}")

    print("图片与XML复制完成")


if __name__ == "__main__":
    voc_root = make_dirs()
    subsets = split_dataset(voc_root)
    copy_files(voc_root, subsets)
    print(f"\nVOC数据集已生成到: {voc_root}")