import json
import os
import cv2
import logging
from PIL import Image
import numpy as np

# 设置日志级别（调试时可设置为 DEBUG）
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def find_images_directory(dataset_root):
    """
    在 dataset_root 中查找图像存放目录，优先查找 "Images"，其次 "images"、"images_test"。
    """
    possible_dirs = ["Images", "images", "images_test"]
    for d in possible_dirs:
        candidate = os.path.join(dataset_root, d)
        if os.path.exists(candidate):
            logging.info(f"Found images directory: {candidate}")
            return candidate
    logging.error("未找到图像目录，请检查 dataset_root 结构。")
    return None

def load_annotations(annotation_file):
    """
    加载标注文件。若文件以 '[' 开头，则当作 JSON 数组加载；否则逐行解析为 JSON 对象（JSON Lines 格式）。
    """
    annotations = []
    try:
        with open(annotation_file, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                annotations = json.load(f)
            else:
                for idx, line in enumerate(f):
                    try:
                        anno = json.loads(line.strip())
                        annotations.append(anno)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON 解析错误，第 {idx+1} 行: {e}")
    except Exception as e:
        logging.error(f"加载标注文件 {annotation_file} 失败: {e}")
    logging.info(f"加载到 {len(annotations)} 条标注数据。")
    return annotations

def is_valid_media_path(media_path):
    """
    判断媒体路径是否有效：长度大于3且以常见图片扩展名结尾。
    """
    if not media_path or len(media_path) < 3:
        return False
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    for ext in valid_exts:
        if media_path.lower().endswith(ext):
            return True
    return False

def load_image_cv2(image_path):
    """
    尝试使用 OpenCV 加载图像。
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("cv2.imread 返回 None")
        return img
    except Exception as e:
        logging.error(f"使用 cv2.imread 加载图像 {image_path} 时出错: {e}")
        return None

def load_image_pillow(image_path):
    """
    当 OpenCV 加载失败时，尝试使用 Pillow 加载图像。
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = np.array(img)
        return img
    except Exception as e:
        logging.error(f"使用 Pillow 加载图像 {image_path} 时出错: {e}")
        return None

def get_image_filename(sample, available_files, sample_index):
    """
    根据样本信息依次检查 "img_file"、"media"、"ID" 字段，
    如均未提供，则从 available_files 中按索引选择 fallback 文件名。
    """
    img_filename = sample.get("img_file", "").strip()
    if img_filename:
        logging.debug("使用字段 img_file 获取图像文件名")
        return img_filename

    if "media" in sample:
        img_filename = sample.get("media", {}).get("media_path", "").strip()
        if img_filename:
            logging.debug("使用字段 media.media_path 获取图像文件名")
            return img_filename

    if "ID" in sample:
        img_filename = sample.get("ID", "").strip() + ".jpg"
        if img_filename:
            logging.debug("使用字段 ID 获取图像文件名")
            return img_filename

    # fallback：从 available_files 列表中按索引选取
    if available_files:
        fallback_name = available_files[sample_index % len(available_files)]
        logging.debug(f"样本缺少图像信息，使用 fallback 文件名: {fallback_name}")
        return fallback_name

    return ""

def preprocess_data(annotation_file, dataset_root):
    """
    数据预处理函数：
      - 加载指定的标注文件（例如 annotation_train.odgt）
      - 在 dataset_root 中查找图像目录
      - 根据样本信息构造图像文件名，并加载图像
      - 如果构造的图像文件不存在，则使用 fallback 文件（available_files 中按 sample 索引选择）
      - 返回 (图像, 检测框列表) 数据对列表
    """
    if not os.path.exists(annotation_file):
        logging.error(f"标注文件不存在: {annotation_file}")
        return []

    annotations = load_annotations(annotation_file)
    if not annotations:
        logging.error("未加载到任何标注数据！")
        return []

    images_dir = find_images_directory(dataset_root)
    if not images_dir:
        return []

    # 列举 images_dir 中所有有效图像文件，用于 fallback
    available_files = [f for f in os.listdir(images_dir) if is_valid_media_path(f)]
    logging.info(f"images 目录中的文件: {available_files}")

    data = []
    for idx, sample in enumerate(annotations):
        img_filename = get_image_filename(sample, available_files, idx)
        full_image_path = os.path.join(images_dir, img_filename)

        # 如果构造的图像文件不存在，则使用 fallback 文件
        if not os.path.exists(full_image_path):
            logging.warning(f"构造的图像不存在: {full_image_path}")
            if available_files:
                fallback_name = available_files[idx % len(available_files)]
                full_image_path = os.path.join(images_dir, fallback_name)
                logging.info(f"使用 fallback 图像: {full_image_path}")
            else:
                logging.error("images 目录中没有有效图像文件")
                continue

        logging.info(f"加载图像: {full_image_path}")
        img = load_image_cv2(full_image_path)
        if img is None:
            logging.info("尝试使用 Pillow 加载图像")
            img = load_image_pillow(full_image_path)
        if img is None:
            logging.warning(f"无法加载图像: {full_image_path}. 使用空数组作为占位。")
            img = np.array([])

        gtboxes = sample.get("gtboxes", [])
        data.append((img, gtboxes))

    logging.info(f"预处理了 {len(data)} 个样本。")
    return data

if __name__ == "__main__":
    # 修改为你的实际路径
    # 使用位于 dataset_root 下的 annotation_train.odgt 来处理所有训练样本
    annotation_file = r"D:\ProgramData\PyCharm Community Edition 2024.3.5\PycharmProjects\PythonProject2\OpenDataLab___CrowdHuman\dsdl\dataset_root\annotation_train.odgt"
    dataset_root = r"D:\ProgramData\PyCharm Community Edition 2024.3.5\PycharmProjects\PythonProject2\OpenDataLab___CrowdHuman\dsdl\dataset_root"
    train_data = preprocess_data(annotation_file, dataset_root)
