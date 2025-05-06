import torch
import logging
import os
from tqdm import tqdm
import numpy as np
from train import CrowdHumanDataset, get_dataloader
from model import YOLOWithAttention
from backbone import ResNetBackbone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def calculate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    计算检测指标（AP、Recall等）
    """
    true_positives = 0
    false_positives = 0
    false_negatives = len(gt_boxes)

    for pred_box in pred_boxes:
        max_iou = 0
        max_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = i

        if max_iou >= iou_threshold:
            true_positives += 1
            false_negatives -= 1
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    """
    # 确保输入是numpy数组
    box1 = np.array(box1)
    box2 = np.array(box2)

    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union = area1 + area2 - intersection

    # 计算IoU
    iou = intersection / union if union > 0 else 0

    return iou


def evaluate_model(model, val_loader, device='cpu'):
    """
    评估模型性能
    """
    model.eval()
    total_metrics = {
        'precision': 0,
        'recall': 0,
        'f1_score': 0
    }
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            try:
                images, targets = zip(*batch)
                images = torch.stack([img for img in images]).to(device)

                # 获取模型预测
                conf_preds, loc_preds = model(images)

                # 对每个样本计算指标
                for i in range(len(images)):
                    pred_boxes = process_predictions(conf_preds[i], loc_preds[i])
                    gt_boxes = process_targets(targets[i])

                    metrics = calculate_metrics(pred_boxes, gt_boxes)

                    for key in total_metrics:
                        total_metrics[key] += metrics[key]

                    num_samples += 1

            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

    # 计算平均指标
    for key in total_metrics:
        total_metrics[key] /= max(1, num_samples)

    return total_metrics


def process_predictions(conf_pred, loc_pred, conf_threshold=0.5):
    """
    处理模型预测输出为边界框格式
    """
    # 获取置信度大于阈值的预测
    conf_mask = conf_pred > conf_threshold
    boxes = []

    for i in range(len(conf_mask)):
        if conf_mask[i]:
            box = loc_pred[i].cpu().numpy()
            boxes.append(box)

    return boxes


def process_targets(target):
    """
    处理目标数据为边界框格式
    """
    return target['loc'].cpu().numpy()


def main():
    try:
        # 设置路径
        base_dir = "D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/OpenDataLab___CrowdHuman"
        val_image_dir = os.path.join(base_dir, "dsdl/dataset_root/Images_val")
        val_annotation_file = os.path.join(base_dir, "dsdl/dataset_root/annotation_val.odgt")
        model_path = "models/best_model.pth"  # 确保这是正确的模型保存路径

        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 创建验证数据加载器
        val_loader = get_dataloader(
            image_dir=val_image_dir,
            annotation_file=val_annotation_file,
            batch_size=4,
            shuffle=False
        )

        # 初始化模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone = ResNetBackbone()
        model = YOLOWithAttention(backbone=backbone, num_classes=2)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # 评估模型
        metrics = evaluate_model(model, val_loader, device)

        # 打印评估结果
        logging.info("Evaluation Results:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()