import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import os
import json
import logging
import torch.nn.functional as F
from backbone import ResNetBackbone
from model import YOLOWithAttention
from loss import DetectionLoss
from transforms import TrainTransform

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class CrowdHumanDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, cache_size=100):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform or TrainTransform()
        self.cache_size = cache_size
        self.cache = {}

        logging.info(f"Initializing dataset with image directory: {image_dir}")
        logging.info(f"Using annotation file: {annotation_file}")

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        self.annotations = self.load_annotations(annotation_file)

    def load_annotations(self, file_path):
        annotations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    anno = json.loads(line.strip())
                    if 'ID' not in anno:
                        continue
                    img_path = os.path.join(self.image_dir, f"{anno['ID']}.jpg")
                    if not os.path.exists(img_path):
                        continue
                    annotations.append(anno)
                except Exception as e:
                    logging.warning(f"Error parsing annotation: {e}")
                    continue

        logging.info(f"Successfully loaded {len(annotations)} valid annotations")
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        img_id = anno['ID']
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        try:
            if img_id in self.cache:
                img = self.cache[img_id].clone()
            else:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float().permute(2, 0, 1)

                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[img_id] = img.clone()

            # 处理标注框
            gtboxes = anno.get('gtboxes', [])
            conf_targets = []
            loc_targets = []

            for box in gtboxes:
                if 'hbox' in box:
                    conf_targets.append(1)
                    loc_targets.append(box['hbox'])
                if 'fbox' in box:
                    conf_targets.append(2)
                    loc_targets.append(box['fbox'])

            # 确保至少有一个边界框
            if not conf_targets:
                conf_targets = [0]  # 背景类
                loc_targets = [[0, 0, 1, 1]]  # 虚拟框

            target = {
                'conf': torch.tensor(conf_targets, dtype=torch.float32),
                'loc': torch.tensor(loc_targets, dtype=torch.float32)
            }

            if self.transform:
                img, target = self.transform(img, target)

            return img, target

        except Exception as e:
            logging.error(f"Error processing item {idx}: {e}")
            # 返回一个有效的替代样本
            return self[max(0, idx - 1)] if idx > 0 else self[idx + 1]


# 修改 collate_fn 函数来确保批次大小一致
def collate_fn(batch):
    """
    自定义批次收集函数，确保所有样本的目标数量一致
    """
    images = []
    conf_targets = []
    loc_targets = []

    max_targets = max(len(target['conf']) for _, target in batch)

    for img, target in batch:
        if img is not None and target is not None:
            images.append(img)

            # 填充目标到相同长度
            num_targets = len(target['conf'])
            if num_targets < max_targets:
                # 填充置信度
                conf_pad = torch.zeros(max_targets - num_targets, dtype=target['conf'].dtype)
                conf = torch.cat([target['conf'], conf_pad])

                # 填充位置
                loc_pad = torch.zeros((max_targets - num_targets, 4), dtype=target['loc'].dtype)
                loc = torch.cat([target['loc'], loc_pad])
            else:
                conf = target['conf']
                loc = target['loc']

            conf_targets.append(conf)
            loc_targets.append(loc)

    if not images:
        raise ValueError("Empty batch")

    # 堆叠所有图像和目标
    images = torch.stack(images)
    conf_targets = torch.stack(conf_targets)
    loc_targets = torch.stack(loc_targets)

    return images, {'conf': conf_targets, 'loc': loc_targets}


def get_dataloader(image_dir, annotation_file, batch_size=8, shuffle=True):
    try:
        transform = TrainTransform(target_size=(640, 640))
        dataset = CrowdHumanDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        logging.info(f"Created dataloader with {len(dataset)} samples")
        return dataloader
    except Exception as e:
        logging.error(f"Error creating dataloader: {e}")
        raise


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            try:
                images = torch.stack(images).to(device)

                batch_conf_targets = []
                batch_loc_targets = []
                for t in targets:
                    batch_conf_targets.append(t['conf'])
                    batch_loc_targets.append(t['loc'])

                conf_targets = torch.cat(batch_conf_targets).to(device)
                loc_targets = torch.cat(batch_loc_targets).to(device)

                conf_preds, loc_preds = model(images)
                loss = criterion(conf_preds, loc_preds, conf_targets, loc_targets)

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logging.error(f"Error during validation: {str(e)}")
                continue

    return total_loss / max(1, num_batches)


# 修改训练函数，简化数据处理流程
def train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs=10, device='cuda'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = model.to(device)
    best_loss = float('inf')

    # 使用简单的学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )

    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, (images, targets) in enumerate(progress_bar):
                try:
                    # 直接使用堆叠后的数据
                    images = images.to(device)
                    conf_targets = targets['conf'].to(device)
                    loc_targets = targets['loc'].to(device)

                    optimizer.zero_grad()

                    # 前向传播
                    conf_preds, loc_preds = model(images)
                    loss = criterion(conf_preds, loc_preds, conf_targets, loc_targets)

                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{total_loss / num_batches:.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                    })

                    # 每1000个批次保存一次
                    if batch_idx > 0 and batch_idx % 1000 == 0:
                        os.makedirs('checkpoints', exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                        }, f'checkpoints/checkpoint_e{epoch}_b{batch_idx}.pth')

                except Exception as e:
                    logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

            # 验证
            val_loss = validate_model(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            logging.info(f"Epoch {epoch + 1}/{num_epochs}, "
                         f"Train Loss: {total_loss / num_batches:.4f}, "
                         f"Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'models/best_model.pth')

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")

    return model


# 修改主函数
def main():
    try:
        # 设置基础路径
        base_dir = "D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/OpenDataLab___CrowdHuman"

        # 创建数据加载器，使用更大的批次大小和更多的工作进程
        train_loader = get_dataloader(
            image_dir=os.path.join(base_dir, "dsdl/dataset_root/Images"),
            annotation_file=os.path.join(base_dir, "dsdl/dataset_root/annotation_train.odgt"),
            batch_size=16,  # 增加批次大小
            shuffle=True
        )

        val_loader = get_dataloader(
            image_dir=os.path.join(base_dir, "dsdl/dataset_root/Images_val"),
            annotation_file=os.path.join(base_dir, "dsdl/dataset_root/annotation_val.odgt"),
            batch_size=16,
            shuffle=False
        )

        # 初始化模型
        backbone = ResNetBackbone()
        model = YOLOWithAttention(backbone=backbone, num_classes=2)

        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.0001,
            amsgrad=True
        )

        criterion = DetectionLoss()

        # 创建必要的目录
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # 开始训练
        train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=10
        )

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()