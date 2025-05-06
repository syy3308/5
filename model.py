import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.conv1(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.conv2(x).view(batch_size, -1, H * W)
        value = self.conv3(x).view(batch_size, -1, H * W)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return out + x


class YOLOWithAttention(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # 修改FPN通道数以匹配ResNet输出
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(512, 256, 1),  # C3
            nn.Conv2d(1024, 256, 1),  # C4
            nn.Conv2d(2048, 256, 1)  # C5
        ])

        # 注意力模块
        self.attention_modules = nn.ModuleList([
            AttentionModule(256)
            for _ in range(3)
        ])

        # 检测头
        self.det_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes + 4, 3, padding=1)
            )
            for _ in range(3)
        ])

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 获取backbone特征
        features = self.backbone(x)

        # FPN处理
        fpn_features = []
        for feature, fpn_conv in zip(features, self.fpn_convs):
            fpn_features.append(fpn_conv(feature))

        # 注意力和检测
        results = []
        for feature, attention, det_conv in zip(
                fpn_features, self.attention_modules, self.det_convs):
            # 注意力处理
            feature = attention(feature)

            # 检测头
            out = det_conv(feature)

            # 分离分类和定位预测
            batch_size = out.size(0)
            conf = out[:, :self.num_classes].permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            loc = out[:, self.num_classes:].permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

            results.append((conf, loc))

        # 合并预测结果
        conf_preds = torch.cat([conf for conf, _ in results], dim=1)
        loc_preds = torch.cat([loc for _, loc in results], dim=1)

        return conf_preds, loc_preds

