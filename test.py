import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import os
from models import UNet
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import *
from utils import *

@torch.inference_mode()   # 禁用反向传播，只进行前向计算
def test(model, test_loader, criterion, device, amp):
    model.eval()

    epoch_loss = 0
    progress = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc='Test')
    for i, batch in progress:
        images, true_masks = batch

        if len(images.shape) == 3:  # [B,H,W]对于单通道（灰度图）的情况
            images = images.unsqueeze(1)  # [B,1,H,W]

        assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels.'
        
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        # memory_format=torch.channels_last不改变张量的维度布局，只改变内存中的存储顺序用于gpu加速
        true_masks = true_masks.to(device=device, dtype=torch.long)

        # 使用混合精度
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            masks_pred = model(images)

            loss1 = criterion(masks_pred, true_masks)

            masks_pred_softmax = F.softmax(masks_pred, dim=1).float()
            true_masks_one_hot = F.one_hot(true_masks, num_classes=model.n_classes)   # shape=(b, h, w, c)
            true_masks_one_hot = true_masks_one_hot.permute(0, 3, 1, 2).float()   # shape=(b, c, h, w)
            loss2 = dice_loss(masks_pred_softmax, true_masks_one_hot, multiclass=True)

            loss3 = multiclass_dice_coeff(masks_pred[:, 1:], true_masks_one_hot[:, 1:], reduce_batch_first=False)
        # epoch_loss += 0.9 * loss1 + 0.1 * loss2
        epoch_loss += loss3

    return epoch_loss / len(test_loader)   # dice_score分数越高越好

def get_args():
    parser = argparse.ArgumentParser(description="UNet模型推理")
    parser.add_argument('--checkpoint', type=str, default="checkpoints_boluo/checkpoint_epoch474.pth", help="模型检查点路径")
    parser.add_argument('--n_channels', type=int, default=3, help="输入通道数")
    parser.add_argument('--classes', type=int, default=2, help="类别数（包括背景）")
    parser.add_argument('--base_channels', type=int, default=112, help="UNet基础通道数")
    parser.add_argument('--alpha1', type=float, default=0.9, help='Hyperparameter Alpha 1')
    parser.add_argument('--alpha2', type=float, default=0.1, help='Hyperparameter Alpha 2')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f"device: {device}")
    
    test_dataset = CustomSegmentationDataset(root_dir='/home/qinyihua/TestData/QipaoData')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = UNet(n_channels=args.n_channels, n_classes=args.classes, base_channels=args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    print(f"模型加载自: {args.checkpoint}")

    criterion = nn.CrossEntropyLoss()
    
    loss = test(model, test_loader, criterion, device, args.amp)

    print(f"loss={loss}")

if __name__ == "__main__":
    main()