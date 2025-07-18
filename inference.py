
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import os
from models import UNet

label_to_color = {
    0: (0, 0, 0),  # 黑色：  background
    1: (0, 255, 0),  # 绿色：  liugua
    # 1: (255, 170, 0),   # 棕色：  huahen
    # 1: (0, 255, 127),   # madian
    # 1: (255, 255, 0),   # qipao
    # 1: (238, 130, 238),   # yuyan
}

@torch.inference_mode()   # 禁用反向传播，只进行前向计算
def predict_image(model, image_path, output_dir, device, visualize=True):
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    image_array = np.transpose(image_array, (2, 0, 1))
    # print(image_array.shape)
    # if len(image_array.shape) == 3:
    #     image_array = image_array[:, :, 0]
    
    image_tensor = torch.from_numpy(image_array).float()  # (h, w)  =>  (1280, 1920)
    image_tensor = image_tensor.unsqueeze(0)   # (b, c, h, w) => (1, 1, 1280, 1920)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        
        output_probs = F.softmax(output, dim=1)
        
        # 获取每个像素的预测类别（取最大概率的索引）
        mask = torch.argmax(output_probs, dim=1).squeeze().cpu().numpy()
        # squeeze()移除尺寸为 1 的张量维度, 在这里是移除最前面的batchsize维度
    
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        colored_mask[mask == label] = color
        # 打印非零数值，检测结果
        non_zero_mask = mask != 0
        non_zero_values = mask[non_zero_mask]
        print("非零值:", non_zero_values)

    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    Image.fromarray(colored_mask).save(mask_path)
    
    return mask, mask_path

def get_args():
    parser = argparse.ArgumentParser(description="UNet模型推理")
    parser.add_argument('--checkpoint', type=str, default="checkpoints_boluo/checkpoint_epoch299.pth", help="模型检查点路径")
    #parser.add_argument('--input', type=str, default="liugua/liugua/liugua/liugua_1.png", help="输入图像路径或包含图像的目录")
    # parser.add_argument('--input', type=str, default="/home/qinyihua/TrainData/HuahenData/huahen/huahen/huahen_3.png", help="输入图像路径或包含图像的目录")
    #parser.add_argument('--checkpoint', type=str, default="checkpoints/checkpoint_epoch499.pth", help="模型检查点路径")
    parser.add_argument('--input', type=str, default="dataset/boluo_test/boluo/boluo_014.png", help="输入图像路径或包含图像的目录")
    parser.add_argument('--output', type=str, default="./output", help="输出结果保存目录")
    parser.add_argument('--n_channels', type=int, default=3, help="输入通道数")
    parser.add_argument('--classes', type=int, default=2, help="类别数（包括背景）")
    parser.add_argument('--base_channels', type=int, default=112, help="UNet基础通道数")
    parser.add_argument('--visualize', action='store_true', default=True, help="是否可视化分割结果")
    return parser.parse_args()

def main():
    args = get_args()
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    model = UNet(n_channels=args.n_channels, n_classes=args.classes, base_channels=args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    print(f"模型加载自: {args.checkpoint}")
    
    input_path = Path(args.input)
    assert input_path.exists(), f"错误: 文件 {args.input} 不存在"
        
    print(f"处理图像: {args.input}")
    mask, save_path = predict_image(model, args.input, args.output, device, args.visualize)
    print(f"分割掩码已保存至: {save_path}")

if __name__ == "__main__":
    main()
