# 单图推理：加载训练好的参数，构造退化输入，再用 SRCNN 重建亮度并保存彩色结果。
# 网络只预测 Y；Cb/Cr 沿用双三次插值图像的色度，最后转换回 RGB。
# 注意：当前脚本的 PSNR 比较的是插值输入与预测结果，不是预测与清晰原图的标准评测。
# 下面保留已有学习笔记，行尾补充精确解释；“ycbyr”应为 YCbCr，“GRB”应为 RGB。

import argparse  # 导入 argparse，用于解析权重路径、图片路径和缩放倍数。

import torch  # 导入 PyTorch，用于创建模型、读取权重及张量计算。
import torch.backends.cudnn as cudnn  # 导入 cuDNN 配置，用于选择 GPU 卷积计算策略。
import numpy as np  # 导入 NumPy，用于图像数组和颜色通道组合。
import PIL.Image as pil_image  # 导入 Pillow 图像模块，用于读图、缩放与保存。

from models import SRCNN  # 导入本项目的 SRCNN 模型结构；加载权重前需要先建立相同结构。
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr  # 导入 RGB/YCbCr 相互转换函数和 PSNR 计算函数。


if __name__ == '__main__':  # 只在直接运行本脚本时执行推理流程。
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器。
    parser.add_argument('--weights-file', type=str, required=True)  # 声明必填的模型权重文件路径，通常为 .pth 文件。
    parser.add_argument('--image-file', type=str, required=True)  # 声明必填的输入图片路径。
    parser.add_argument('--scale', type=int, default=3)  # 设置模拟低分辨率时的倍数，默认 3；应与所用模型的训练数据倍数匹配。
    args = parser.parse_args()  # 读取命令行参数，生成 args 对象。

    cudnn.benchmark = True  # 允许 cuDNN 为当前输入选择较快的卷积算法。
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 优先使用第一块 CUDA GPU，没有可用 CUDA 时使用 CPU。

    model = SRCNN().to(device)  # 创建单通道 SRCNN，将参数移动到计算设备。

    state_dict = model.state_dict()  # 获取模型参数字典，其张量与模型参数共享底层存储，可据此复制权重。
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():  # 读取权重并遍历名称 n 与张量 p；map_location 回调让加载的存储保留在 CPU。
        if n in state_dict.keys():  # 确认当前权重名称存在于模型的参数字典。
            state_dict[n].copy_(p)  # 把权重值原地复制到模型对应张量；两边形状必须匹配，也可跨设备复制。
        else:  # 如果权重包含模型中没有的名称，进入错误分支。
            raise KeyError(n)  # 抛出 KeyError 并显示不匹配的名称；此循环本身没有检查模型是否缺少某些权重。

    model.eval()  # 切换为评估模式；关闭梯度则由下方 no_grad 单独控制。

    # 尺寸对齐 + 人为退化 + Bicubic 放大。
    image = pil_image.open(args.image_file).convert('RGB')  # 打开输入图片并转成 RGB 三通道。

    image_width = (image.width // args.scale) * args.scale  # 把宽度向下对齐到 scale 的整数倍，确保缩小后能按整数倍放大。
    image_height = (image.height // args.scale) * args.scale  # 把高度向下对齐到 scale 的整数倍。
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)  # 用双三次插值调整到对齐尺寸；这里修改尺寸的方式是缩放。
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)  # 缩小图像以人为模拟低分辨率，丢失一部分细节。
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)  # 再用双三次插值放回对齐后的尺寸，作为 SRCNN 的输入。
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))  # 保存插值基线图；replace 会替换路径中的所有点号，因此路径含多个点号时需留意命名。
    # 改变数据类型
    image = np.array(image).astype(np.float32)  # 把 Pillow 图像转为 [H, W, 3] 的 float32 数组，数值仍约为 0～255。
    # 改变颜色通道
    ycbcr = convert_rgb_to_ycbcr(image)  # 把 RGB 转为 YCbCr，形状仍是 [H, W, 3]。
# ---------------------------------------------------------------------------------
#   取出ycbyr三通道中的y通道
    y = ycbcr[..., 0]  # 取最后一维的第 0 个通道 Y；省略号 ... 表示保留前面的所有维度。
    # 归一化
    y /= 255.  # 原地除以 255，把亮度归一化；y 是切片视图，此操作也会修改 ycbcr 的 Y 通道。
    y = torch.from_numpy(y).to(device)  # 从 NumPy 数组创建张量，再移动到计算设备；此时形状为 [H, W]。
    # 增加维度，以符合Conv2d的格式要求
    y = y.unsqueeze(0).unsqueeze(0)  # 依次增加通道和批量维，从 [H, W] 变为 Conv2d 要求的 [1, 1, H, W]。

    with torch.no_grad():  # 推理时不记录梯度，减少计算图所占用的内存。
        preds = model(y).clamp(0.0, 1.0)  # 预测清晰亮度并把结果限制到 0～1，形状仍为 [1, 1, H, W]。
    # 计算机PSNR  图像超分辨率的图像质量指标
    psnr = calc_psnr(y, preds)  # 计算输入 y 与预测 preds 的 PSNR；这里没有与清晰原图比较，不能据此判断标准重建质量。
    print('PSNR: {:.2f}'.format(psnr))  # 输出上述 PSNR，单位为 dB，保留小数点后两位。
    # 把数值从0-1变成0-255，之后把数据从gpu放回cpu，然后把tensor变成numpy，去掉conv2d前边两个维度
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)  # 恢复到约 0～255，移到 CPU 后转 NumPy，再去掉批量轴和通道轴，得到 [H, W]。
    # 把处理后的y通道和没有处理的cb，cr通道拼接成原图片，并且转换数组顺序
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])  # 将预测 Y 与插值图像的 Cb/Cr 组合，再从 [3, H, W] 重排为 [H, W, 3]。
    # 把图片转换回GRB通道
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)  # 转回 RGB，将越界值限制在 0～255，再转为 8 位无符号整数用于保存图片。
    # 变回图片
    output = pil_image.fromarray(output)  # 把 NumPy 数组转换为 Pillow 图像对象。
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))  # 保存 SRCNN 结果，文件名加入倍数后缀；这里同样会替换路径中的全部点号。
