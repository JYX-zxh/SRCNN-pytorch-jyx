# 数据准备：从清晰 RGB 图像生成用于训练或验证的 HDF5 文件。
# 流程：调整尺寸 → 缩小以模拟低分辨率 → 双三次插值放大 → 提取亮度 Y → 保存。
# 训练数据切成固定大小图块；验证数据保留整张图像。lr/hr 的空间尺寸始终对应。
# 示例：python prepare.py --images-dir data --output-path train.h5 --scale 2
# 加上 --eval 则生成验证文件；输入目录应只包含需要处理的原始图像。

import argparse  # 导入命令行参数解析模块 argparse。
import glob  # 导入 glob，用通配符收集输入目录中的文件路径。
import h5py  # 导入 h5py，用于写入 HDF5 数据文件。
import numpy as np  # 导入 NumPy，用于图像数组转换和保存图块集合。
import PIL.Image as pil_image  # 导入 Pillow 图像模块并取别名 pil_image，用于读图和缩放。
from utils import convert_rgb_to_y  # 从本项目工具模块导入 RGB 转亮度 Y 的函数。


def train(args):  # 定义训练数据生成函数；args 保存命令行传入的配置。
    h5_file = h5py.File(args.output_path, 'w')  # 以写入模式创建输出文件；注意 'w' 会覆盖同路径的已有文件。

    lr_patches = []  # 创建列表，用于收集退化图像的小图块，作为网络输入。
    hr_patches = []  # 创建列表，用于收集对应的清晰图块，作为学习目标。

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):  # 匹配输入目录下的条目并排序后逐个处理；这里没有过滤扩展名，要求条目可作为图片读取。
        hr = pil_image.open(image_path).convert('RGB')  # 打开图像并统一转为 RGB 三通道格式。
        hr_width = (hr.width // args.scale) * args.scale  # 将宽度向下取整到 scale 的整数倍；// 是整数除法。
        hr_height = (hr.height // args.scale) * args.scale  # 将高度向下取整到 scale 的整数倍，以便后续缩小再放大。
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)  # 用双三次插值调整到对齐后的尺寸；这是缩放，不是裁剪。
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)  # 把清晰图像的高宽分别缩小 scale 倍，模拟细节丢失。
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)  # 把低分辨率图像放大回原尺寸；尺寸恢复了，但丢失的细节并未恢复。
        hr = np.array(hr).astype(np.float32)  # 将清晰图像转换为 float32 数组，形状为 [H, W, 3]。
        lr = np.array(lr).astype(np.float32)  # 将退化图像转换为 float32 数组，便于后续浮点计算。
        hr = convert_rgb_to_y(hr)  # 提取清晰图像的亮度 Y，形状变为 [H, W]。
        lr = convert_rgb_to_y(lr)  # 提取退化图像的亮度 Y，作为网络将要处理的单通道输入。

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):  # 沿高度滑动窗口；i 是图块起始行，stride 控制步长，+1 保证最后一个合法起点可被取到。
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):  # 沿宽度滑动窗口；j 是图块起始列，只取完全位于图像内的窗口。
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])  # 按 [起始行:结束行, 起始列:结束列] 切出输入图块，切片不含结束位置。
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])  # 在清晰图像的相同位置切出目标图块，保证输入与目标一一对应。

    lr_patches = np.array(lr_patches)  # 把输入图块列表合成数组；存在图块时形状为 [图块数, patch_size, patch_size]。
    hr_patches = np.array(hr_patches)  # 把清晰目标图块列表合成对应的数组。

    h5_file.create_dataset('lr', data=lr_patches)  # 在 HDF5 文件中创建名为 lr 的数据集，保存所有输入图块。
    h5_file.create_dataset('hr', data=hr_patches)  # 创建名为 hr 的数据集，保存所有目标图块；此处尚未除以 255。

    h5_file.close()  # 关闭文件，完成写入并释放文件资源。


def eval(args):  # 定义验证数据生成函数；这里的 eval 是本文件函数名，不是模型的 eval 方法。
    h5_file = h5py.File(args.output_path, 'w')  # 创建验证输出文件；写入模式会覆盖同名文件。

    lr_group = h5_file.create_group('lr')  # 创建 lr 分组，相当于一个容器，用于存放不同尺寸的输入图像。
    hr_group = h5_file.create_group('hr')  # 创建 hr 分组，用于保存对应的完整清晰图像。

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):  # 排序后遍历图片；enumerate 同时提供从 0 开始的编号 i 和图片路径。
        hr = pil_image.open(image_path).convert('RGB')  # 打开当前图像，统一为 RGB 格式。
        hr_width = (hr.width // args.scale) * args.scale  # 计算可被放大倍数整除的目标宽度。
        hr_height = (hr.height // args.scale) * args.scale  # 计算可被放大倍数整除的目标高度。
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)  # 将清晰图像缩放到对齐后的尺寸，作为验证目标。
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)  # 通过双三次插值缩小，构造低分辨率图像。
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)  # 再放大到与清晰目标相同的尺寸，构造验证输入。
        hr = np.array(hr).astype(np.float32)  # 将清晰图像转为 float32 NumPy 数组。
        lr = np.array(lr).astype(np.float32)  # 将输入图像转为 float32 NumPy 数组。
        hr = convert_rgb_to_y(hr)  # 从清晰图像中提取亮度通道。
        lr = convert_rgb_to_y(lr)  # 从退化图像中提取亮度通道。

        lr_group.create_dataset(str(i), data=lr)  # 用字符串编号作为键保存整张输入图像，例如 lr/0、lr/1。
        hr_group.create_dataset(str(i), data=hr)  # 用相同编号保存清晰目标，保证验证时能够配对读取。

    h5_file.close()  # 关闭验证文件并完成写入。


if __name__ == '__main__':  # 仅在直接运行本脚本时执行下面代码；被其他文件 import 时不执行。
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器，自动提供 --help 帮助。
    parser.add_argument('--images-dir', type=str, required=True)  # 声明必填的图像目录参数；命令行 --images-dir 对应 args.images_dir。
    parser.add_argument('--output-path', type=str, required=True)  # 声明必填的输出 HDF5 文件路径。
    parser.add_argument('--patch-size', type=int, default=33)  # 设置训练图块的边长，默认每块 33×33 像素。
    parser.add_argument('--stride', type=int, default=14)  # 设置截取训练图块时的滑动步长，默认 14 像素，相邻图块可重叠。
    parser.add_argument('--scale', type=int, default=2)  # 设置退化时的缩放倍数，默认 2，使用时应传入正整数。
    parser.add_argument('--eval', action='store_true')  # 声明布尔开关：写了 --eval 就为 True，未写则为 False。
    args = parser.parse_args()  # 解析命令行参数，存入 args 对象。

    if not args.eval:  # 如果没有开启 --eval，就准备训练数据。
        train(args)  # 执行训练图块生成与保存。
    else:  # 否则进入验证数据准备流程。
        eval(args)  # 执行完整验证图像的生成与保存。
