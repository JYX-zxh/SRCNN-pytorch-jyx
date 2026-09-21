# 公共工具：颜色空间转换、PSNR 指标和加权平均值统计。
# NumPy 彩色图像按 [H, W, C] 排列；这里的 Tensor 分支预期 [C, H, W] 或 [1, C, H, W]。
# 颜色转换函数使用约 0～255 的数值尺度；calc_psnr 则假定图像已归一化到 0～1。
# 注意：下方两个三通道转换函数的 Tensor 分支沿用原实现，cat 后的维度存在问题，详见行内说明。
# 当前 prepare.py/test.py 的颜色转换使用 NumPy 分支。

import torch  # 导入 PyTorch，用于识别张量类型和计算 PSNR。
import numpy as np  # 导入 NumPy，用于数组操作和颜色通道重排。


def convert_rgb_to_y(img):  # 定义从 RGB 图像中计算亮度 Y 的函数，返回单通道二维图像。
    if type(img) == np.ndarray:  # 判断是否为 NumPy 数组；这里比较的是精确类型，不包含其子类。
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.  # 对 R/G/B 通道作加权求和并加偏移，得到 Y；[:, :, 0/1/2] 分别取红、绿、蓝通道。
    elif type(img) == torch.Tensor:  # 如果输入的精确类型是 PyTorch Tensor，则使用通道在前的索引方式。
        if len(img.shape) == 4:  # 检查是否带有批量维度，例如 [1, 3, H, W]。
            img = img.squeeze(0)  # 移除大小为 1 的第 0 维；若批量大小大于 1，不会移除，因此此分支不能通用地处理批量。
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.  # 从 [3, H, W] 的三个通道计算亮度，公式与 NumPy 分支相同。
    else:  # 处理未支持的输入类型。
        raise Exception('Unknown Type', type(img))  # 抛出异常并附上实际类型，提醒调用者输入应为数组或张量。


def convert_rgb_to_ycbcr(img):  # 定义 RGB 转 YCbCr：Y 表示亮度，Cb/Cr 表示色度分量。
    if type(img) == np.ndarray:  # 处理形状为 [H, W, 3] 的 NumPy 图像。
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.  # 按 RGB 的加权组合计算亮度 Y，16 是该转换公式的偏移量。
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.  # 计算蓝色色度 Cb；128 为色度的中心偏移，系数属于颜色空间转换公式。
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.  # 计算红色色度 Cr；输入各通道仍使用约 0～255 的尺度。
        return np.array([y, cb, cr]).transpose([1, 2, 0])  # 先组合为 [3, H, W]，再将坐标轴重排为 [H, W, 3] 返回。
    elif type(img) == torch.Tensor:  # 处理 PyTorch 张量输入。
        if len(img.shape) == 4:  # 检查输入是否为包含批量维的四维张量。
            img = img.squeeze(0)  # 仅当第 0 维大小为 1 时去掉批量轴，得到预期的 [3, H, W]。
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.  # 用通道在前的索引计算二维亮度 Y。
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.  # 用 RGB 三个通道计算二维色度 Cb。
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.  # 用 RGB 三个通道计算二维色度 Cr。
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)  # 原实现注意：y/cb/cr 均为二维，cat 后仍是二维，接着 permute 三个轴会报错；此处保留原逻辑。
    else:  # 处理既不是 NumPy 数组也不是 Tensor 的输入。
        raise Exception('Unknown Type', type(img))  # 抛出包含实际输入类型的异常。


def convert_ycbcr_to_rgb(img):  # 定义 YCbCr 转回 RGB 的函数，用于将预测亮度与原色度合成为彩色图像。
    if type(img) == np.ndarray:  # 处理通道在最后一维的 NumPy 数组。
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921  # 利用亮度 Y 和红色色度 Cr 还原红色通道 R。
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576  # 利用亮度 Y 与两个色度分量还原绿色通道 G。
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836  # 利用亮度 Y 和蓝色色度 Cb 还原蓝色通道 B。
        return np.array([r, g, b]).transpose([1, 2, 0])  # 组合 R/G/B 并从 [3, H, W] 转为 [H, W, 3]；调用方还需处理越界值。
    elif type(img) == torch.Tensor:  # 处理通道在前的 PyTorch 张量。
        if len(img.shape) == 4:  # 检查张量是否有额外的批量维。
            img = img.squeeze(0)  # 移除大小为 1 的批量维；这里同样只适用于单张图像。
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921  # 从 Y/Cr 计算二维红色通道。
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576  # 从 Y/Cb/Cr 计算二维绿色通道。
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836  # 从 Y/Cb 计算二维蓝色通道。
        return torch.cat([r, g, b], 0).permute(1, 2, 0)  # 原实现注意：r/g/b 为二维，cat 不会创建通道轴，后续三轴 permute 会报错；此处保留原逻辑。
    else:  # 处理不受支持的数据类型。
        raise Exception('Unknown Type', type(img))  # 抛出异常，让调用者知道传入的实际类型。


def calc_psnr(img1, img2):  # 定义峰值信噪比 PSNR，输入应为同形状、按 0～1 尺度表示的图像张量。
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))  # 先计算均方误差 MSE，再算 10×log10(1/MSE)；越大表示越接近，两图完全相同时结果为正无穷。


class AverageMeter(object):  # 定义统计类，用来记录损失或 PSNR 的当前值与加权平均值。
    def __init__(self):  # 创建统计器时执行初始化。
        self.reset()  # 调用 reset 将所有统计量清零。

    def reset(self):  # 定义重置方法，通常在每轮训练或验证开始时使用。
        self.val = 0  # val 保存最近一次传入的数值，初始为 0。
        self.avg = 0  # avg 保存累计加权平均值，初始为 0。
        self.sum = 0  # sum 保存累计加权总和，初始为 0。
        self.count = 0  # count 保存累计权重，按样本数加权时就是累计样本数。

    def update(self, val, n=1):  # 加入一次统计；val 是本次的值，n 是权重，默认代表 1 个样本。
        self.val = val  # 记录本次传入的值。
        self.sum += val * n  # 将本次值乘以权重后加入总和；批平均损失乘批大小可还原该批的损失总和。
        self.count += n  # 累加权重，记录累计参与统计的样本数。
        self.avg = self.sum / self.count  # 用加权总和除以总权重得到平均值；调用时应保证累计权重大于 0。
