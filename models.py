# 模型定义：SRCNN 用三层卷积学习从模糊图像到清晰图像的映射。
# 建议阅读顺序：models.py → utils.py → prepare.py → datasets.py → train.py → test.py。
# 形状记法：N 为批量大小，C 为通道数，H/W 为图像高/宽；卷积输入为 [N, C, H, W]。
# 本项目先用双三次插值放大图像，网络本身保持空间尺寸不变，只学习改善图像细节。

from torch import nn  # 从 PyTorch 导入神经网络模块 nn，用于定义卷积层和激活函数。


class SRCNN(nn.Module):  # 定义 SRCNN 类，继承 nn.Module 后可使用参数管理、设备迁移等功能。
    def __init__(self, num_channels=1):  # 初始化模型；self 表示当前对象，默认只处理亮度 Y 这 1 个通道。
        super(SRCNN, self).__init__()  # 调用父类初始化方法，建立 PyTorch 管理子层和参数所需的内部结构。
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)  # 第一层：将输入变成 64 张特征图；9×9 卷积核，四周补 4 个像素以保持高宽。
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)  # 第二层：将 64 个通道映射为 32 个通道；5×5 卷积核，padding=2 保持高宽。
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)  # 第三层：将 32 个特征通道重建为输出图像通道；默认输出形状为 [N, 1, H, W]。
        self.relu = nn.ReLU(inplace=True)  # 创建 ReLU 激活函数，把负数置零；inplace=True 表示就地修改输入以节省内存。

    def forward(self, x):  # 定义前向计算；调用 model(x) 时，PyTorch 会执行这个方法。
        x = self.relu(self.conv1(x))  # 先执行第一层卷积，再用 ReLU 引入非线性；输出形状为 [N, 64, H, W]。
        x = self.relu(self.conv2(x))  # 执行第二层卷积和 ReLU；输出形状为 [N, 32, H, W]。
        x = self.conv3(x)  # 执行最后一层卷积得到重建结果；此处不加 ReLU，输出暂未限制在 0～1。
        return x  # 把预测图像返回给调用方，形状与输入图像相同。
