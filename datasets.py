# 数据读取：把 prepare.py 生成的 HDF5 文件包装为 PyTorch 数据集。
# lr 是退化后再放大的输入，hr 是清晰目标；两者高宽相同，训练时学习对应像素的关系。
# 单样本形状为 [1, H, W]；DataLoader 将多个样本组合成 [N, 1, H, W]。

import h5py  # 导入 h5py，用于读取 HDF5（.h5）文件中的数组和分组。
import numpy as np  # 导入 NumPy 并取别名 np，用于数组归一化和增加维度。
from torch.utils.data import Dataset  # 导入 PyTorch 数据集基类，自定义数据集需要实现取样本和获取长度的方法。


class TrainDataset(Dataset):  # 定义训练数据集类；训练文件中 lr/hr 都是由等大小小图块组成的数组。
    def __init__(self, h5_file):  # 初始化数据集，h5_file 是训练文件的路径。
        super(TrainDataset, self).__init__()  # 调用 Dataset 父类的初始化方法。
        self.h5_file = h5_file  # 保存文件路径，之后取样本时再打开文件。

    def __getitem__(self, idx):  # 定义按索引取样本的方法；dataset[idx] 会调用它。
        with h5py.File(self.h5_file, 'r') as f:  # 以只读模式打开文件；with 代码块结束后会自动关闭文件。
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)  # 取出第 idx 对图块，除以 255 归一化，再在第 0 维增加通道轴；返回 (输入, 目标)。

    def __len__(self):  # 定义数据集长度；len(dataset) 会调用它。
        with h5py.File(self.h5_file, 'r') as f:  # 以只读模式打开 HDF5 文件，读取样本数量。
            return len(f['lr'])  # 返回 lr 数组的第一维长度，即训练图块数量。


class EvalDataset(Dataset):  # 定义验证数据集类；验证文件用分组保存整张图像，因此不同样本可有不同尺寸。
    def __init__(self, h5_file):  # 初始化验证数据集，接收验证文件的路径。
        super(EvalDataset, self).__init__()  # 调用父类初始化方法。
        self.h5_file = h5_file  # 保存验证文件路径。

    def __getitem__(self, idx):  # 按索引获取一对完整的验证图像。
        with h5py.File(self.h5_file, 'r') as f:  # 以只读方式打开文件，退出 with 时自动关闭。
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)  # 把索引转为字符串键，[:, :] 读取完整二维图像；归一化并补通道轴后返回 (输入, 目标)。

    def __len__(self):  # 定义验证数据集的样本数查询方法。
        with h5py.File(self.h5_file, 'r') as f:  # 打开验证文件以读取分组内容。
            return len(f['lr'])  # 返回 lr 分组内子数据集的数量，即验证图像数量。
