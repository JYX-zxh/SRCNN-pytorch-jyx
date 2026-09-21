# 训练入口：读取成对图像，用均方误差训练 SRCNN，并按验证 PSNR 保存最佳权重。
# 一轮 epoch 会遍历一次训练数据；每个 batch 执行前向计算、反向传播和参数更新。
# 张量流：输入/目标 [N, 1, H, W] → 预测 [N, 1, H, W] → 标量损失。
# --scale 在本脚本中仅用于输出目录命名；输入 HDF5 数据的退化倍数需自行匹配。
# 学习时可先用 --num-epochs 1 --num-workers 0 运行小规模数据。

import argparse  # 导入 argparse，用于接收训练文件路径、学习率等命令行参数。
import os  # 导入 os，用于路径拼接、检查目录和创建目录。
import copy  # 导入 copy，用于深拷贝最佳模型的参数，避免其随训练继续改变。

import torch  # 导入 PyTorch，负责张量计算、设备管理和权重保存。
from torch import nn  # 导入神经网络模块，用于创建损失函数。
import torch.optim as optim  # 导入优化器模块并取别名 optim。
import torch.backends.cudnn as cudnn  # 导入 cuDNN 配置模块，用于设置 NVIDIA GPU 卷积计算策略。
from torch.utils.data.dataloader import DataLoader  # 导入 DataLoader，用于分批加载样本、打乱顺序和调用数据集。
from tqdm import tqdm  # 导入 tqdm，用进度条显示每轮训练进度。

from models import SRCNN  # 导入本项目定义的 SRCNN 网络。
from datasets import TrainDataset, EvalDataset  # 导入训练和验证数据集类。
from utils import AverageMeter, calc_psnr  # 导入加权平均统计器和 PSNR 计算函数。


if __name__ == '__main__':  # 仅在直接运行此脚本时启动训练；这一保护也有助于 Windows 的多进程数据加载。
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器。
    parser.add_argument('--train-file', type=str, required=True)  # 指定训练 HDF5 文件路径，required=True 表示必须提供。
    parser.add_argument('--eval-file', type=str, required=True)  # 指定验证 HDF5 文件路径，必须提供。
    parser.add_argument('--outputs-dir', type=str, required=True)  # 指定保存训练权重的根目录，必须提供。
    parser.add_argument('--scale', type=int, default=3)  # 设置倍数标记，默认 3；这里不会自动改变训练数据或模型尺寸。
    parser.add_argument('--lr', type=float, default=1e-4)  # 设置基础学习率，默认 1e-4，即 0.0001，控制参数更新的步幅。
    parser.add_argument('--batch-size', type=int, default=16)  # 设置每个训练批次的样本数，默认 16。
    parser.add_argument('--num-epochs', type=int, default=400)  # 设置训练轮数，默认完整遍历训练集 400 次。
    parser.add_argument('--num-workers', type=int, default=8)  # 设置数据加载子进程数，默认 8；设为 0 时在主进程加载。
    parser.add_argument('--seed', type=int, default=123)  # 设置随机种子，默认 123。
    args = parser.parse_args()  # 解析参数；命令行参数名中的短横线会转换成属性名中的下划线。

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))  # 在输出根目录下拼接倍数子目录，例如 outputs/x3。

    if not os.path.exists(args.outputs_dir):  # 判断输出目录是否还不存在。
        os.makedirs(args.outputs_dir)  # 创建输出目录，必要时同时创建上级目录。

    cudnn.benchmark = True  # 允许 cuDNN 选择较快的卷积算法，固定输入尺寸时通常有帮助；不保证完全确定性。
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # CUDA 可用时选第一块 GPU，否则选择 CPU。

    torch.manual_seed(args.seed)  # 设置 PyTorch 随机种子以帮助复现实验，但仅靠这一行不能保证所有计算完全可复现。

    model = SRCNN().to(device)  # 创建默认单通道 SRCNN，并将模型参数移动到选定设备。
    criterion = nn.MSELoss()  # 创建均方误差损失，默认对所有样本、通道和像素的平方误差取平均。
    optimizer = optim.Adam([  # 创建 Adam 优化器，下面用参数组为不同卷积层设置学习率。
        {'params': model.conv1.parameters()},  # 把第一层参数放入一个参数组，使用下面指定的基础学习率。
        {'params': model.conv2.parameters()},  # 把第二层参数放入一个参数组，同样使用基础学习率。
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}  # 为第三层单独设置基础学习率的十分之一，让重建层更新得更小。
    ], lr=args.lr)  # 结束参数组列表，并为未单独配置的组设置默认学习率。

    train_dataset = TrainDataset(args.train_file)  # 构建训练数据集对象，此时保存路径，实际取样本时再读文件。
    train_dataloader = DataLoader(dataset=train_dataset,  # 创建训练数据加载器，数据来源是上面定义的数据集。
                                  batch_size=args.batch_size,  # 每次组合指定数量的样本，得到形状为 [N, 1, H, W] 的输入与目标。
                                  shuffle=True,  # 每轮打乱样本顺序，降低固定数据顺序对训练的影响。
                                  num_workers=args.num_workers,  # 用指定数量的子进程加载训练样本。
                                  pin_memory=True,  # 启用锁页内存选项，在使用 GPU 时有助于主机到设备的数据传输。
                                  drop_last=True)  # 丢弃最后不足一个完整批次的样本；数据量小于 batch_size 时将没有训练批次。
    eval_dataset = EvalDataset(args.eval_file)  # 构建验证数据集，逐张读取完整图像。
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)  # 验证每次只取一张图，方便处理不同高宽的样本。

    best_weights = copy.deepcopy(model.state_dict())  # 深拷贝初始权重作为最佳权重的初值；state_dict 包含模型参数和持久缓冲区。
    best_epoch = 0  # 初始化最佳轮次为 0，训练轮次使用从 0 开始的编号。
    best_psnr = 0.0  # 初始化最佳 PSNR 为 0.0，用于和每轮验证结果比较。

    for epoch in range(args.num_epochs):  # 循环指定轮数，epoch 从 0 到 num_epochs-1。
        model.train()  # 切换到训练模式；本模型没有 Dropout/BatchNorm，但仍遵循标准训练流程。
        epoch_losses = AverageMeter()  # 新建本轮损失统计器，每轮从零开始统计。

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:  # 创建以样本数计量的进度条；总数排除了 drop_last 丢弃的尾部样本，% 表示取余。
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))  # 显示当前轮次与最后轮次，例如 epoch: 0/399。
            # 计算pth里边的数据
            for data in train_dataloader:  # 遍历训练加载器，每次获得一批配对样本；NumPy 数组会由默认整理逻辑转换为张量。
                inputs, labels = data  # 将这一批数据拆为退化输入 inputs 和清晰目标 labels。

                inputs = inputs.to(device)  # 把输入批次移动到与模型相同的 CPU 或 GPU。
                labels = labels.to(device)  # 把目标批次移动到相同设备，确保损失运算可以进行。

                preds = model(inputs)  # 调用模型做前向计算，得到与输入同形状的预测图像。

                loss = criterion(preds, labels)  # 计算预测与清晰目标之间的均方误差，得到标量张量。

                epoch_losses.update(loss.item(), len(inputs))  # item() 将损失转成 Python 数值；按本批样本数加权，更新本轮平均损失。

                optimizer.zero_grad()  # 清除上一次更新留下的梯度，防止梯度意外累积。
                loss.backward()  # 反向传播，计算损失对每个可训练参数的梯度。
                optimizer.step()  # 让 Adam 根据梯度更新模型参数，完成一次训练步骤。

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))  # 在进度条后显示累计平均损失，格式保留小数点后 6 位。
                t.update(len(inputs))  # 把进度向前推进本批样本数，而不是仅增加 1。

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))  # 每轮训练后保存当前参数到 epoch_轮次.pth；该文件不含优化器状态。

        model.eval()  # 切换到评估模式；这一步本身不会关闭梯度计算。
        epoch_psnr = AverageMeter()  # 新建本轮验证 PSNR 的平均值统计器。

        for data in eval_dataloader:  # 遍历验证集，每批只有一张完整图像。
            inputs, labels = data  # 拆分验证输入与清晰参考图像。

            inputs = inputs.to(device)  # 将验证输入移动到计算设备。
            labels = labels.to(device)  # 将清晰参考图像移动到相同设备。

            with torch.no_grad():  # 在此代码块中关闭梯度记录，验证时减少内存开销。
                preds = model(inputs).clamp(0.0, 1.0)  # 计算预测并截断到 0～1，符合归一化图像的取值范围。

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))  # 计算预测相对清晰目标的 PSNR，并按图像数统计平均值。

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))  # 输出本轮验证集的平均 PSNR，保留两位小数。

        if epoch_psnr.avg > best_psnr:  # 如果当前验证平均 PSNR 超过历史最佳值，就更新最佳记录。
            best_epoch = epoch  # 记录得到最佳结果的轮次。
            best_psnr = epoch_psnr.avg  # 记录新的最佳验证 PSNR。
            best_weights = copy.deepcopy(model.state_dict())  # 深拷贝当前权重，避免后续训练更新同一组张量而覆盖最佳结果。

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))  # 全部训练结束后，输出最佳轮次及其验证 PSNR。
    # torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))  # 将最佳模型参数保存为 best.pth，供 test.py 加载使用。
    torch.save(best_weights, os.path.join(args.outputs_dir, 'jyx.pth'))