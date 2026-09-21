# SRCNN PyTorch 复现与学习笔记

## 1. 项目整体理解

本项目是 SRCNN（Super-Resolution Convolutional Neural Network）的 PyTorch 实现，主要用于单图像超分辨率重建。

整体目标可以理解为：

```text
低质量图像 / Bicubic 插值图像
        ↓
      SRCNN
        ↓
生成超分辨率图像 SR
        ↓
与高清图像 HR 进行比较
```

项目整体的数据流如下：

```text
原始高清图片
    ↓
prepare.py
    ↓
生成 LR / HR 数据
    ↓
保存为 .h5 数据集
    ↓
datasets.py
    ↓
读取并预处理数据
    ↓
DataLoader
    ↓
train.py
    ↓
训练 SRCNN
    ↓
得到 .pth 模型权重
    ↓
test.py
    ↓
加载权重并测试图片
    ↓
得到 SR 图像和 PSNR
```

---

## 2. models.py —— 定义 SRCNN 模型

`models.py` 是整个项目中最核心的文件之一，它定义了 SRCNN 网络的结构。

SRCNN 主要处理 YCbCr 颜色空间中的 **Y 亮度通道**，因此输入和输出都是 **1 个通道**。

网络结构：

```text
输入 Y 通道
    ↓
Conv1：1 → 64，9×9
    ↓
ReLU
    ↓
Conv2：64 → 32，5×5
    ↓
ReLU
    ↓
Conv3：32 → 1，5×5
    ↓
输出 SR 的 Y 通道
```

所以 SRCNN 一共包含：

- 3 个 `Conv2d` 卷积层
- 2 个 `ReLU` 激活函数

可以简单理解为：

- 第一层：提取局部特征
- 第二层：对特征做进一步的非线性映射
- 第三层：根据特征重建图像

卷积层中的 `weight` 和 `bias` 会在训练过程中不断被修改，最终训练好的参数会保存到 `.pth` 文件中。

---

## 3. prepare.py —— 制作训练数据

`prepare.py` 的主要作用是：

> 将普通高清图片加工成模型训练所需要的 `.h5` 数据集。

基本流程：

```text
原始 HR 图片
    ↓
调整图片尺寸
    ↓
Bicubic 下采样
    ↓
得到真正的小尺寸 LR
    ↓
Bicubic 放大回 HR 尺寸
    ↓
得到模糊的模型输入
    ↓
切成许多小的图像 Patch
    ↓
分别保存 LR 和 HR
    ↓
生成 .h5 文件
```

其中 LR 和 HR 是一一对应的：

```text
lr[0] ↔ hr[0]
lr[1] ↔ hr[1]
lr[2] ↔ hr[2]
...
```

因此：

```text
prepare.py = 制作数据集
```

---

## 4. datasets.py —— 读取训练数据

`datasets.py` 负责从 `.h5` 文件中读取数据，并进行简单预处理。

例如：

```python
f['lr'][idx]
f['hr'][idx]
```

分别表示：

- 读取第 `idx` 个 LR 图像块
- 读取第 `idx` 个 HR 图像块

随后：

```python
/ 255.
```

将像素值从：

```text
0 ~ 255
```

归一化到：

```text
0 ~ 1
```

然后：

```python
np.expand_dims(..., 0)
```

增加一个通道维度。

例如：

```text
[33, 33]
```

变成：

```text
[1, 33, 33]
```

这里的 `1` 表示 Y 通道。

Dataset 最后返回：

```text
(LR, HR)
```

DataLoader 再将多个样本组合成一个 Batch。

例如 `batch_size = 16`：

```text
单个样本：
[1, 33, 33]

组合后：
[16, 1, 33, 33]
```

对应 PyTorch 常见图像格式：

```text
[N, C, H, W]
```

因此：

```text
datasets.py = 读取和简单处理数据
DataLoader = 将多个样本组合成 Batch
```

---

## 5. train.py —— 训练模型

`train.py` 是模型训练的主要程序。

训练数据中：

```text
inputs = LR / 模糊输入
labels = 对应的 HR 高清标准答案
```

一次训练的核心流程为：

```text
读取一个 Batch
    ↓
inputs / labels 放到 GPU
    ↓
preds = model(inputs)
    ↓
SRCNN 前向传播，得到预测 SR
    ↓
loss = criterion(preds, labels)
    ↓
使用 MSELoss 计算 SR 与 HR 的误差
    ↓
optimizer.zero_grad()
    ↓
清除上一轮梯度
    ↓
loss.backward()
    ↓
反向传播，计算各参数梯度
    ↓
optimizer.step()
    ↓
Adam 根据梯度修改模型参数
    ↓
进入下一个 Batch
```

最重要的 PyTorch 训练代码可以概括为：

```python
preds = model(inputs)

loss = criterion(preds, labels)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

其中：

- `MSELoss`：计算预测结果和 HR 相差多少
- `backward()`：计算每个模型参数应该往什么方向修改
- `Adam`：根据梯度决定模型参数如何更新
- `learning rate`：控制一次参数更新的幅度
- `optimizer.step()`：真正执行参数更新

---

## 6. Epoch 和 Batch

### Batch

一次拿一部分训练样本进行训练。

例如：

```text
batch_size = 16
```

表示一次使用 16 个训练样本。

每完成一个 Batch：

```python
optimizer.step()
```

就会更新一次模型参数。

### Epoch

整个训练集完整训练一遍。

例如：

```text
num_epochs = 400
```

表示模型会完整学习整个训练集 400 次。

因此模型参数不是一次计算出来的，而是在大量 Batch 和 Epoch 中逐渐学习出来的。

---

## 7. .pth —— 模型训练结果

`.pth` 文件主要用来保存 PyTorch 模型训练得到的参数。

例如 SRCNN 的 `.pth` 中可能包含：

```text
conv1.weight
conv1.bias

conv2.weight
conv2.bias

conv3.weight
conv3.bias
```

这些参数一开始通常是初始化得到的。

经过：

```text
前向传播
→ Loss
→ backward
→ Adam 更新
```

不断训练后，参数逐渐变化。

最终：

```python
torch.save(model.state_dict(), "xxx.pth")
```

将训练好的参数保存下来。

因此可以简单理解为：

```text
models.py
= 模型结构

.pth
= 模型训练后学到的参数

models.py + .pth
= 一个训练好的 SRCNN 模型
```

---

## 8. test.py —— 使用训练好的模型

`test.py` 负责加载训练好的 `.pth` 文件，并对图片进行超分辨率测试。

基本流程：

```text
读取测试图片
    ↓
调整图片尺寸
    ↓
Bicubic 下采样
    ↓
Bicubic 再放大
    ↓
得到模糊输入
    ↓
RGB → YCbCr
    ↓
取 Y 通道
    ↓
归一化到 0~1
    ↓
NumPy → Tensor
    ↓
[H, W] → [1, 1, H, W]
    ↓
加载 SRCNN 权重
    ↓
模型进行前向推理
    ↓
得到 SR 的 Y 通道
    ↓
计算 PSNR
    ↓
与原来的 Cb、Cr 通道重新组合
    ↓
YCbCr → RGB
    ↓
保存最终 SR 图片
```

---

## 9. utils.py —— 辅助函数

`utils.py` 主要保存一些辅助函数，不属于 SRCNN 模型的核心结构。

主要包括：

- RGB → YCbCr
- YCbCr → RGB
- PSNR 计算
- AverageMeter 平均值统计

其中经典 SRCNN 主要对 Y 通道进行超分辨率处理。

YCbCr：

```text
Y  = 亮度信息
Cb = 蓝黄色差信息
Cr = 红青色差信息
```

SRCNN：

```text
Y → SRCNN → 新的 Y
```

然后：

```text
新的 Y + 原来的 Cb + Cr
```

重新组合成彩色图像。

---

## 10. 我的理解总结

SRCNN 项目实际上可以分成几个阶段：

```text
prepare.py
→ 准备训练数据

datasets.py
→ 读取数据

models.py
→ 定义模型

train.py
→ 训练模型并得到权重

test.py
→ 加载权重进行测试

utils.py
→ 提供辅助函数
```

我目前对 SRCNN 的整体理解是：

模型通过 LR 图像预测 SR 图像，再将 SR 与真实 HR 计算 MSE Loss。

利用反向传播计算梯度，再由 Adam 优化器不断修改三层卷积中的 `weight` 和 `bias`。

经过大量 Batch 和 Epoch 后，模型得到一组较好的参数，最终保存为 `.pth` 文件。

测试时重新创建 SRCNN 网络，加载 `.pth` 中训练好的参数，即可对新的低质量图像进行超分辨率重建。

---

## 11. GitHub 仓库说明建议

如果将该项目上传到个人 GitHub，建议在 README 开头说明来源，例如：

```markdown
# SRCNN PyTorch 复现与学习记录

本仓库用于个人学习图像超分辨率以及 PyTorch 模型训练流程。

原始实现：
yjn870/SRCNN-pytorch

对应论文：
Image Super-Resolution Using Deep Convolutional Networks

本仓库主要记录：
- SRCNN 代码阅读与中文注释
- PyTorch 训练流程学习
- 自行训练模型权重
- SRCNN 实验结果
- 后续个人修改与实验
```

建议优先 Fork 原仓库，再加入自己的中文注释、学习笔记和实验结果，避免让仓库看起来像原始代码由自己编写。
