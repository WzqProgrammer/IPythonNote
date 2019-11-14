'''
@Author: your name
@Date: 2019-11-08 14:08:38
@LastEditTime: 2019-11-08 18:55:57
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \手动深度学习\d2lzh_pytorch.py
'''
import torch
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')
    
    
def set_figsize(figsize = (3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 小批量随机读取数据样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 打乱索引顺序
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        # 选取每批样本，注意最后一次可能不足一个batch
        j = torch.LongTensor(indices[i : min(i+batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


def linreg(X, w, b):
    # 使用 mm 函数做矩阵乘法
    return torch.mm(X, w) + b


# 平方损失函数
def squared_loss(y_hat, y):
    # 返回的是向量，pytorch里的MSELoss没有除以2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 小批量随机梯度下降算法
# lr 学习率
def sgd(params, lr, batch_size):
    for param in params:
        # 注意这里更改param时用的是param.data
        param.data -= lr * param.grad / batch_size


# 将数值标签转成文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 在一行中画出多张图像和对应标签
def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(mnist_train, mnist_test, batch_size):
    '''
    @description: 加载数据集   
    @param {type} 
    @return: train_iter, test_iter
    '''
    if sys.platform.startswith('win'):
        num_workers = 0  # 0 表示不用额外的进程来加速读取速度
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter