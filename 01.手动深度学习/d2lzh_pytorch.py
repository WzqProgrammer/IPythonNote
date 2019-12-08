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
import time
import sys
from torch import nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms


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


def load_data_fashion_mnist(batch_size):
    '''
    @description: 加载数据集   
    @param {type} 
    @return: train_iter, test_iter
    '''
    if sys.platform.startswith('win'):
        num_workers = 0  # 0 表示不用额外的进程来加速读取速度
    else:
        num_workers = 4

    mnist_train = torchvision.datasets.FashionMNIST(root='D:\CodeProjects\Datasets',
                                               train=True, download=False, 
                                               transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='D:\CodeProjects\Datasets',
                                              train=False, download=False, 
                                              transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


# softmax运算
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition   # 这里应用了广播机制


# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
                    
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()   
                
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n += y.shape[0]
            
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f"
             % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))


# 训练模型
def train_ch5(net, train_iter, test_iter, batch_size,
              optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.2f sec"%
             (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time()-start))


# x 的形状转换功能
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
        
    def forward(self, x):
        return x.view(x.shape[0], -1)


# 作图函数
def semilogy(x_vals, y_vals, x_label, y_label, 
             x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)


# 模型准确度评估
def evaluate_accuracy(data_iter, net,
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            # torch中定义的模型
            if isinstance(net, nn.Module):
                net.eval()   # 评估模式，会关闭丢弃层
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            # 自定义的模型
            else:
                # 如果有is_training这个参数
                if('is_training' in net.__code__.co_varnames):
                    # 将is_training设置成false
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


# 二维卷积运算
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j : j + w] * K).sum()
            
    return Y