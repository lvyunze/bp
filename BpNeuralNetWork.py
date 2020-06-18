#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : yunze
from numpy.random.mtrand import randint
import numpy as np


# 双曲函数
def tanh(value):
    return 1 / (1 + np.math.e ** (-value))


# 双曲函数导数
def tanhDer(value):
    tanhValue = tanh(value)
    return tanhValue * (1 - tanhValue)


# BP神经网络模型
class BpNeuralNetWorkModel:
    def __init__(self, trainningSet, label, layerOfNumber, studyRate):
        # 学习率
        self.studyRate = studyRate
        # 计算隐藏层神经元的数量'
        self.hiddenNeuronNum = int(np.sqrt(trainningSet.shape[1] + label.shape[1]) + randint(1, 10))
        # 层数据
        self.layers = []
        # 创建输出层
        currentLayer = Layer()
        currentLayer.initW(trainningSet.shape[1], self.hiddenNeuronNum)
        self.layers.append(currentLayer)

        # 创建隐藏层
        for index in range(layerOfNumber - 1):
            currentLayer = Layer()
            self.layers.append(currentLayer)
            # 输出层后面不需要求权重值
            if index == layerOfNumber - 2:
                break
            nextLayNum = 0

            # 初始化各个层的权重置
            if index == layerOfNumber - 3:
                # 隐藏层到输出层
                nextLayNum = label.shape[1]
            else:
                # 隐藏层到隐藏层
                nextLayNum = self.hiddenNeuronNum
            currentLayer.initW(self.hiddenNeuronNum, nextLayNum)
        """输出层的分类值"""
        currentLayer = self.layers[len(self.layers) - 1]
        currentLayer.label = label

    # 神经网络前向传播

    def forward(self, trainningSet):
        # 计算输入层的输出值
        currentLayer = self.layers[0]
        currentLayer.alphas = trainningSet
        currentLayer.caculateOutPutValues()

        preLayer = currentLayer
        for index in range(1, len(self.layers)):
            currentLayer = self.layers[index]
            # 上一层的out put values就是这一层的zValues
            currentLayer.zValues = preLayer.outPutValues
            # 计算alphas
            currentLayer.caculateAlphas()
            # 最后一层不需要求输出值，只要求出alpha
            if index == len(self.layers) - 1:
                break
            # 输入层计算out puts
            currentLayer.caculateOutPutValues()
            # 指向上一层的layer
            preLayer = currentLayer

    # 神经网络反向传播
    def backPropogation(self):
        layerCount = len(self.layers)

        # 输出层的残差值
        currentLayer = self.layers[layerCount - 1]
        currentLayer.caculateOutPutLayerError()

        # 输出层到隐藏层
        preLayer = currentLayer
        layerCount = layerCount - 1
        while layerCount >= 1:
            # 当前层
            currentLayer = self.layers[layerCount - 1]
            # 更新权重
            currentLayer.updateWeight(preLayer.errors, self.studyRate)
            if layerCount != 1:
                currentLayer.culateLayerError(preLayer.errors)
            layerCount = layerCount - 1
            preLayer = currentLayer


# 创建层
class Layer:
    def __init__(self):
        self.b = 0

    # 使用正态分布的随机值初始化w的值

    def initW(self, numOfAlpha, nextLayNumOfAlpha):
        self.w = np.mat(np.random.randn(nextLayNumOfAlpha, numOfAlpha))

    # 计算当前层的alphas

    def caculateAlphas(self):
        # alpha = f(z)
        self.alphas = np.mat([tanh(self.zValues[row1, 0]) for row1 in range(len(self.zValues))])
        # 求f'(z)的值（即f的导数值）
        self.zDerValues = np.mat([tanhDer(self.zValues[row1, 0]) for row1 in range(len(self.zValues))])

    # 计算out puts

    def caculateOutPutValues(self):
        # 计算当前层z = w * alpha的的下一层的输入值
        self.outPutValues = self.w * self.alphas.T + self.b

    # 计算输出层的残差

    def caculateOutPutLayerError(self):
        self.errors = np.multiply(-(self.label - self.alphas), self.zDerValues)
        print("输出层 ..." + str(self.alphas))

    # 计算其它层的残差

    def culateLayerError(self, preErrors):
        self.errors = np.mat([(self.w[:, column].T * preErrors.T * self.zDerValues[:, column])[0, 0] for column in
                              range(self.w.shape[1])])
    """更新权重"""

    def updateWeight(self, preErrors, studyRate):
        data = np.zeros((preErrors.shape[1], self.alphas.shape[1]))
        for index in range(preErrors.shape[1]):
            data[index, :] = self.alphas * (preErrors[:, index][0, 0])
        self.w = self.w - studyRate * data


"""
训练神经网络模型
@param train_set: 训练样本
@param labelOfNumbers: 训练总类别
@param layerOfNumber:  神经网络层数，包括输出层，隐藏层和输出层(默认只有一个输入层，隐藏层和输出层)
"""


def train(train_set, label, layerOfNumber=3, sampleTrainningTime=5000, studyRate=0.6):
    neuralNetWork = BpNeuralNetWorkModel(train_set, label, layerOfNumber, studyRate)
    # 训练数据
    for row in range(train_set.shape[0]):
        """当个样本使用梯度下降的方法训练sampleTrainningTime次"""
        for time in range(sampleTrainningTime):
            """前向传播"""
            neuralNetWork.forward(train_set[row, :])
            """反向传播"""
            neuralNetWork.backPropogation()


# 模拟数据
train_set = np.mat([[0.05, 0.1], [0.3, 0.2], [0.4, 0.5], [0.4, 0.8]])
labelOfNumbers = np.mat([0.1, 0.99])
# 确定层数
layerOfNumber = 3

# 调用函数并使用数据
train(train_set, labelOfNumbers, layerOfNumber)





