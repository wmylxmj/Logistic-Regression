# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 19:35:55 2018

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt

#增加常数项1.0
def LoadDataSet():
    DataMatrix = []
    LabelMatrix = []
    fr = open('testset.txt')
    for line in fr.readlines():
        LineArray = line.strip().split()
        DataMatrix.append([1.0, float(LineArray[0]), float(LineArray[1])])
        LabelMatrix.append(int(LineArray[2]))
    return DataMatrix, LabelMatrix

def Sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

#找到最适合的三个权重值w0, w1, w2使得输出标签h接近于真实标签
def GradAscent(testmatrix, testlabels):
    DataMatrix = np.mat(testmatrix)
    #转置
    LabelMatrix = np.mat(testlabels).transpose()
    m,n = np.shape(DataMatrix)
    #步长
    Alpha = 0.001
    MaxCycles = 500
    #列向量
    Weights = np.ones((n,1))
    for k in range(MaxCycles):
        #输出
        h = Sigmoid(DataMatrix * Weights)
        #误差 = 目标 - 当前
        error = (LabelMatrix - h)
        #修改权重值
        Weights = Weights + Alpha * DataMatrix.transpose() * error
    return Weights

DataMatrix, LabelMatrix = LoadDataSet()
#100*3
print(DataMatrix)
print(LabelMatrix)
print(GradAscent(DataMatrix, LabelMatrix))

#当x=0的时候，sigmoid（x）产生阶跃，x=0为临界点
#输入x=w0x0+w1x1+w2x2
#解方程w0x0+w1x1+w2x2=0
#求出x2=f(x1)
#x2 = (-w0x0-w1*x1)/w2
#为划分直线的直角坐标方程
def PlotBestFitLine(datamatrix, labelmatrix, weights):
    DataArray = np.array(datamatrix)
    n = np.shape(DataArray)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    #画点
    for i in range(n):
        if int(labelmatrix[i])==1:
            xcord1.append(DataArray[i,1])
            ycord1.append(DataArray[i,2])
        else:
            xcord2.append(DataArray[i,1])
            ycord2.append(DataArray[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    #draw the line
    x = np.arange(-3.0, 3.0, 0.1)
    #0 = w0x0 + w1x1 + w2x2
    #x2 = f(x1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
Weights = GradAscent(DataMatrix, LabelMatrix)
Weights.getA()
PlotBestFitLine(DataMatrix, LabelMatrix, Weights.getA())
    
def StocGradAscent0(datamatrix, labelmatrix):
    m,n = np.shape(datamatrix) 
    Alpha = 0.01
    Weights = np.ones(n)
    for i in range(m):
        h = Sigmoid(sum(datamatrix[i]*Weights))
        error = labelmatrix[i] - h
        Weights = Weights + Alpha * error * datamatrix[i]
    return Weights
    
Weights = StocGradAscent0(np.array(DataMatrix), LabelMatrix)
print(Weights)
PlotBestFitLine(DataMatrix, LabelMatrix, Weights)

def StocGradAscent1(datamatrix, labelmatrix, numberiter=150):
    m,n = np.shape(datamatrix) 
    Weights = np.ones(n)
    for i in range(numberiter):         
        DataIndex = list(range(m))
        for j in range(m):
            Alpha = 4/(1.0+i+j)+0.01
            RandomIndex = int(np.random.uniform(0, len(DataIndex)))
            h = Sigmoid(sum(datamatrix[RandomIndex]*Weights))
            error = labelmatrix[RandomIndex] - h
            Weights = Weights + Alpha * error * datamatrix[RandomIndex]
            del(DataIndex[RandomIndex])
    return Weights

Weights = StocGradAscent1(np.array(DataMatrix), LabelMatrix)
print(Weights)
PlotBestFitLine(DataMatrix, LabelMatrix, Weights)

print('Test!')
for i in range(15):
    Weights = StocGradAscent1(np.array(DataMatrix), LabelMatrix)
    print(Weights)
    PlotBestFitLine(DataMatrix, LabelMatrix, Weights)
    
def Classify(X, weights):
    Prob = Sigmoid(sum(X*weights))
    if Prob > 0.5:
        return 1.0
    else:
        return 0.0
