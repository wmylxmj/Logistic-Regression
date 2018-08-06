# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 19:35:55 2018

@author: wmy
"""

import numpy as np

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

def GradAscent(testmatrix, testlabels):
    DataMatrix = np.mat(testmatrix)
    LabelMatrix = np.mat(testlabels).transpose()
    m,n = np.shape(DataMatrix)
    Alpha = 0.001
    MaxCycles = 500
    #列向量
    Weights = np.ones((n,1))
    for k in range(MaxCycles):
        h = Sigmoid(DataMatrix * Weights)
        error = (LabelMatrix - h)
        Weights = Weights + Alpha * DataMatrix.transpose() *error
    return Weights

DataMatrix, LabelMatrix = LoadDataSet()
print(DataMatrix)
print(LabelMatrix)
print(GradAscent(DataMatrix, LabelMatrix))

