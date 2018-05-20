# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


"""绘制混淆矩阵图"""
def confusion_matrix_plot_matplotlib(y_truth, y_predict, cmap=plt.cm.Blues):
    """Matplotlib绘制混淆矩阵图
    parameters
    ----------
        y_truth: 真实的y的值, 1d array
        y_predict: 预测的y的值, 1d array
        cmap: 画混淆矩阵图的配色风格, 使用cm.Blues，更多风格请参考官网
    """
    cm = confusion_matrix(y_truth, y_predict)
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print(cm_normalized)
    plt.matshow(cm, cmap=cmap)  # 混淆矩阵图
    plt.colorbar()  # 颜色标签
 
    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm_normalized[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
 
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签