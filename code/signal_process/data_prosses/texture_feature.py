import numpy as np
import matplotlib.pyplot as plt


'''
基于直方图的信号纹理特征
'''


def signal_to_values(num_values, signal):
    '''
    #将数据划分成 num_values 等份
    :param num_values:  划分等份数目
    :param signal:       一维信号,array类型
    :return:
    '''
    signal = np.array(signal, dtype=np.float32)
    sig_min = signal.min()
    sig_max = signal.max()

    signal_vol = num_values * (signal - sig_min) / (sig_max - sig_min)
    signal_vol = np.floor(signal_vol) + 1

    signal_vol[signal_vol == num_values + 1] = num_values
    return signal_vol


def compute_histogram(num_values, signal_vol):
    '''
    计算直方图和每个信号值的概率
    :param num_values:
    :param signal_vol:[1, length]
    :return:
    '''

    signal_len = signal_vol.shape[1]
    print('signal_len: %s' % signal_len)
    #计算直方图

    vox_val_hist = np.zeros(num_values, )

    for this_vox_value in range(num_values):
        vox_val_hist[this_vox_value] = np.sum(signal_vol == (this_vox_value+1))

    #计算直方图的相对概率
    vox_val_probs = vox_val_hist / signal_len

    return vox_val_hist, vox_val_probs


def compute_histogram_metrices(vox_val_probs, num_values):
    '''
    函数用来计算基于基于直方图的信号纹理特征，输入是信号的直方图概率，输出是六个特征
    (1)Mean
    (2)Variance
    (3)Skewness
    (4)Kurtosis
    (5)Energy
    (6)Entropy
    :param vox_val_probs:
    :param num_values:
    :return:
    '''

    metrices_vect = {'Mean': None,
                     'Variance': None,
                     'Skewness': None,
                     'Kurtosis': None,
                     'Energy': None,
                     'Entropy': None}
    vox_val_indices = np.arange(1, num_values+1)

    #(1)Mean
    metrices_vect['Mean'] = np.sum(vox_val_indices * vox_val_probs)

    #(2)Variance
    metrices_vect['Variance'] = np.sum(((vox_val_indices - metrices_vect['Mean'])**2)*vox_val_probs)

    if metrices_vect['Variance'] > 0:
        #(3)Skewness
        metrices_vect['Skewness'] = np.sum(((vox_val_indices - metrices_vect['Mean'])**3)*vox_val_probs) / (metrices_vect['Variance']**(3/2))


        #(4)Kurtosis
        metrices_vect['Kurtosis'] = np.sum(((vox_val_indices - metrices_vect['Mean'])**4)*vox_val_probs) / (metrices_vect['Variance']**2)
        metrices_vect['Kurtosis'] = metrices_vect['Kurtosis'] - 3

    else:
        metrices_vect['Skewness'] = 0
        metrices_vect['Kurtosis'] = 0

    #(5)Energy
    metrices_vect['Energy'] = np.sum(vox_val_probs**2)

    #(6)Entropy
    hist_nz_bin_indices = (vox_val_probs != np.nan)
    metrices_vect['Entropy'] = - np.sum(vox_val_probs[hist_nz_bin_indices] * np.log(vox_val_probs[hist_nz_bin_indices]))

    return metrices_vect


'''
 基于GLCM的信号纹理特征
'''
def sig_glcm(signal_vol, kernal_size=1, stride=1):
    '''
    计算信号的灰度共生矩阵
    :param signal_vol: 信号
    :param kernal_size: 和当前位置的距离
    :param stride: 每次移动步长
    :return:
    '''
    signal_len = signal_vol.shape[1]
    width = signal_vol.max()
    width = int(width)
    glcm = np.zeros((width, width))
    i = 0
    while(True):
        if i + kernal_size < signal_len:
            l = signal_vol[0, i]
            k = signal_vol[0, i + kernal_size]
            # print(int(l), int(k))
            glcm[int(l-1), int(k-1)] += 1
            i += stride
        else:
            break

    return glcm

def compute_glcm_distributions(glcm):
    N_g = glcm.shape[0]
    p = glcm / np.sum(glcm)
    p_x = np.sum(p, axis=1)
    p_y = np.sum(p, axis=0)

    p_x = p_x[:, np.newaxis]
    p_y = p_y[:, np.newaxis]

    #p_{x+y}
    p_xpy = np.zeros((2*N_g, 1))
    for this_row in range(N_g):
        for this_col in range(N_g):
            p_xpy[this_row + this_col] = p_xpy[this_row + this_col] + p[this_row, this_col]

    #p_{x-y}
    p_xmy = np.zeros((N_g, 1))
    for this_row in range(N_g):
        for this_col in range(N_g):
            p_xmy[np.abs(this_row - this_col)] = p_xmy[np.abs(this_row - this_col)] + p[this_row, this_col]
    return p, p_x,  p_y, p_xpy, p_xmy, N_g


def compute_glcm_metrics(p, p_x,  p_y, p_xpy, p_xmy, N_g):
    metrics_vect ={
        'Angular Second Moment': None,
        'Contrast': None,
        'Correlation': None,
        'Sum of squares variance': None,
        'Inverse Difference moment': None,
        'Sum average': None,
        'Sum variance': None,
        'Sum Entropy': None,
        'Entropy': None,
        'Difference Variance': None,
        'Difference Entropy': None,
        'Information Correlation 1': None,
        'Information Correlation 2': None,
        'Autocorrelation': None,
        'Dissimilarity': None,
        'Cluster Shade': None,
        'Cluster Prominence': None,
        'Maximum Probability': None,
        'Inverse Difference': None}

    #SE Entropy
    SE = - np.sum(p_xpy[p_xpy > 0] * np.log(p_xpy[p_xpy > 0]))

    #Entropy This is also HXY used later
    HXY = - np.sum(p[p > 0] * np.log(p[p > 0]))

    # Needed for later
    pp_xy = p_x * p_y.T

    HXY1 = - np.sum(p[pp_xy > 0] * np.log(pp_xy[pp_xy > 0]))
    HXY2 = - np.sum(pp_xy[pp_xy > 0] * np.log(pp_xy[pp_xy > 0]) )
    HX = - np.sum(p_x[p_x > 0] * np.log(p_x[p_x > 0]))
    HY = - np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0]))


    n = [num for num in np.arange(N_g)]
    n = np.array(n, np.float32)
    n = n[:, np.newaxis]

    ndr, ndc = np.mgrid[0:N_g, 0:N_g]

    #(1)Angular Second Moment
    metrics_vect['Angular Second Moment'] = np.sum(p**2)

    #(2)Contrast
    temp = (n**2)*p_xmy
    print('temp:', temp.shape)
    metrics_vect['Contrast'] = np.sum(temp)

    #(3)Correlation
    nn = n+1
    mu_x = np.sum(nn * p_x)
    mu_y = np.sum(nn * p_y)
    sg_x = np.sqrt(np.sum((nn - mu_x)**2 * p_x))
    sg_y = np.sqrt(np.sum((nn - mu_y)**2 * p_y))

    if sg_x*sg_y == 0:
        metrics_vect['Correlation'] = 0
    else:
        metrics_vect['Correlation'] = (np.sum(ndr.reshape([-1, ]) * ndc.reshape([-1, ]) *
                                              p.reshape([-1, ])) - (mu_x * mu_y)) / (sg_x * sg_y)

    #(4)Sum of squares variance
    metrics_vect['Sum of squares variance'] = np.sum(((ndr.reshape([-1, ]) - np.mean(p.reshape([-1, ])))**2) *
                                                     p.reshape([-1, ]))
    #(5)Inverse Difference moment
    metrics_vect['Inverse Difference moment'] = np.sum((1. / (1+((ndr.reshape([-1, ]) - ndc.reshape([-1, ])) * p.reshape([-1, ])))))

    #(6)Sum average
    metrics_vect['Sum average'] = np.sum(np.arange(1, 2*N_g+1) * p_xpy.reshape([-1, ]))
    #(7)Sum variance
    metrics_vect['Sum variance'] = np.sum(((np.arange(1, 2 * N_g+1) - metrics_vect['Sum average'])**2) *
                                          p_xpy.reshape([-1, ]))

    #(8)Sum Entropy
    metrics_vect['Sum Entropy'] = SE
    #(9)Entropy
    metrics_vect['Entropy'] = HXY
    #(10)Difference Variance
    mu_xmy = np.sum(np.arange(0, N_g) * p_xmy.reshape([-1, ]))
    metrics_vect['Difference Variance'] = np.sum(((np.arange(0, N_g) - mu_xmy)**2) * p_xmy.reshape([-1, ]))
    #(11)Difference Entropy
    metrics_vect['Difference Entropy'] = - np.sum(p_xmy[p_xmy > 0] * np.log(p_xmy[p_xmy > 0]))
    #(12) and (13) Information Correlation
    if (np.max([HX, HY]) == 0):
        metrics_vect['Information Correlation 1'] = 0
    else:
        metrics_vect['Information Correlation 1'] = (HXY - HXY1) / np.max([HX, HY])

    metrics_vect['Information Correlation 2'] = np.sqrt((1 - np.exp(-2 * (HXY2 - HXY))))

    #(14)Autocorrelation
    metrics_vect['Autocorrelation'] = np.sum((ndr.reshape([-1, ]) * ndc.reshape([-1, ])) * p.reshape([-1, ]))
    #(15)Dissimilarity
    metrics_vect['Dissimilarity'] = np.sum(np.abs((ndr.reshape([-1, ]) - ndc.reshape([-1, ]))) * p.reshape([-1, ]))
    #(16)Cluster Shade
    metrics_vect['Cluster Shade'] = np.sum(((ndr.reshape([-1, ]) + ndc.reshape([-1, ]) - mu_x - mu_y)**3) *
                                           p.reshape([-1, ]))
    #(17)Cluster Prominence
    metrics_vect['Cluster Prominence'] = np.sum(((ndr.reshape([-1, ]) + ndc.reshape([-1, ]) - mu_x - mu_y)**4) *
                                                p.reshape([-1, ]))
    #(18)Maximum Probability
    metrics_vect['Maximum Probability'] = np.max(p)

    #(19)Inverse Difference
    metrics_vect['Inverse Difference'] = np.sum((1. / (1 + np.abs(ndr.reshape([-1, ]) - ndc.reshape([-1, ])))) *
                                                p.reshape([-1, ]))

    return metrics_vect
if __name__ == '__main__':
    data = np.random.random((1, 1024))
    signal = signal_to_values(128, data)
    print('归一化后信号：', signal.max(), signal.min())
    print('信号平均值：', np.sum(signal)/1024)

    v_hist, v_probs = compute_histogram(128, signal)
    print('统计直方图：', v_hist.max(), v_hist.min())
    print('信号概率：', v_probs.max(), v_probs.min(), np.sum(v_probs))

    nist_me = compute_histogram_metrices(v_probs, 128)
    print(nist_me)

    glcm = sig_glcm(signal, 1, 1)
    print(glcm.shape, glcm.max(), glcm.min())

    p, p_x, p_y, p_xpy, p_xmy, N_g = compute_glcm_distributions(glcm)
    print(p)
    print('p:', np.sum(p), np.sum(p_x), np.sum(p_y), p.shape, p_x.shape, p_y.shape, p_xpy.shape, p_xmy.shape)
    test = p_xpy[p_xpy > 0] * np.log(p_xpy[p_xpy > 0])
    print('test:', test.shape)
    testt = np.arange(1, 2 * 128 + 1) * p_xpy
    print('testt:', testt.shape)
    pp = p_x * p_y
    print('pp:', pp.shape)
    met = compute_glcm_metrics(p, p_x, p_y, p_xpy, p_xmy, N_g)
    print('met:', met)
