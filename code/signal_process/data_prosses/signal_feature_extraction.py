import numpy as np
import scipy.stats as sp
import pywt
from scipy.signal import argrelextrema

def hjorth(input):
    '''
    :param input: [batch signal]
    :return:
    '''
    realinput = input
    lenth = len(realinput)
    hjorth_activity = np.zeros(len(realinput))
    hjorth_mobility = np.zeros(len(realinput))
    hjorth_diffmobility = np.zeros(len(realinput))
    hjorth_complexity = np.zeros(len(realinput))
    diff_input = np.diff(realinput)
    diff_diffinput = np.diff(diff_input)
    k = 0
    for signal in realinput:
        hjorth_activity[k] = np.var(signal)
        hjorth_mobility[k] = np.sqrt(np.var(diff_input[k]) / hjorth_activity[k])
        hjorth_diffmobility[k] = np.sqrt(np.var(diff_diffinput[k]) / np.var(diff_input[k]))
        hjorth_complexity[k] = hjorth_diffmobility[k] / hjorth_mobility[k]
        k = k + 1

    # returning hjorth activity, hjorth mobility , hjorth complexity
    return np.sum(hjorth_activity) / lenth, np.sum(hjorth_mobility) / lenth, np.sum(
        hjorth_complexity) / lenth


#Kurtosis, 2nd Diff Mean, 2nd Diff Max

def my_kurtosis(inputs):
    b = inputs
    lenth = len(b)
    output = np.zeros(len(b))
    k = 0
    for i in b:
        mean_i = np.mean(i)
        std_i = np.std(i)
        t = 0.0
        for j in i:
            # Formula: (1/N)*(summation(x_i-mean)/standard_deviation)^4-3
            t += (pow((j - mean_i) / std_i, 4) - 3)
        kurtosis_i = t / len(i)
        # Saving the kurtosis in the array created
        output[k] = kurtosis_i
        # Updating the current row no.
        k += 1
    return np.sum(output) / lenth


def secDiffMean(inputs):
    b = inputs
    lenth = len(b)

    output = np.zeros(len(b))
    temp1 = np.zeros(len(b[0]) - 1)
    k = 0
    for i in b:
        t = 0.0
        for j in range(len(i) - 1):
            temp1[j] = abs(i[j + 1] - i[j])
        for j in range(len(i) - 2):
            t += abs(temp1[j + 1] - temp1[j])
        output[k] = t / (len(i) - 2)
        k += 1
    return np.sum(output) / lenth


def secDiffMax(inputs):
    b = inputs
    lenth = len(b)
    output = np.zeros(len(b))
    temp1 = np.zeros(len(b[0]) - 1)
    k = 0
    for i in b:
        for j in range(len(i) - 1):
            temp1[j] = abs(i[j + 1] - i[j])
        t = temp1[1] - temp1[0]
        for j in range(len(i) - 2):
            if abs(temp1[j + 1] - temp1[j]) > t:
                t = temp1[j + 1] - temp1[j]

        output[k] = t
        k += 1
    return np.sum(output) / lenth


def coeff_var(inputs):
    b = inputs
    lenth = len(b)
    output = np.zeros(len(b))
    k = 0
    for i in b:
        mean_i = np.mean(i)
        std_i = np.std(i)
        output[k] = std_i / mean_i
        k = k + 1
    return np.sum(output) / lenth


def skewness(inputs):
    data = inputs
    lenth = len(data)
    skew_array = np.zeros(len(data))
    index = 0
    for i in data:
        skew_array[index] = sp.stats.skew(i, axis=0, bias=True)
        index += 1
    return np.sum(skew_array) / lenth


def first_diff_mean(inputs):
    data = inputs
    lenth = len(data)
    diff_mean_array = np.zeros(len(data))
    index = 0
    for i in data:
        sum = 0.0
        for j in range(len(i) - 1):
            sum += abs(i[j + 1] - i[j])
        diff_mean_array[index] = sum / (len(i) - 1)
        index += 1
    return np.sum(diff_mean_array) / lenth



def first_diff_max(inputs):
    data = inputs
    lenth = len(data)
    diff_max_array = np.zeros(len(data))
    first_diff = np.zeros(len(data[0]) - 1)
    index = 0
    for i in data:
        max = 0.0
        for j in range(len(i) - 1):
            first_diff[j] = abs(i[j + 1] - i[j])
            if first_diff[j] > max:
                max = first_diff[j]
        diff_max_array[index] = max
        index += 1
    return np.sum(diff_max_array) / lenth




# # Wavelet transform features
def wavelet_features(epoch):
    lenth = len(epoch)
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy = []
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    for i in range(lenth):
        cA, cD = pywt.dwt(epoch[i, :], 'coif1')
        cA_values.append(cA)
        cD_values.append(cD)
    # calculating the coefficients of wavelet transform.
    for x in range(lenth):
        cA_mean.append(np.mean(cA_values[x]))
        cA_std.append(np.std(cA_values[x]))
        cA_Energy.append(np.sum(np.square(cA_values[x])))
        cD_mean.append(
            np.mean(cD_values[x]))
        cD_std.append(np.std(cD_values[x]))
        cD_Energy.append(np.sum(np.square(cD_values[x])))
        Entropy_D.append(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x]))))
        Entropy_A.append(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x]))))
    return np.sum(cA_mean) / lenth, np.sum(cA_std) / lenth, np.sum(cD_mean) / lenth, np.sum(cD_std) / lenth, np.sum(
        cA_Energy) / lenth, np.sum(cD_Energy) / lenth, np.sum(Entropy_A) / lenth, np.sum(Entropy_D) / lenth



#Variance and Mean of Vertex to Vertex Slope

def first_diff(inputs):
    b = inputs
    c = np.diff(b)
    return c

def slope_mean(inputs):
    b = inputs
    lenth = len(inputs)
    output = np.zeros(len(b))
    k = 0
    for i in b:
        x = i
        # amp_max = i[argrelextrema(x, np.greater)[0]]  # storing maxima value
        t_max = argrelextrema(x, np.greater)[0]
        # amp_min = i[argrelextrema(x, np.less)[0]]  # storing minima value
        t_min = argrelextrema(x, np.less)[0]
        t = np.concatenate((t_max, t_min), axis=0)
        t.sort()  # sort on the basis of time

        amp = np.zeros(len(t))
        res = np.zeros(len(t) - 1)
        for l in range(len(t)):
            amp[l] = i[t[l]]

        amp_diff = first_diff(amp)

        t_diff = first_diff(t)

        for q in range(len(amp_diff)):
            res[q] = amp_diff[q] / t_diff[q]
        output[k] = np.mean(res)
        k = k + 1
    return np.sum(output) / lenth


def slope_var(inputs):
    b = inputs
    lenth = len(b)
    output = np.zeros(len(b))
    k = 0
    for i in b:
        x = i
        # amp_max = i[argrelextrema(x, np.greater)[0]]  # storing maxima value
        t_max = argrelextrema(x, np.greater)[0]  # storing time for maxima
        # amp_min = i[argrelextrema(x, np.less)[0]]  # storing minima value
        t_min = argrelextrema(x, np.less)[0]  # storing time for minima value
        t = np.concatenate((t_max, t_min), axis=0)  # making a single matrix of all matrix
        t.sort()  # sorting according to time

        amp = np.zeros(len(t))
        res = np.zeros(len(t) - 1)
        for l in range(len(t)):
            amp[l] = i[t[l]]

        amp_diff = first_diff(amp)

        t_diff = first_diff(t)

        for q in range(len(amp_diff)):
            res[q] = amp_diff[q] / t_diff[q]  # calculating slope

        output[k] = np.var(res)
        k = k + 1
    return np.sum(output) / lenth



if __name__ == '__main__':
    data = np.random.random([1, 1024])
    print(data.shape)
    a = wavelet_features(data[0])
    print(a.shape)
    print(a)