# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:22:17 2025

@author: Z
"""
import numpy as np
import math
import time
import scipy.spatial.distance as dis

def calcE(X, coli, colj, flag):
    sum = 0.0
    if flag == 0:
        sum = np.sum((X[:, coli]-X[:, colj])**2)
        return math.sqrt(sum)
    elif flag == 1:
        s = np.linalg.norm(X[:, coli])*np.linalg.norm(X[:, colj])
        if s == 0:
            sum = 0.0
        else:
            sum = np.dot(X[:, coli], X[:, colj])/s
        return sum
    else:
        t = np.dot(X[:, coli], X[:, coli])+np.dot(X[:, colj], X[:, colj]) - \
            (np.linalg.norm(X[:, coli])*np.linalg.norm(X[:, colj]))
        if t == 0:
            sum == 0.0
        else:
            sum = np.dot(X[:, coli], X[:, colj])/t
        return sum


def Euclidean(X, n):

    Euclideandata = np.zeros([n, n])

    for i in range(n):
        for j in range(i, n):
            Euclideandata[i, j] = calcE(X, i, j, 0)
            Euclideandata[j, i] = Euclideandata[i, j]
    Euclidean_distance = []

    for i in range(n):
        sum = np.sum(Euclideandata[i, :])
        Euclidean_distance.append(sum/n)

    return Euclidean_distance

def correlation(X, n):
    X=np.array(X).T
    distance = dis.pdist(X, "correlation")
    distance_matrix = dis.squareform(distance)
    Correlation_distance = []

    for i in range(n):
        sum = np.sum(distance_matrix[i, :])
        Correlation_distance.append(sum/n)
        
    return Correlation_distance


def cos(X, n):

    Cosinedata = np.zeros([n, n])

    for i in range(n):
        for j in range(i, n):
            Cosinedata[i, j] = calcE(X, i, j, 1)
            Cosinedata[j, i] = Cosinedata[i, j]
    Cosine_distance = []

    for i in range(n):
        sum = np.sum(Cosinedata[i, :])
        Cosine_distance.append(sum/n)
    return Cosine_distance

def varience(data,avg1,col1,avg2,col2):

    return np.average((data[:,col1]-avg1)*(data[:,col2]-avg2))

def Person(X, y, n):
    feaNum=n
    #label_num=len(y[0,:])
    label_num=1
    PersonData=np.zeros([n])
    for i in range(feaNum):
        for j in range(feaNum,feaNum+label_num):
            #print('. ', end='')
            average1 = np.average(X[:,i])
            average2 = np.average(y)
            yn=(X.shape)[0]
            y=y.reshape((yn,1))
            dataset = np.concatenate((X,y),axis=1)
            numerator = varience(dataset, average1, i, average2, j);
            denominator = math.sqrt(
                varience(dataset, average1, i, average1, i) * varience(dataset, average2, j, average2, j));
            if (abs(denominator) < (1E-10)):
                PersonData[i]=0
            else:
                PersonData[i]=abs(numerator/denominator)
    
    return list(PersonData)

def Tanimoto(X, n):

    Tanimotodata = np.zeros([n, n])

    for i in range(n):
        for j in range(i, n):
            Tanimotodata[i, j] = calcE(X, i, j, 2)
            Tanimotodata[j, i] = Tanimotodata[i, j]
    Tanimoto_distance = []

    for i in range(n):
        sum = np.sum(Tanimotodata[i, :])
        Tanimoto_distance.append(sum/n)

    return Tanimoto_distance

#二分查找索引信息
def binary_search(arr, low, high, new_item):
    if high <= low:
        return low
    mid = (low + high) // 2
    if arr[mid] == new_item:
        return mid
    elif arr[mid] < new_item:
        return binary_search(arr, mid + 1, high, new_item)
    else:
        return binary_search(arr, low, mid, new_item)

def find_index(arr, new_item):
    return binary_search(arr, 0, len(arr) - 1, new_item)

def split_X_byFeature(mrmd, feature_cate):
    m = len(feature_cate)
    results = [[] for _ in range(m)]
    for i in mrmd:
        index = find_index(feature_cate,int(i[0]))
        results[index].append(i[0])
    return results

def run(X, y, feature_cate, k=1):
    features_name = [str(i) for i in range(X.shape[1])]
    print("The demension of the feature is {}".format(len(features_name)))
    mrmrValue = []
    n = len(features_name)-1
    print(time.ctime())
    # ee = testE(X, n)
    # max_ee = max(ee)
    # ee = [x/max_ee for x in ee]
    # print("Euclidean is calculated")
    # print(time.ctime())
    e = Euclidean(X, n)
    max_e = max(e)
    e = [x/max_e for x in e]
    print("Euclidean is calculated")
    print(time.ctime())
    # cc = cos(X, n)
    # max_cc = max(cc)
    # cc = [x/max_cc for x in cc]
    # print(time.ctime())
    c = cos(X, n)
    max_c = max(c)
    c = [x/max_c for x in c]
    print("Cosine is calculated")
    print(time.ctime())
    t = Tanimoto(X, n)
    max_t = max(t)
    t = [x/max_t for x in t]
    print("Tanimoto is calculated")
    print(time.ctime())
    # cor = correlation(X, n)
    cor = Person(X, y, n)
    max_cor = max(cor)
    cor = [x/max_cor for x in cor]
    print("Correlation is calculated")
    print(time.ctime())
    for j, z, w, v in zip(e, c, t, cor):
        mrmrValue.append(j/6+z/6+w/6+v/2)
    # for j,z in zip(e,c):
    #     mrmrValue.append((j+z)/2)
    mrmr_max = max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    # features 和 mrmrvalue绑定
    mrmrValue = [(i, j) for i, j in zip(features_name[1:], mrmrValue)]
    # 按mrmrValue 由大到小排序
    mrmd = sorted(mrmrValue, key=lambda x: x[1], reverse=True)
    m = int(X.shape[1]*k) + 1   # 以k作为阈值，保留特征数量
    # print(mrmd)
    # print(time.ctime())
    Eachfeature = split_X_byFeature(mrmd[:m], feature_cate=feature_cate)
    X_new = []
    for i in mrmd[:m]:
        j = i[0]
        X_new.append(X[:, int(j)].tolist())
    return np.array(X_new).T, Eachfeature
