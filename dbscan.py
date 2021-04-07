# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import copy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

outputDir = "./out/"  # 结果输出地址
filename = "data_cut.txt"
corpus = []
f = open(filename, 'r', encoding='utf-8')  # 语料库 按题读入成[]\
for line in f.readlines():
    line = line.strip('\n')
    corpus.append(line)
save_model_name = 'word2vec.model'
size = 150

def countIdf(corpus):
    vectorizer = CountVectorizer() #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    words = vectorizer.get_feature_names()  #获取所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return weight, words

def output(result, outputDir, clusters, title):
    outputFile = title + 'out'
    type = '.txt'
    count = 0
    while (os.path.exists(outputDir + outputFile + type)):
        count += 1
        outputFile = title + 'out' + str(count)
    doc = open(outputDir + outputFile + type, 'w')
    for i in range(0, clusters):
        print(result[i], file=doc)
    doc.close()

def sort_tfidf():
    sorted_words_list = list()
    for i in range(len(weight)):
        words_dict = dict()
        sorted_words = list()
        for j in range(len(words)):
            if weight[i][j] != 0:
                words_dict[words[j]] = weight[i][j]
        # 建立每一句话中tfidf的值词典
        sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)  # 将每句话中的单词按照权重值进行排序
        sorted_words_list.append(sorted_words)
    return sorted_words_list

def PCA(weight, dimension):
    from sklearn.decomposition import PCA
    print('原有维度: ', len(weight[0]))
    pca = PCA(n_components=dimension)  # 初始化PCA
    weight_1 = pca.fit_transform(weight)  # 返回降维后的数据
    print('降维后维度: ', len(weight_1[0]))
    return weight_1

def dist(a,b):
    """
    :param a: 样本点
    :param b: 样本点
    :return: 两个样本点之间的欧式距离
    """
    length = len(a)
    sum = 0;
    for i in range(0,length):
        sum += math.pow(a[i]-b[i],2)
    return math.sqrt(sum)

def returnDk(matrix, k):
    """
    :param matrix: 距离矩阵
    :param k: 第k最近
    :return: 第k最近距离集合
    """
    Dk = []
    for i in range(len(matrix)):
        Dk.append(matrix[i][k])
    return Dk


def returnDkAverage(Dk):
    """
    :param Dk: k-最近距离集合
    :return: Dk的平均值
    """
    sum = 0
    for i in range(len(Dk)):
        sum = sum + Dk[i]
    return sum / len(Dk)


def CalculateDistMatrix(dataset):
    """
    :param dataset: 数据集
    :return: 距离矩阵
    """
    DistMatrix = [[0 for j in range(len(dataset))] for i in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            DistMatrix[i][j] = dist(dataset[i], dataset[j])
    return DistMatrix


def returnEpsCandidate(dataSet):
    """
    :param dataSet: 数据集
    :return: eps候选集合
    """
    DistMatrix = CalculateDistMatrix(dataSet)
    tmp_matrix = copy.deepcopy(DistMatrix)
    for i in range(len(tmp_matrix)):
        tmp_matrix[i].sort()
    EpsCandidate = []
    for k in range(1, len(dataSet)):
        Dk = returnDk(tmp_matrix, k)
        DkAverage = returnDkAverage(Dk)
        EpsCandidate.append(DkAverage)
    return EpsCandidate,DistMatrix


def returnMinptsCandidate(DistMatrix, EpsCandidate):
    """
    :param DistMatrix: 距离矩阵
    :param EpsCandidate: Eps候选列表
    :return: Minpts候选列表
    """
    MinptsCandidate = []
    for k in range(len(EpsCandidate)):
        tmp_eps = EpsCandidate[k]
        tmp_count = 0
        for i in range(len(DistMatrix)):
            for j in range(len(DistMatrix[i])):
                if DistMatrix[i][j] <= tmp_eps:
                    tmp_count = tmp_count + 1
        MinptsCandidate.append(tmp_count / len(weight))
    return MinptsCandidate


def returnClusterNumberList(dataset, EpsCandidate, MinptsCandidate):
    """
    :param dataset: 数据集
    :param EpsCandidate: Eps候选列表
    :param MinptsCandidate: Minpts候选列表
    :return: 聚类数量列表
    """
    np_dataset = np.array(dataset)  # 将dataset转换成numpy_array的形式
    ClusterNumberList = []
    for i in range(len(EpsCandidate)):
        clustering = DBSCAN(eps=EpsCandidate[i], min_samples=MinptsCandidate[i]).fit(np_dataset)
        num_clustering = max(clustering.labels_)
        ClusterNumberList.append(num_clustering)
    return ClusterNumberList

def dbscan_SC(weight,eps,min_sam):  # 待聚类点阵,聚类个数
    #聚类数量通过调这两个参数
    dbscan_model = DBSCAN(eps=eps,min_samples=min_sam)
    y = dbscan_model.fit_predict(weight)
    print(y)
    # 此处已经进行预测，y中为对应的类，如四个类，【0，0，1，2，3...】
    result = []
    for i in range(np.min(y), np.max(y)+1):
        label_i = []
        for j in range(0, len(y)):
            if y[j] == i:
                label_i.append(j + 1)
        result.append('类别' + '(' + str(i) + ')' + ':' + str(label_i))
    return result, np.max(y)+2

#v4 DBSCAN
weight, words = countIdf(corpus)
weight = PCA(weight, dimension=350)  # 将原始权重数据降维
#找适合eps minpts 计算慢
# DistMatrix = []
# EpsCandidate,DistMatrix = returnEpsCandidate(weight)
# MinptsCandidate = returnMinptsCandidate(DistMatrix, EpsCandidate)
# ClusterNumberList = returnClusterNumberList(weight, EpsCandidate, MinptsCandidate)
# print(EpsCandidate)
# print(MinptsCandidate)
# print('cluster number list is')
# print(ClusterNumberList)
#1.0956556287429138 1.65 __50类
#1.191462289340973,2.7055555555555557 __19类
#1.2349539051088525 4.05 __5类
#结论 不适合DBSCAN
result, clusters = dbscan_SC(weight,1.0956556287429138,1.65)
output(result, outputDir, clusters, "PCA_dbscan_SC_")
print('finish')