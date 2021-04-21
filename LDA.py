import jieba
import jieba.posseg as jp
from gensim import corpora, models
# -*- coding: utf-8 -*-
import os
import gensim.corpora
from gensim.models import Phrases
from gensim.models import phrases
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import pyLDAvis.sklearn

outputDir = "./out/"  # 结果输出地址
filename = "data_cut.txt"
corpus = []
f = open(filename, 'r', encoding='utf-8')  # 语料库 按题读入成[]\
for line in f.readlines():
    line = line.strip('\n')
    corpus.append(line.split(' '))
size = 150
num_topics = 12
# 5得出的图较好
ldas = []

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

def createmodel(num_topics, corpus_vec, dictionary):
    for i in range(1, num_topics):
        lda = models.ldamodel.LdaModel(corpus=corpus_vec, id2word=dictionary, num_topics=i)
        ldas.append(lda)

def perplexity_visible_model(ldas,corpus_vec):
    '''
    @description: 绘制困惑度-主题数目曲线
    @param {type}
    @return:
    '''
    x_list = []
    y_list = []
    for i in range(1, num_topics):
        perplexity = ldas[i-1].log_perplexity(corpus_vec)
        print(perplexity)
        x_list.append(i)
        y_list.append(perplexity)
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')
    plt.legend(('perplexity_values'), loc='best')
    plt.plot(x_list, y_list)
    plt.show()

def visible_model(ldas,corpus,dictionary):
    '''
    @description: 可视化模型
    @param :topic_num:主题的数量
    @param :data_num:数据的量
    @return: 可视化lda模型
    '''

    x_list = []
    y_list = []
    for i in range(1, num_topics):
        cv_tmp = CoherenceModel(model=ldas[i-1], texts=corpus, dictionary=dictionary, coherence='c_v')
        x_list.append(i)
        y_list.append(cv_tmp.get_coherence())
    plt.plot(x_list, y_list)
    plt.xlabel('num topics')
    plt.ylabel('coherence score')
    plt.legend(('coherence_values'), loc='best')
    plt.show()

if __name__ == "__main__":
    # 生成语料词典
    dictionary = corpora.Dictionary(corpus)
    # 生成稀疏向量集
    corpus_vec = [dictionary.doc2bow(words) for words in corpus]
    # LDA模型，num_topics设置聚类数，即最终主题的数量
    # lda = models.ldamodel.LdaModel(corpus=corpus_vec, id2word=dictionary, num_topics=num_topics)
    createmodel(num_topics,corpus_vec,dictionary)
    # # 可视化困惑度,一致度，找合适主题数 时间长
    # perplexity_visible_model(ldas, corpus_vec)
    # visible_model(ldas, corpus, dictionary)
    # 观察得到最佳主题数5
    lda = ldas[4]
    # # 可视化
    # vis = pyLDAvis.gensim.prepare(lda, corpus_vec, dictionary)
    # pyLDAvis.show(vis, open_browser=True)
    # 展示每个主题的前5的词语
    for topic in lda.print_topics(num_words=5):
        print(topic)
    # 推断每个语料库中的主题类别
    label_i = []
    for j in range(0, num_topics):
        label_i.append([])
    for e, values in enumerate(lda.inference(corpus_vec)[0]):
        topic_val = 0
        topic_id = 0
        for tid, val in enumerate(values):
            if val > topic_val:
                topic_val = val
                topic_id = tid
        label_i[topic_id].append(e+1)
        #题号从1开始
    result = []
    best_num = 5
    for i in range(0, best_num):
        result.append('类别' + '(' + str(i) + ')' + ':' + str(label_i[i]))
    output(result, outputDir, best_num, "LDA_")
    print('finish')