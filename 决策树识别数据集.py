# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:31:09 2021
decision tree
@author: hhh
"""
#%%
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from math import exp, log
from time import *
from random import sample

#%% data
datasets = [['青年', '否', '否', '一般', '否'],
              ['青年', '否', '否', '好', '否'],
              ['青年', '是', '否', '好', '是'],
              ['青年', '是', '是', '一般', '是'],
              ['青年', '否', '否', '一般', '否'],
              ['中年', '否', '否', '一般', '否'],
              ['中年', '否', '否', '好', '否'],
              ['中年', '是', '是', '好', '是'],
              ['中年', '否', '是', '非常好', '是'],
              ['中年', '否', '是', '非常好', '是'],
              ['老年', '否', '是', '非常好', '是'],
              ['老年', '否', '是', '好', '是'],
              ['老年', '是', '否', '好', '是'],
              ['老年', '是', '否', '非常好', '是'],
              ['老年', '否', '否', '一般', '否'],
              ]

labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']

df = pd.DataFrame(datasets, columns = labels)  

#%%
class my_dtree:
    #无默认值的参数放前面，有默认值的参数放后面
    def __init__(self, train_data, pruning:bool, method):
        #data
        self.data = train_data
        #数据个数
        self.m = self.data.shape[0]
        #保存所有feature名
        self.feature_names = list(train_data.columns)[:-1]
        #label名
        self.label_name = list(train_data.columns)[-1]
        #所有可能的label值
        self.labels = list(set(df[self.label_name]))
        #给定阈值
        self.eps = 0.1
        #选择方法
        self.method = method
        #下一级节点
        self.feature = self.next_feature()
        self.next = None
        self.pruning = pruning
        
        
    def split(self, feature_name):
        '''利用某个feature的不同取值将数据集分开，返回一个字典，该字典的键为属性所有取值，值为根据键生成的子怀表'''
        feature_values = list(set(self.data[feature_name])) #该属性所有可能值
        # 根据该属性所有可能值得到新的若干df
        splitted_data = [self.data[(self.data[feature_name] == feature_values[i])]\
                         for i in range(len(feature_values))]
        # 根据键，值生成字典，其中字典的键为属性值，字典的值为属性值对应的子dataframe
        return  dict(zip(feature_values, splitted_data))   
    
    def x_logx(self,a,b): #直接用log不好处理0出现的情况
        if a < 0 or b < 0:
            print('Does not fit the requirement of log')
        elif a == 0 or b == 0:
            return 0.0
        else:
            return (a/b)*log(b/a)
        
    def ent(self): #总熵
        # value_counts()函数返回每个值在该列出现的次数
        counts = self.data[self.label_name].value_counts()
        # 总频数
        sum_all = self.data.shape[0]
        return sum([self.x_logx(counts.iloc[i], sum_all)\
                    for i in range(counts.shape[0])])
            
    def f_ent(self, feature_name): #属性熵，C4.5方法中算信息增益比时要用
        # value_counts()函数返回每个值在该列出现的次数
        counts = self.data[feature_name].value_counts()
        # 总频数
        sum_all = self.data.shape[0]
        return sum([self.x_logx(counts.iloc[i], sum_all)\
                    for i in range(counts.shape[0])])
    
    def cond_ent(self, feature_name): #条件熵，feature_name为条件
        feature_values = list(set(self.data[feature_name])) #该属性所有可能值
        # 将每个属性值对应的类别数记录在一个列表中
        feature_label = [[self.data[(self.data[feature_name]==feature_values[i])\
                         & (self.data[self.label_name]==self.labels[j])].shape[0]\
                         for j in range(len(self.labels))]\
                         for i in range(len(feature_values))]
        # 每个类别的熵    
        print(feature_label)     
        print(len(feature_values))  
        # print(self.next_feature())
        ent_list = [sum([self.x_logx(feature_label[i][j], sum(feature_label[i]))\
                    for j in range(len(feature_label[i]))])\
                    for i in range(len(feature_values))]
        # 总熵
        cond_ent = sum([ent_list[i]*sum(feature_label[i])/self.m\
                    for i in range(len(feature_values))])
        print(cond_ent)
        return cond_ent
    
    def next_feature(self): #根据给定方法来选择最优决策属性
        if self.method == 'id3': #根据信息增益选择
            #选择信息增益最大，即条件熵最小的属性
            feature_name = min(self.feature_names\
                           , key=lambda name: self.cond_ent(name))
            self.feature = feature_name
            return feature_name
        
        elif self.method == 'c4.5': #根据信息增益比选择
            ent = self.ent()
            # 选择信息增益比最大的属性
            feature_name = max(self.feature_names, key=lambda name\
                           : (ent-self.cond_ent(name))/self.f_ent(name))
            self.feature = feature_name
            return feature_name
        else:
            print('Do not support '+self.method+' !')
            
    def next_tree(self): #得到下一级的若干节点，以字典形式给定
        feature_name = self.next_feature()
        print(feature_name)
        feature_values = list(set(self.data[feature_name]))
        next_dict = self.split(feature_name)
        # dict_values对象无法像列表一样进行索引，需要先把他转换成列表才行
        next_tree_list = [my_dtree(\
                    train_data = next_dict[feature_value],\
                    pruning = self.pruning, method = self.method)\
                    for feature_value in feature_values]
        return dict(zip(feature_values, next_tree_list))
    
    def train(self): #得到决策树
        if (len(self.labels) == 1) or (self.ent() <= self.eps):
            print(0)
            return 0
        else:
            # print(self.next_feature())
            print(1)
            # print(self.next_tree())
            self.next = self.next_tree()
            for value in self.next.values():
                value.train() 
            return 1
    
    def predict(self, x: pd.DataFrame):
        # 若该节点中仅有一个类别，则直接将其归为该类
        if len(self.labels) == 1:
            return self.labels[0]
        # 若该节点的总熵较小，则将其归为较多的一类
        elif self.ent() <= self.eps:
            # 所有类别的个数
            counts = self.data[self.label_name].value_counts()
            # 该分支最多的类别
            i = max(range(len(counts)), key=lambda x: counts[x])
            return list(counts.index)[i]
        # 若不存在以上两情况，则返回下一个节点的预测值，直到能够输出
        else: 
        # 不能直接用x[self.feature]，因为x[self.feature]为一个df，需要用iloc获取值
            return self.next[x[self.feature].iloc[0]].predict(x)
    
    
#%%
model = my_dtree(df, pruning = False, method = 'id3')
model.train()
x_list = ['青年', '否', '否', '一般']
x_features = ['年龄', '有工作', '有自己的房子', '信贷情况']
x = pd.DataFrame([x_list], columns = x_features)
# dict0 = model.split('年龄')
# print(model.ent())
# print(model.cond_ent('年龄'))
