# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:03:17 2020

@author: hhh
"""

#%%     
'''#%%创建将代码分为不同代码块，ctrl+enter执行当前代码块，
shift+enter执行并将光标移到下一代码块'''
#In[1]
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from math import exp
from time import *
from random import sample

#%%
#In[2]
'''读取数据'''
iris = load_iris() #加载iris数据集
df=pd.DataFrame(iris.data,columns=iris.feature_names) #将数据集填入列表中
df["label"]=iris.target

'''每一类别的不同个数'''
df.label.value_counts() #对于panda中的series对象，value_counts()方法返回每一列的不同频数

df1=df[:100]
#修改lebel使其变成-1或1
#败招：不能这样索引，否则改变不了dataframe的值
# for i in range(new_df.shape[0]):
#     new_df.loc[i][-1]=2*new_df.loc[i][-1]-1

#正招：需要将两层索引【i】【-1】改为一层，这样才能改变dataframe的值
df1["label"]=2*df1["label"]-1

#%%
#In[3]
# '''画未分类的二维散点图'''
# plt.scatter(df[:50]['sepal length (cm)'], df[:50]['sepal width (cm)'], label='0')
# plt.scatter(df[50:100]['sepal length (cm)'], df[50:100]['sepal width (cm)'], label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend() #图位于右上角“plots”

#%%
#In[3]
class my_svm:
    def __init__(self,train_data,lambda_0=1.0,max_iter=100,kernel_type="linear"):
        #data为一个列表，最后一列为标签
        self.data = train_data
        (self.m,self.n)=self.data.shape
        self.lambda_0 = lambda_0
        self.kernel = kernel_type
        self.max_iter = max_iter
        self.index=[i for i in range(self.m)] #需要抽样获取下标
        
        #模型参数
        self.X=[self.x(i) for i in range(self.m)]
        self.Y=[self.y(i) for i in range(self.m)]
        self.alpha = np.ones([self.m,1])#*self.lambda_0/2
        self.b=0
        self.E=self.E_()
        self.eps=1.0e-7
        self.length=50
        
    def x(self,i:int):
        '''将数据转化为向量自变量'''
        return np.transpose(np.array([self.data.loc[i][:-1]]))
     
    def y(self,i:int):
        '''将标签转化为因变量'''
        return self.data.loc[i][-1]
    
    def K(self,x:np.array,y:np.array):
        if self.kernel == "linear":
            return np.dot(np.transpose(x),y)
        elif self.kernel == "poly":
            return (np.dot(np.transpose(x),y)+1)**2
        elif self.kernel == "gauss":
            return exp(-np.dot(np.transpose(x-y),x-y)/2)
    
    def eta(self,i:int,j:int):
        '''计算Kii+Kjj-2Kij'''
        return self.K(self.X[i],self.X[i])+self.K(self.X[j],self.X[j])-2*self.K(self.X[i],self.X[j])
    
    def g(self,x:np.array):
        '''给定x，算w^(T)*phi(x)+b'''
        g_ = self.b
        for i in range(self.m):
            g_ += self.alpha[i]*self.Y[i]*self.K(self.X[i],x)
        return g_    
                                                 
    def _E(self,i:int):
        '''计算Ei'''
        return self.g(self.X[i])-self.Y[i]
    
    def E_(self):
        '''将所有的Ei保存到一个列表里'''
        list_ = [self._E(i) for i in range(self.m)];
        return list_
    
    def violation(self,i:int):
        # 败招1，会导致每次循环的判断时间大大延长
        # '''判断违反KKT的程度'''
        # if alpha[i]==0:
        #     return max(0,1-self.y(i)*self.g(alpha,self.x(i),b))
        # elif alpha[i]==self.lambda_0:
        #     return max(0,self.y(i)*self.g(alpha,self.x(i),b)-1)
        # elif (0<alpha[i])and(alpha[i]<self.lambda_0):
        #     return abs(self.y(i)*self.g(alpha,self.x(i),b)-1)
        
        if self.alpha[i] == 0:
            return (1 > self.Y[i]*self.g(self.X[i]))
        elif self.alpha[i] == self.lambda_0:
            return (self.Y[i]*self.g(self.X[i]) > 1)
        else:
            return (self.Y[i]*self.g(self.X[i]) != 1)
    
    def cycle(self):
        '''外层循环，选择第一个变量'''
        #等式下标集
        equal_list=[i for i in range(self.m) if 0<self.alpha[i]<self.lambda_0]
        #不等式下标集
        unequal_list=[i for i in range(self.m) if not(0<self.alpha[i]<self.lambda_0)]
        #合并
        _list=equal_list+unequal_list
        #从等式下标集开始考虑是否满足KKT条件
        for i in range(self.m):
            if self.violation(_list[i]):
                break
        i1 = _list[i]
        
        '''内层循环，选择第二个变量'''
        if self.E[i1]>=0:
            i2=min(range(self.m), key=lambda x: self.E[x])
        else:
            i2=max(range(self.m), key=lambda x: self.E[x])
        return i1,i2
    
    def cut(self,L:float,H:float,alpha_0:float):
        '''对alpha_0剪枝,L为下限，H为上限'''
        if alpha_0<L:
            alpha_0=L
        elif alpha_0>H:
            alpha_0=H
        return alpha_0
    
    def train(self):
        #误差序列，用于决定是否停止训练
        alpha_delta_list=[]
        
        for t in range(self.max_iter):
            # i1, i2 = self.cycle()
            begin_time = time()
            alpha_i2_delta=0
            k=0
            while alpha_i2_delta==0:
                samples = sample(self.index,2)
                i1, i2 = samples[0], samples[1]
                
                #得到L与H，并判断是否需要重新选择i1与i2
                if self.eta(i1,i2)<=0:
                    continue
                delta = (self.E[i1]-self.E[i2])/self.eta(i1,i2)
                if self.Y[i1] == self.Y[i2]:
                    L = max(self.alpha[i1]+self.alpha[i2]-self.lambda_0,0)
                    H = min(self.lambda_0,self.alpha[i1]+self.alpha[i2])
                else:
                    L = max(self.alpha[i2]-self.alpha[i1],0)
                    H = min(self.lambda_0,self.lambda_0+self.alpha[i2]-self.alpha[i1])
                
                #更新alpha，
                #注意将数组中的元素赋给一个值时，该值会随该元素变化而变化
                #若将数组中元素运算处理过后再赋给一个值时，该值则不随该元素变化
                alpha_i1_old = self.alpha[i1]/1.0
                alpha_i2_old = self.alpha[i2]/1.0
                alpha_i2_unc = alpha_i2_old+self.Y[i2]*delta
                alpha_i2_new = self.cut(L,H,alpha_i2_unc)
                alpha_i2_delta = alpha_i2_new-alpha_i2_old
            end_time = time()
            time1_ = end_time-begin_time
            alpha_delta_list.append(abs(alpha_i2_delta))
            #若连续几次的更新量均较小，则认为训练结束，可跳出循环
            if t>=(self.length+1):
                del(alpha_delta_list[0])
                if max(alpha_delta_list)<=self.eps:
                    break
            
            self.alpha[i2] = alpha_i2_new
            # print(self.alpha[i1].shape)
            # print(alpha_i2_old.shape)
            # print(alpha_i2_new.shape)
            self.alpha[i1] += self.Y[i1]*self.Y[i2]*(alpha_i2_old-self.alpha[i2])
            alpha_i1_delta = self.alpha[i1]-alpha_i1_old
            alpha_i2_delta = self.alpha[i2]-alpha_i2_old
            
            #更新b,首先计算b1与b2，再根据情况更新
            b1 = self.b-self.E[i1]
            b1 -= self.Y[i1]*self.K(self.X[i1],self.X[i1])*(self.alpha[i1]-alpha_i1_old)
            b1 -= self.Y[i2]*self.K(self.X[i1],self.X[i2])*(self.alpha[i2]-alpha_i2_old)
            b2 = self.b-self.E[i2]
            b2 -= self.Y[i1]*self.K(self.X[i1],self.X[i2])*(self.alpha[i1]-alpha_i1_old)
            b2 -= self.Y[i2]*self.K(self.X[i2],self.X[i2])*(self.alpha[i2]-alpha_i2_old)
            b_old = self.b
            if 0<self.alpha[i1]<self.lambda_0:
                self.b = b1
            elif 0<self.alpha[i2]<self.lambda_0:
                self.b = b2
            else:
                self.b = (b1+b2)/2
                
            # self.E[i1]=self._E(i1)
            # self.E[i2]=self._E(i2)
            b_delta = self.b-b_old
            
            # 更新Ei
            begin_time2=time()
            for i in range(self.m):
                self.E[i] += b_delta
                self.E[i] += self.Y[i1]*alpha_i1_delta*self.K(self.X[i],self.X[i1])
                self.E[i] += self.Y[i2]*alpha_i2_delta*self.K(self.X[i],self.X[i2])
            end_time2=time()
            time2_=end_time2-begin_time2
            print("b from",t,"th iteration:",self.b)
            print("i1 from",t,"th iteration:",i1)
            print("i2 from",t,"th iteration:",i2)
            print("delta from",t,"th iteration:",delta)
            print("alpha_i2_delta from",t,"th iteration:",alpha_i2_delta)
            print("alpha_i2_old:",alpha_i2_old)
            print("alpha_i2_new:",self.alpha[i2])
            print("time of choosing variables from",t,"th iteration:",time1_)
            # print("time of updating error from",t,"th iteration:",time2_)
            print("\n")
        print(t)
        return "what a shot!"
                           
    def predict(self,x:np.array):
        if self.g(x)>=0:
            return 1
        else:
            return -1
        
    def score(self):
        score=0
        for i in range(self.m):
            if self.predict(self.X[i])==self.Y[i]:
                score+=1
        return score/self.m
    
#%%
begin_time=time()
model=my_svm(df1,max_iter=1000)
model.train()
end_time=time()
print("score=",model.score())
print("training time:",end_time-begin_time)


