#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 

import os 
from datetime import date
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.modules.pooling import *
import time
import random
import datetime
from torch.autograd import Variable
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils import *
from data_processing import * 
import scipy as sp 
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from matplotlib.ticker import NullFormatter
import matplotlib as mlp


# # Transform Learning

# In[2]:


def calOutShape(input_shape,ksize1=3,stride=1,maxpool1=False, mpl_ksize=2):
    #dim1 = int((input_shape[1]-kernel_size+2*pad)/stride) + 1
    mpl_stride = 2
    pad = ksize1//2
    dim2 = int((input_shape[2]-ksize1+2*pad)/stride) + 1
    if maxpool1 == True:
        dim2 = (dim2 - mpl_ksize)//mpl_stride + 1
    return dim2



class Transform(nn.Module):
    
    def __init__(self,input_shape, out_planes1 = 8, ksize1 = 3,maxpool1 = False, mpl_ksize=2):
        super(Transform, self).__init__()
        self.ksize1 = ksize1
        self.mpl_ksize = mpl_ksize
        self.out_planes1 = out_planes1
        self.init_T()
        self.maxpool1 = maxpool1
        self.input_shape = input_shape
        self.i = 1
        self.atom_ratio = 0.5
        self.init_X()
        self.gap = AdaptiveAvgPool1d(1)
        
        
        
    
    def init_T(self):
        conv = nn.Conv1d(1, out_channels = self.out_planes1, kernel_size = self.ksize1, stride=1, bias=True)
        self.T1 = conv._parameters['weight']
        ###print('T11 in transform: ',self.T1.shape)
        ###print(self.T1)
        
       
           
        
    def init_X(self):
        dim2 = calOutShape(self.input_shape,self.ksize1,stride = 1,maxpool1 = self.maxpool1, mpl_ksize = self.mpl_ksize)
        #print('dim2:',dim2)
        X_shape = [self.input_shape[0],self.out_planes1,dim2]
        self.X  = nn.Parameter(torch.randn(X_shape), requires_grad=True)
        self.num_features = self.out_planes1*dim2
        self.num_atoms = int(self.num_features*self.atom_ratio*5) 
        T_shape = [self.num_atoms,self.num_features]
        self.T = nn.Parameter(torch.randn(T_shape), requires_grad=True)
#         print('T : ')
#         print(self.T)
#         print('+'*100)
#         print('+'*100)
        
        
    def forward(self, inputs):
        ###print('inputs before conv1 : ', inputs.shape)
        x = F.conv1d(inputs, self.T1,padding = self.ksize1//2)
        ###print('x after conv1 : ', x.shape)
        if self.maxpool1:
            x = F.max_pool1d(x, 2)
        x = F.selu(x)
        y = torch.mm(self.T,x.view(x.shape[0],-1).t())
        return x, y
        
          
    def get_params(self):
        return self.T1, self.X, self.T
    
    
    def X_step(self):
        self.X.data = torch.clamp(self.X.data, min=0)


    def Z_step(self):
        self.Z.data = torch.clamp(self.Z.data, min=0)
        
        
    def get_TZ_Dims(self):
        return self.num_features,self.num_atoms, self.input_shape[0]
        
        
class Network(nn.Module): 
    def __init__(self,inputs_shape=(4,5,1),out_planes1 = 8, ksize1 = 3,
             maxpool1=False,  mpl_ksize=2,num_classes=2):
        super(Network, self).__init__()
        self.Transform1 = Transform(inputs_shape,out_planes1 = out_planes1,ksize1 = ksize1, 
                                    maxpool1=maxpool1, mpl_ksize=mpl_ksize)
        self.Transform2 = Transform(inputs_shape,out_planes1 = out_planes1, ksize1 = ksize1,
                                    maxpool1=maxpool1, mpl_ksize=mpl_ksize)
        self.Transform3 = Transform(inputs_shape,out_planes1 = out_planes1,ksize1 = ksize1,
                                    maxpool1=maxpool1,mpl_ksize=mpl_ksize)
        self.Transform4 = Transform(inputs_shape,out_planes1 = out_planes1,ksize1 = ksize1,
                                    maxpool1=maxpool1, mpl_ksize=mpl_ksize)
        self.Transform5 = Transform(inputs_shape,out_planes1 = out_planes1,ksize1 = ksize1,
                                    maxpool1=maxpool1, mpl_ksize=mpl_ksize)
        self.num_features,self.num_atoms, self.input_shape = self.Transform1.get_TZ_Dims()
        Z_shape = [self.num_atoms,self.input_shape]
        self.Z = nn.Parameter(torch.randn(Z_shape), requires_grad=True)
        self.pred_list = []
        self.init_TX()
        

    def init_TX(self):
        self.T11, self.X1, self.Tp1 = self.Transform1.get_params()
        self.T12, self.X2, self.Tp2 = self.Transform2.get_params()
        self.T13, self.X3, self.Tp3 = self.Transform3.get_params()
        self.T14, self.X4, self.Tp4 = self.Transform4.get_params()
        self.T15, self.X5, self.Tp5 = self.Transform5.get_params()
        self.T1 = torch.stack((self.T11,self.T12,self.T13,self.T14,self.T15),1)
        self.X = torch.stack((self.X1,self.X2,self.X3,self.X4,self.X5),1) 
        self.T = torch.stack((self.Tp1,self.Tp2,self.Tp3,self.Tp4,self.Tp5),1) 
        
        
        
    def forward(self,x):
        batch_size, no_of_series, no_of_days = x.shape
        
        close = np.reshape(x[:,0],(batch_size,1,no_of_days))
        out1,out1p = self.Transform1(close)
        #print('out1 shape : ', out1.shape)
        
        _open = np.reshape(x[:,1],(batch_size,1,no_of_days))
        out2,out2p = self.Transform2(_open)
        
        
        high = np.reshape(x[:,2],(batch_size,1,no_of_days))
        out3,out3p = self.Transform3(high)
        
        low = np.reshape(x[:,3],(batch_size,1,no_of_days))
        out4,out4p = self.Transform4(low)
        
        volume = np.reshape(x[:,4],(batch_size,1,no_of_days))
        out5, out5p = self.Transform5(volume)
        
        self.pred_list = [out1,out2,out3,out4,out5]

        gp1 = out1p + out2p + out3p + out4p + out5p
        return gp1
        #return gp 
    
    
    def X_step(self):
        self.Transform1.X_step()
        self.Transform2.X_step()
        self.Transform3.X_step()
        self.Transform4.X_step()
        self.Transform5.X_step()
        
        
    def Z_step(self):
        self.Z.data = torch.clamp(self.Z.data, min=0)
        
    
    def conv_loss_distance(self):
        self.init_TX()
        
        loss = 0.0
        X_list = [self.X1,self.X2,self.X3,self.X4,self.X5]
        for i in range(len(self.pred_list)): 
            X = X_list[i].view(X_list[i].size(0), -1)
            predictions = self.pred_list[i].view(self.pred_list[i].size(0), -1)
            Y = predictions - X
            loss += Y.pow(2).mean()
            
        return loss
    
        
    def conv_loss_logdet(self):

        loss = 0.0
        for T in [self.T11,self.T12,self.T13,self.T14,self.T15]:
            T = T.view(T.shape[0],-1)
            U, s, V = torch.svd(T)
            loss += -s.log().sum()
        return loss
        
        
    def conv_loss_frobenius(self):
        loss = 0.0
        for T in [self.T11,self.T12,self.T13,self.T14,self.T15]:
            loss += T.pow(2).sum()
        return loss
    

    def loss_distance(self,predictions):

        loss = 0.0
        Y = predictions - self.Z
        loss += Y.pow(2).mean()    
        
        return loss
        
    def loss_logdet(self):
        loss = 0.0
        T = torch.stack((self.Tp1,self.Tp2,self.Tp3,self.Tp4,self.Tp5),1)
        T = T.view(T.shape[0],-1)
        U, s, V = torch.svd(T)
        loss = -s.log().sum()
        return loss
        
        
    def loss_frobenius(self):
        loss = 0.0
        t_p = torch.stack((self.Tp1,self.Tp2,self.Tp3,self.Tp4,self.Tp5),1)
        loss = t_p.pow(2).sum()
        return loss


    def computeLoss(self,predictions,mu,eps,lam):
        loss1 = self.conv_loss_distance()
        loss2 = self.conv_loss_frobenius() * eps
        loss3 = self.conv_loss_logdet() * mu
        loss4 = self.loss_distance(predictions)
        loss5 = self.loss_frobenius() * eps
        loss6 = self.loss_logdet() * mu
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss

    
    def getTZ(self):
        return self.T.view(self.T.shape[0],-1), self.Z


# ## Training Related Functions 

# In[3]:


def train_model(epoch, model, optimizer, train_loader, batch_size, mu, eps, lam):
    model.train()
    t0 = time.time()
    correct = 0
    total = 0
    final_loss = 0
    i = 0 
    j = 0
    T_list = []
    Z_list = []
    for batch_idx, (X,future_prices) in enumerate(train_loader):
        data,future_prices = map(lambda x: Variable(x), [X,future_prices])
        data_size1 = data.shape[0]
        if j == 0: 
            prev_data = data
            prev_future_prices = future_prices
            j += 1
        if data.shape[0]<batch_size:
            diff = batch_size - data.shape[0]
            temp_data,temp_labels = prev_data[-diff:,:,:], prev_future_prices[-diff:]
            i = 1
            data, temp_future_prices = torch.cat((data,temp_data),0),torch.cat((future_prices,temp_future_prices),0)
            print('appended data')
            
        optimizer.zero_grad()
        
        output = model(data)

        final_output = output
        
        loss = model.computeLoss(final_output,mu,eps,lam)
        if epoch%plot_epoch_interval==0:
            train_loss.append(loss)
            epochs_list.append(epoch)
        final_loss += loss
        loss.backward()
        optimizer.step()
        model.X_step()
        model.Z_step()
        prev_data = data
        prev_future_prices = future_prices
    print('Epoch : {} , Training loss : {:.4f}\n'.format(epoch, final_loss.item()))
    return train_loss

 
    
def train_on_batch(lr,epochs,momentum,X_train,Y_train,X_test,Y_test,batch_size):
    print('seed:',seed)
    cuda = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(RegFinancialData(X_train,Y_train),batch_size=batch_size,shuffle=True) 
    test_loader = DataLoader(RegFinancialData(X_test,Y_test),batch_size=batch_size,shuffle=False) 
    
    
    mu = 0.01
    eps = 0.0001
    lam = 0 
    out_planes1 = out_pl1
    ksize1 = ks1
    maxpool1 = maxpl1
    mpl_ksize = mpl_ks#2
    model = Network(inputs_shape=(batch_size,1,window_size),out_planes1 = out_planes1, 
                    ksize1 = ksize1, maxpool1 = maxpool1, mpl_ksize=mpl_ksize)
#     for params in model.parameters():
#         print(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5, 
                                 amsgrad=False)

    for epoch in range(1, epochs + 1):
        train_loss = train_model(epoch, model, optimizer, train_loader, batch_size, mu, eps, lam)
    model.eval()


    
    S_train = Variable(torch.from_numpy(X_train).float(), requires_grad=False)
    S_test  = Variable(torch.from_numpy(X_test).float(), requires_grad=False)
    Z_train =  model(S_train).cpu().data.numpy()
    Z_test  = model(S_test).cpu().data.numpy()
    print('*'*100)
    print("Shape of Z_train: " + str(Z_train.shape))
    print("Shape of Z_test:  " + str(Z_test.shape))
    print('*'*100)

    return Z_train.transpose(),Z_test.transpose(),train_loss


# In[4]:


def plotGraph(title, train_loss):
    plt.figure()
    plt.plot(epochs_list,train_loss, label = 'Train Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss plot for ' + title)
    #plt.ticklabel_format(useOffset=False, style='plain')
    #plt.show()
    plt.savefig(base_path+'Results2/Loss_Plots/'+ title + '.eps',format='eps',dpi = 1000)
    plt.figure()
    plt.plot(epochs_list,train_loss, label = 'Train Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss plot for ' + title)
    #plt.ticklabel_format(useOffset=False, style='plain')
    #plt.show()
    plt.savefig(base_path+'Results2/Loss_Plots/'+ title + '.jpg')


# In[5]:


def checkClassImbal(Y_train):
    Ytrain_df= pd.DataFrame(Y_train,columns=[0])
    print(Ytrain_df.shape)
    print(Ytrain_df.columns)
    df = Ytrain_df.groupby(0).size()
    print(df)
    return df


# # Main

# In[6]:


window_size = 5#15#10#5#15#10#5
fileName = 'phd_research_data.csv'
data_df = getData(fileName)
if fileName == 'phd_research_data.csv':
    data_df.drop(['Unnamed: 0'],inplace=True,axis=1)
data_df,labels_df = labelData(data_df.copy())
data = np.asarray(data_df)


# In[7]:


stocks_list = getStocksList(data_df)


# In[8]:


start = 0#25#0#25#0
end = 150#25#150#25#150#1#150#25#2#25
seed_range = 10


# In[9]:


train_loss = []
train_accuracies = []
epochs_list = []
learning_rates = []
epoch_interval = 10
plot_epoch_interval = 5
test_accuracies = []


# In[10]:


lr = 0.001#0.0005
momentum = 0.9
epochs = 100#150#100
test_size = 0.2
features_list = ['CLOSE','OPEN','HIGH','LOW','CONTRACTS']


# In[11]:


out_pl1 = 4
maxpl1 = False
ks1 = 5
mpl_ks = 2
custom_batch_size_flag = False
bs = 32
if custom_batch_size_flag == True:
    param_path = '_op1_' + str(out_pl1) +'_mp1_' + str(maxpl1) + '_ks1_' + str(ks1) + '_bs_' + str(bs) + '_new'
else:
    param_path = '_op1_' + str(out_pl1)+'_mp1_' + str(maxpl1) + '_ks1_' + str(ks1) + '_new'
print(param_path)


# In[12]:


t_0 = time.time()
tr_loss_dict = {}
for stock in stocks_list[start:end]:
    t0 = time.time()
    _,windowed_data,_, next_day_values = getWindowedDataReg(data_df,stock,window_size)
    feat_wise_data = getFeatWiseData(windowed_data,features_list)
    prev_day_values = getPrevDayFeatures(feat_wise_data)
    next_day_values = next_day_values[:,0]
    next_day_values = next_day_values[0:next_day_values.shape[0]-1]
    X_train,Y_train,X_test,Y_test = splitData(feat_wise_data,next_day_values,test_size=test_size)
    X_test = X_test[0:X_test.shape[0]-1]
    prev_day_values = prev_day_values[X_train.shape[0]:][:,0]
    prev_day_values = prev_day_values[0:prev_day_values.shape[0]-1]
    print('prev_day_values.shape:',prev_day_values.shape)
    print('X_test.shape:',X_test.shape)
    print('Y_test.shape:',Y_test.shape)
    print('next_day_values.shape:', next_day_values.shape)
    prev_val_path = base_path + 'data/Reg3/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_yprev_cp.npy'
    np.save(prev_val_path,prev_day_values)
    for sd in range(1,seed_range+1):
        t01 = time.time()
        seed = sd
        print('Training for stock :',stock)
        print('seed : ',seed)
        print('starting at time:',t0)
        print('*'*100)
        if custom_batch_size_flag:
            batch_size = bs
        else:
            batch_size = X_train.shape[0]
        Ztrain, Ztest,train_loss = train_on_batch(lr,epochs,momentum,X_train,Y_train,X_test,Y_test,batch_size)
        tr_loss_dict[stock] = {}
        tr_loss_dict[stock] = train_loss
        train_loss = []
        xtr_path = base_path + 'data/Reg3/TL_Train/' + stock + param_path +'_' + str(test_size) + '_tl_xtrain' + str(seed) + '.npy'
        ytr_path = base_path + 'data/Reg3/TL_Train/' + stock + param_path +  '_' + str(test_size) + '_tl_ytrain' + str(seed) + '.npy'
        xte_path = base_path + 'data/Reg3/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_xtest' + str(seed) + '.npy'
        yte_path = base_path + 'data/Reg3/TL_Test/' + stock + param_path +  '_' + str(test_size) + '_tl_ytest' + str(seed) + '.npy'
        np.save(xtr_path,Ztrain)
        np.save(ytr_path,Y_train)
        np.save(xte_path,Ztest)
        np.save(yte_path,Y_test)
        t11 = time.time()
        print('*'*100)
        #print('*'*100)
        print('\n')
        print('time taken for training one stock:',datetime.timedelta(seconds = t11-t01))
    t1 = time.time()
    print('time taken for one stock through all seeds',datetime.timedelta(seconds = t1-t0))
t_1 = time.time()
print('time taken for 125 stocks through all seeds : ',str(datetime.timedelta(seconds = t_1-t_0)))


# # External Regressor

# In[13]:


def ridge_regressor(Xtrain, Ytrain, Xtest, Ytest, alpha = 1.0, random_state = 1):
    clf = Ridge(alpha=alpha,random_state = random_state)
    clf.fit(Xtrain, Ytrain)
    y_pred = clf.predict(Xtest)
    #y_tr_pred = 
    mae = mean_absolute_error(Ytest, y_pred)
    mse = mean_squared_error(Ytest, y_pred)
    rmse = math.sqrt(mse)
    return y_pred, mae, mse,rmse



def clfRF(Ztrain,Y_train,Ztest,Y_test,n_clf=5,depth=1,rnd_state=11):
    clf_rf = RandomForestClassifier(n_estimators=n_clf, max_depth=depth,random_state=rnd_state)
    clf_rf.fit(Ztrain, Y_train)
    ytr_rf_pred = clf_rf.predict(Ztrain)
    yte_rf_pred = clf_rf.predict(Ztest)
    tr_rf_score = round(accuracy_score(Y_train, ytr_rf_pred)*100,3)
    te_rf_score = round(accuracy_score(Y_test, yte_rf_pred)*100,3)
    tr_scores = clf_rf.predict_proba(Ztrain)
    te_scores = clf_rf.predict_proba(Ztest)
    print("RF classification, train acc: {:2.2f}%".format(tr_rf_score))
    print("RF classification, test acc: {:2.2f}%".format(te_rf_score))
    return ytr_rf_pred, yte_rf_pred, tr_rf_score, te_rf_score, tr_scores, te_scores





