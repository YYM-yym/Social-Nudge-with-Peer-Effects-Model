# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from itertools import permutations 
import random
import pandas as pd
import numpy as np
#from cvxopt import *

def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup
def V(node):
    V = set([v for v in range(node)])
    return V
def E(V, permu, prob_f = 0.7):
    followed_lst = []
    for per in permu:
        per_prob_f = random.random()
        if per_prob_f < prob_f:
            followed_lst.append(per)
    return followed_lst
def E_V_Network(V, permu, E, node):
    dic = {}
    for per in permu: 
        edge = False
        p_e = 0                
        if per in E:
            edge = True   
            p_e += random.uniform(0.1, 1)
            if Reverse(per) in E:
                p_e += random.uniform(0.1, 1)         
        if edge:
            dic[p_e] = (dic.get(p_e,set()))
            dic[p_e].add(per)
        
    g = nx.DiGraph()
    g.add_nodes_from(V)
    for key in dic.keys():
        g.add_edges_from(set(dic[key]), weight = key)
    return g

def show_network(G):   
    pos = nx.spring_layout(G,weight = 'weight')
    labels = nx.get_edge_attributes(G,'weight')
    layout = nx.spring_layout(G)

    nx.draw_networkx_nodes(G,pos, node_size=10,node_color='r')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("Random Graph Generation Example")
    plt.show()
    
def get_dic_labels(G, permu, E_pop):
    labels = nx.get_edge_attributes(G,'weight')
    dic = {}
    for per in permu:
        if per in E_pop:
            dic[str(per)] = [ per[0], per[1], True, labels[per]] 
        else:
            dic[str(per)] = [per[0], per[1], False, 0] 
    return dic

def get_D(df, permu, E_pop):
    dimention = len(permu)
    d_l_e =[]
    for i in range(dimention):
        e = permu[i]
        if e not in E_pop:
            d_l_i = [0]*dimention
            d_l_e.append(d_l_i)
        else:
            d_l_i = []
            for j in range(dimention):
                l = permu[j]
                if l not in E_pop:
                    d_l_i.append(0)
                else:
                    if e[0] != l[1]:
                        d_l_i.append(0)
                    else:
                        d_l_i.append(random.uniform(0.1, 1)) #如何界定d_l_e?
            d_l_e.append(d_l_i)
    return np.array(d_l_e).T

### downsample estimation

def get_DownSample_V(V_pop, Size = 5):
    return random.sample(list(V_pop), Size)

def get_DownSample_E(Down_V, E_pop):
    Down_E = []   
    for edge in E_pop:
        if edge[-1] in Down_V:
            Down_E.append(edge)
    return Down_E

def get_DownSample_L(Down_V, E_pop):
    Down_L = []
    for edge in E_pop:
        if edge[0] in Down_V:
            Down_L.append(edge)
    return Down_L

def DownSample_Approximation(V_pop, E_pop, df_D, alpha_d, alpha_p, sample_size, D_norm, permu):
    Down_V = get_DownSample_V(V_pop,sample_size)
    Down_E = get_DownSample_E(Down_V, E_pop)
    Down_L = get_DownSample_L(Down_V, E_pop)
    
    w_0, w_1 = 0,0
   # i = 9
    for i in Down_V:
        for e in Down_E:
            if e[-1] == i:
                w_0 += (df_D.loc[str(e), 'Mu_e：e这条边的固定nudge数量'] * df_D.loc[str(e), 'p_e：e这条边的影响力']) / (1-alpha_p)
                for l in Down_L:
                    if l[0] == i:
                        w_1 += (
                        df_D.loc[str(e), 'Mu_e：e这条边的固定nudge数量'] * df_D.loc[str(l), 'p_e：e这条边的影响力'] * 
                        (D_norm[ permu.index(e) , permu.index(l)])
                        ) / ( (1-alpha_d) * (1-alpha_p))


    return (len(V_pop) / len(Down_V)) * (w_0 + w_1)        

def get_df_D(dic):
    df = pd.DataFrame(dic, index = ['e_o：e的出发点','e_d：e的结束点','e是否在E内', 'p_e：e这条边的影响力']).T
    df['Mu_e：e这条边的固定nudge数量'] = 0
    for i in range(df.shape[0]):
        df.iloc[i, -1] = random.uniform(0.1, 1)    
    df['Eta_e'] = df.loc[:,'p_e：e这条边的影响力'] / (1-alpha_p)
    return df

def check_normD(D, shrink = 0.5):
    while np.linalg.norm((1/(1-alpha_d))*D, 1) >= 1:
        D = D*shrink
    return D
    #np.linalg.norm(D,1) > (1-alpha_d):
  #      df.loc[:,'p_e：e这条边的影响力'] = df.loc[:, 'p_e：e这条边的影响力']/2
        #df.loc[:,'Mu_e：e这条边的固定nudge数量'] = df.loc[:, 'Mu_e：e这条边的固定nudge数量']/2
   #     D = matrix(get_D(df, permu, E_pop))  

def get_XY_star(I, Mu, Eta, D):
    y_star = np.dot(np.linalg.inv(I - ((1/(1-alpha_d))*D)), Mu)
    x_star = np.dot(Eta.T, y_star)

    return  x_star    

def get_XY_k(I, D_norm, Mu, k_max):
    list_x_k = []
    #print('this is D', D)  
    for k in range(1, k_max):
        sum_list = []
        D_k = D_norm
        if k == 1:
            y_k = np.dot(I +((D_k) * (1/(1-alpha_d))) , Mu)
            x_k = np.dot(Eta.T, y_k)
            list_x_k.append(x_k)
        else:
            for i in range(k):
                sum_list.append((D_k) * (1/((1-alpha_d)**(i+1))))
                D_k = D_k * D_norm
            y_k = np.dot( (I + sum(x for x in sum_list)) , Mu)
            x_k = np.dot(Eta.T, y_k)
            list_x_k.append(x_k)
    return list_x_k

def plot1(x_star, list_x_k, node):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    y1 = list_x_k
    x =  [x+1 for x in range(len(list_x_k))]    
    y2 = [x_star]*(len(list_x_k))
    #y3 = [x_DownSample]*39
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)    
    ax.plot(x,y1, color = 'red', label = "Approximation Value")
    ax.plot(x,y2, color = 'black', label = 'True Value')
    #ax.plot(x,y3, color = 'blue', label = 'DownSample Approximation Value')
    
    fig.legend(loc="upper right")
    
    ax.set_title("Approximation Value VS True Value: N = "+ str(node)) 
    ax.set_xlabel("K的取值") 
    ax.set_ylabel('x_i')
    plt.show()    

def plot2(V_pop, E_pop, df_D, alpha_d, alpha_p, sample_size, times, D_norm, permu, list_x_k):
    list_x_DownSample = []
    for i in range(times):
        x_DownSample = DownSample_Approximation(V_pop, E_pop, df_D, alpha_d, alpha_p, sample_size, D_norm, permu)
        list_x_DownSample.append(x_DownSample)
        
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False  
    plt.rcParams["figure.figsize"] = (12,6)
    x = list_x_DownSample
    n, bins, patches = plt.hist(x=list_x_DownSample, 
        bins = 10,
        color = 'red', 
        #density = True, stacked = True
        #alpha = 1, rwidth = 0.5
        )
    plt.axvline(list_x_k[0], color='k', linestyle='dashed', linewidth=2)
    min_ylim, max_ylim = plt.ylim()
    plt.text(list_x_k[0]*1.01, max_ylim*0.9, 'x_1: {:.2f}'.format(list_x_k[0]))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('DownSample Value')
    plt.ylabel('Frequency')
    plt.title('DownSample 预测分布')
    
    return list_x_DownSample


if __name__=='__main__':
    node = 30
    alpha_d, alpha_p = 0.1, 0.1
    shrink_size = 0.1
    
    V_pop = V(node)
    permu = list(permutations(V_pop,2))
    E_pop = E(V_pop, permu,prob_f = 0.7)
    G = E_V_Network(V_pop, permu, E_pop, node)
    #show_network(G)
    
    dic = get_dic_labels(G, permu, E_pop)
    df_D = get_df_D(dic)   
    Eta = np.array(df_D['Eta_e'])
    Mu = np.array(df_D['Mu_e：e这条边的固定nudge数量'])
    I = np.identity(len(permu), dtype=int)
     
    D = matrix(get_D(df_D, permu, E_pop))        
    D_norm = check_normD(D,shrink_size)
   # print(D_norm)    

    x_star = get_XY_star(I, Mu, Eta, D_norm)
    k_max = 15
    list_x_k = get_XY_k(I, D_norm, Mu, k_max)                                
    plot1(x_star, list_x_k, node)

    sample_size, times = 10, 100
    list_DownSample = plot2(V_pop, E_pop, df_D, alpha_d, alpha_p, sample_size, times, D_norm, permu, list_x_k)

    np.mean(list_DownSample)
    
    
    



