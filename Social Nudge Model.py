import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations 
import random
import pandas as pd
import numpy as np

def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup

def build_V(node):
    V = set([v for v in range(node)])
    return V

def random_E(V, permu, prob_f = 0.7):
    followed_lst = []
    for per in permu:
        per_prob_f = random.random()
        if per_prob_f < prob_f:
            followed_lst.append(per)
    return followed_lst

def get_input_E(permu, relation_list):
    E_list = []
    for i in range(len(permu)):
        if relation_list[i] == 1:
            E_list.append(permu[i])
    return E_list

def random_E_V_Network(V, permu, E, node):
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

def input_E_V_Network(V, permu, E, node, pe_list):
    dic = {}
    for i in range(len(permu)):
        nudge = permu[i]
        pe_nudge = pe_list[i]
        if nudge in E:
            dic[pe_nudge] = (dic.get(pe_nudge, set()))
            dic[pe_nudge].add(nudge)
    g = nx.DiGraph()
    g.add_nodes_from(V)        
    for key in dic.keys():
        g.add_edges_from(set(dic[key]), weight = key)        
    return g

def get_dic_labels(G, permu, E_pop):
    labels = nx.get_edge_attributes(G,'weight')
    dic = {}
    for per in permu:
        if per in E_pop:
            dic[str(per)] = [ per[0], per[1], True, labels[per]] 
        else:
            dic[str(per)] = [per[0], per[1], False, 0] 
    return dic

def get_random_D(df, permu, E_pop):
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
                        d_l_i.append(random.uniform(0.1, 1)) 
            d_l_e.append(d_l_i)
    return np.matrix(np.array(d_l_e).T)

def get_input_D(df, permu, E):
    d_l_e = []
    for i in range(len(permu)):
        edge = permu[i]
        if edge not in E:
            d_l_i = [0]*len(permu)
            d_l_e.append(d_l_i)
        else:
            d_l_i = []
            for j in range(len(permu)):
                l = permu[j]
                if l not in E:
                    d_l_i.append(0)
                else:
                    if edge[0] != l[1]:
                        d_l_i.append(0)
                    else:
                        d_l_i.append(float(input(
            "Enter the d_le value where e is " + str(edge) + " and l is " + str(l) + "\n")))
            d_l_e.append(d_l_i)
    D = np.matrix(np.array(d_l_e).T)
    return D

def get_random_DF(dic):
    df = pd.DataFrame(dic, index = ['e_o：e的出发点','e_d：e的结束点','e是否在E内', 'p_e：e这条边的影响力']).T
    df['Mu_e：e这条边的固定nudge数量'] = 0
    for i in range(df.shape[0]):
        df.iloc[i, -1] = random.uniform(0.1, 1)    
    df['Eta_e'] = df.loc[:,'p_e：e这条边的影响力'] / (1-alpha_p)
    return df

def get_input_DF(dic, mu_list):
    df = pd.DataFrame(dic, index = ['e_o：e的出发点','e_d：e的结束点','e是否在E内', 'p_e：e这条边的影响力']).T
    df['Mu_e：e这条边的固定nudge数量'] = mu_list  
    df['Eta_e'] = df.loc[:,'p_e：e这条边的影响力'] / (1-alpha_p)
    return df

def check_normD(D, shrink = 0.5):
    while np.linalg.norm((1/(1-alpha_d))*D, 1) >= 1:
        D = D*shrink
    return D

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

# real value calculation
def get_XY_star(I, Mu, Eta, D):
    y_star = np.dot(np.linalg.inv(I - ((1/(1-alpha_d))*D)), Mu)
    x_star = np.dot(Eta.T, y_star)
    return  x_star[(0,0)] 

# estimation
def get_XY_k(I, D_norm, Mu, k_max):
    list_x_k = []
    #print('this is D', D)  
    for k in range(1, k_max):
        sum_list = []
        D_k = D_norm
        if k == 1:
            y_k = np.dot(I +((D_k) * (1/(1-alpha_d))) , Mu)
            x_k = np.dot(Eta.T, y_k)
            list_x_k.append(x_k[(0,0)])
        else:
            for i in range(k):
                sum_list.append((D_k) * (1/((1-alpha_d)**(i+1))))
                D_k = D_k * D_norm
            y_k = np.dot( (I + sum(x for x in sum_list)) , Mu)
            x_k = np.dot(Eta.T, y_k)
            list_x_k.append(x_k[0,0])
    return list_x_k

def get_DownSampleX_list(V, E, DF, alpha_d, alpha_p, sample_size, times, D, permu):
    list_x_DownSample = []
    for i in range(times):
        x_DownSample = DownSample_Approximation(V, E, DF, alpha_d, alpha_p, sample_size, D, permu)
        list_x_DownSample.append(x_DownSample) 
    return list_x_DownSample

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
    ax.set_xlabel("k") 
    ax.set_ylabel('x_i')
    plt.show()    

def plot2(list_x_DownSample, list_x_k):        
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False  
    plt.rcParams["figure.figsize"] = (12,6)
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
    plt.title('DownSample value distribution')   

def random_SN_model(node, alpha_d, alpha_p, shrink_size, prob_f):
    V = build_V(node)
    permu = list(permutations(V,2))
    E = random_E(V, permu, prob_f)
    G = random_E_V_Network(V, permu, E, node)
    dic = get_dic_labels(G, permu, E)
    DF = get_random_DF(dic)       
    D =get_random_D(DF, permu, E)     
    D_norm = check_normD(D,shrink_size)    
    return V, permu,E, G, DF, D_norm

def get_relation_list(permu):
    #Inputs follow the same order and length as permutation and seperate with comma. Each value should be integer and 1 present "True" while 0 present "False"
    #Example: 1,1,0,1,0,0
    while True:
        try:            
            relation_list = [int(x) for x in input("Enter whether the nudge is in the set of following relationship(lenth is " + str(len(permu)) + "): \n").split(',')]
            break
        except ValueError:
            print("That was no valid value. Try again...")
    return relation_list

def get_pe_list(permu):
    #Inputs follow the same order and length as permutation and seperate with comma. Each value should be a positive true value.
    #Example: 1,2.5,4,5.26,0,0.454  
    while True:
        try:            
            pe_list = [float(x) for x in input("Enter the p_e value for each nudge(lenth is " + str(len(permu)) + "):\n").split(',')]
            break
        except ValueError:
            print("That was no valid value. Try again...")
    return pe_list        

def get_mu_list(permu):
    #Inputs follow the same order and length as permutation and seperate with comma. Each value should be a positive true value.
    #Example: 1, 1.5, 2, 245, 3.567, 0
    while True:
        try:            
            mu_list = [float(x) for x in input("Enter the mu value for each nudge(lenth is " + str(len(permu)) + "):\n").split(',')]
            break
        except ValueError:
            print("That was no valid value. Try again...")
    return mu_list     

def input_SN_model(node, alpha_d, alpha_p, shrink_size):
    V = build_V(node)
    permu = list(permutations(V, 2))
    relation_list = get_relation_list(permu)
    pe_list = get_pe_list(permu)
    mu_list = get_mu_list(permu)
    n = True
    while n:
        if len(relation_list) != len(permu):
            print('The list length for whether the nudge is in the set of following is wrong. Please re-enter the value.')
            relation_list = get_relation_list(permu)
        else:
            if len(pe_list) != len(permu):
                print('The pe list length is wrong. Please re-enter the value.')
                pe_list = get_pe_list(permu)
            else:
                if len(mu_list) != len(permu):
                    print('The mu list length is wrong. Please re-enter the value.')
                    mu_list = get_mu_list(permu)
                else:
                    n = False
    
    E = get_input_E(permu, relation_list)
    G = input_E_V_Network(V, permu, E, node, pe_list)
    dic = get_dic_labels(G, permu, E)
    DF = get_input_DF(dic, mu_list)
    D = get_input_D(DF, permu, E)
    D_norm = check_normD(D,shrink_size) 
    return V, permu, E, G, DF, D_norm

def show_network(G):   
    pos = nx.spring_layout(G,weight = 'weight')
    labels = nx.get_edge_attributes(G,'weight')
    layout = nx.spring_layout(G)

    nx.draw_networkx_nodes(G,pos, node_size=10,node_color='r')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("Social Network Diagram")
    plt.show()
    
if __name__=='__main__':
    node = int(input("Enter the node number: \n"))
    alpha_d = float(input("Enter the alpha_d: \n"))
    alpha_p = float(input("Enter the alpha_p: \n"))
    shrink_size = float(input("Enter the value for matrix to multiply if matrix norm is greater than 1: \n"))
    #shrink_size should be a value between (0,1)
    k_max = int(input("Enter the times for k_max: \n"))
    sample_size = int(input("Enter the sample size for each downsample estimation: \n"))
    times = int(input("Enter times for downsample estimation: \n"))
    
    check = input("Enter the module you want to use(Random or Self-input): \n")
    if check == 'Random':
        prob_f = float(input("Enther the probability for each nudge is in the following relationship: \n"))
        V, permu, E, G, DF, D_norm = random_SN_model(node, alpha_d, alpha_p, shrink_size, prob_f)
    if check == "Self-input":
         V, permu, E, G, DF, D_norm = input_SN_model(node, alpha_d, alpha_p, shrink_size)
       
    Eta = np.array(DF['Eta_e']).reshape(len(permu),1)
    Mu = np.array(DF['Mu_e：e这条边的固定nudge数量']).reshape(len(permu),1)
    I = np.identity((DF.shape[0]), dtype=int)
    
    x_star = get_XY_star(I, Mu, Eta, D_norm)    
    list_x_k = get_XY_k(I, D_norm, Mu, k_max)   
    DownSampleX = get_DownSampleX_list(V, E, DF, alpha_d, alpha_p, sample_size, times, D_norm, permu)  
                    
    plot1(x_star, list_x_k, node)
    plot2(DownSampleX, list_x_k)
   # np.mean(list_DownSample)
    #show_network(G)
