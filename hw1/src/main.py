'''
@HW1 for Machine Learning for text mining, CMU

Release date: 2019.09.xx
Created by Seowoo Han
'''
from collections import Counter  # for check reduplication in list
import warnings  # for ignoring the SparseEfficiencyWarning
import os
import numpy as np
import math
import time
from scipy.sparse import csc_matrix
# for ignoring the SparseEfficiencyWarning
from scipy.sparse import SparseEfficiencyWarning
# for ignoring the SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

def GPR(GIPV, Zt, alpha):
    startGPR = time.time()
    #iterative update
    i = 0
    while True:
        i = i+1
        WS = 0
        GIPV_Before = GIPV  # temporally stores r^(k-1)
        GIPV = (1-alpha)*O.dot(GIPV_Before)+(1-alpha)*Zt+alpha*p0  # r^(k)

        error = GIPV-GIPV_Before
        if np.linalg.norm(error, ord=1) < 10**(-8):
            break

        for j in range(0, len(zero)):
            WS = WS + GIPV[zero[j]]
        WS = WS/(max_value+1)
        Zt = np.zeros((max_value+1, 1))  # Zero tranpose
        Zt[:] = WS
    
    
    ff = open("./GPR.txt", "w")
    for kk in range(0,max_value+1):
        ff.write(str(kk+1) + " " +str(GIPV[kk,0])+"\n")
    ff.close()   
    GPR_time = time.time() - startGPR

    
    # print("convergence i: ",i)
    # print("stationary vector is: ", GIPV)
    # print("sum: ", sum(GIPV))
    for m in range(0,3):
        stratGPR_re = time.time()
        MakeTreeEvalFileforGPR(Dir_Wholetext,GIPV,m)
        GPR_re_time = time.time() - stratGPR_re
        if m == 0:
            print("GPR-NS:", (Setting_time+GPR_time), "secs for PageRank,", GPR_re_time, "secs for retrieval" )
        elif m == 1:
            print("GPR-WS:", (Setting_time+GPR_time), "secs for PageRank,", GPR_re_time, "secs for retrieval" )
        else:
            print("GPR-CM:", (Setting_time+GPR_time), "secs for PageRank,", GPR_re_time, "secs for retrieval" )

def TSPR(Input_dir, alpha, beta, gamma):
    #notation of the used algorithm.(0: QTSPR, 1:PTSPR)
    startTSPR = time.time()
    if Input_dir == Dir_query_topic_distro:
        state = 0
    else:
        state = 1

    #p0 setting
    p0 = (1/float(max_value+1))*np.ones((max_value+1, 1))

    IPV = np.zeros((max_value+1, NumOfTopic))
    for i in range(0, NumOfTopic):
        IPV[:, :] = p0

    #file open and save the list
    doc, topic = [], []
    f2 = open(Dir_doc_topics, 'r')
    lines = f2.readlines()
    for line in lines:
        result = line.split()
        doc.append(int(result[0])-1)  # text index begins 1
        topic.append(int(result[1])-1)  # text index begins 1
    f2.close()

    topic = np.array(topic)
    doc = np.array(doc)

    #topic sort
    S_topic, S_doc = [], []
    S_topic_idx = np.argsort(topic)
    S_topic = topic[S_topic_idx]
    S_doc = doc[S_topic_idx]

    Topic_Value, Topic_num = [], []
    Topic_Value, Topic_num = np.unique(S_topic, return_counts=True)
    # print("Topic_num: ", Topic_num)

    #make p matrix
    p = np.zeros((max_value+1, NumOfTopic), dtype=float)
    j = 0
    for T in range(0, NumOfTopic):
        for i in range(0, Topic_num[T]):
            p[S_doc[j+i], T] = 1/float(Topic_num[T])
        j = j + Topic_num[T]
    # print("p: ",p)

    #r^k for NumOfTopic topics
    error = np.zeros((max_value+1, NumOfTopic))
    IPV_Before = np.zeros((max_value+1, NumOfTopic))
    Zt = Ori_Zt

    for j in range(0, NumOfTopic):
        i = 0
        while True:
            i = i+1
            WS = 0
            IPV_Before[:, j] = IPV[:, j]  # temporally stores r^(k-1)

            Zt = Zt.reshape(-1)
            p0 = p0.reshape(-1)
            IPV[:, j] = alpha*O.dot(IPV_Before[:, j]) + \
                alpha*Zt+beta*p[:, j]+gamma*p0  # r^(k)

            error[:, j] = IPV[:, j]-IPV_Before[:, j]
            if np.linalg.norm(error[:, j], ord=1) < 10**(-8):
                break

            for k in range(0, len(zero)):
                WS = WS + IPV[zero[k], j]
            WS = WS/(max_value+1)
            Zt = np.zeros((max_value+1, 1))  # Zero tranpose
            Zt[:] = WS

    #read file query-topic-distro.txt
    user, query, pr_l = [], [], []
    topic = np.array((NumOfUser, NumOfTopic))
    f3 = open(Input_dir, 'r')
    lines = f3.readlines()
    for line in lines:
        for i in range(NumOfTopic, 0, -1):
            line = line.replace("%d:" % i, "")

        result = line.split()
        user.append(int(result[0]))
        query.append(int(result[1]))
        for i in range(2, 14):
            pr_l.append(result[i])
    f3.close()

    pr = np.zeros((NumOfUser, NumOfTopic))
    for i in range(0, NumOfUser):
        for j in range(0, NumOfTopic):
            pr[i, j] = pr_l[i*NumOfTopic+j]
    
    
    temprq=[]
    if state == 0:
        temprq=Definerq(max_value, NumOfTopic, 2, 1, pr, IPV, state)
        QTSPR_time = time.time() - startTSPR
        ff = open("./QTSPR-U2Q1.txt", "w")
        for kk in range(0,max_value+1):
            ff.write(str(kk+1) + " " +str(temprq[kk,0])+"\n")
        ff.close()
        for i in range(0, 3):
            stratTSPR_re = time.time()
            MakeTreeEvalFileforTSPR(Dir_Wholetext, pr, IPV, state, i)
            QTSPR_re_time = time.time() - stratTSPR_re
            if i == 0:
                print("QTSPR-NS:", QTSPR_time, "secs for PageRank,", QTSPR_re_time, "secs for retrieval" )
            elif i == 1:
                print("QTSPR-WS:", QTSPR_time, "secs for PageRank,", QTSPR_re_time, "secs for retrieval" )
            else:
                print("QTSPR-CM:", QTSPR_time, "secs for PageRank,", QTSPR_re_time, "secs for retrieval" )

    else:
        temprq=Definerq(max_value, NumOfTopic, 2, 1, pr, IPV, state)
        PTSPR_time = time.time() - startTSPR
        ff = open("./PTSPR-U2Q1.txt", "w")
        for kk in range(0,max_value+1):
            ff.write(str(kk+1) + " " +str(temprq[kk,0])+"\n")
        ff.close()     
        for i in range(0, 3):
            stratTSPR_re = time.time()
            MakeTreeEvalFileforTSPR(Dir_Wholetext, pr, IPV, state, i)
            PTSPR_re_time = time.time() - stratTSPR_re
            if i == 0:
                print("PTSPR-NS:", PTSPR_time, "secs for PageRank,", PTSPR_re_time, "secs for retrieval" )
            elif i == 1:
                print("PTSPR-WS:", PTSPR_time, "secs for PageRank,", PTSPR_re_time, "secs for retrieval" )
            else:
                print("PTSPR-CM:", PTSPR_time, "secs for PageRank,", PTSPR_re_time, "secs for retrieval" )

def MakeWholeTxt():
    '''
    This function should be make all indir-lists files into one text file.
    Input: None
    Output: None
    '''
    fn = []  # file name list
    for file in os.listdir("./data/indri-lists"):
        if file.endswith(".txt"):
            fn.append(file)

    #file directory
    QID, Q0, DID, R, S, RID = [], [], [], [], [], []
    for i in range(0, len(fn)):
        default_dir = "./data/indri-lists/"
        dir = default_dir + str(fn[i])
        ID = fn[i].split('.')
        f = open(dir, 'r')
        lines = f.readlines()
        for line in lines:
            result = line.split()
            QID.append(ID[0])
            Q0.append(result[1])
            DID.append(result[2])
            R.append(result[3])
            S.append(result[4])
            RID.append(result[5])
        f.close()
    ff = open("./data/Wholetext.txt", "w")
    for j in range(0, len(QID)):
        ff.write(QID[j] + " " + Q0[j] + " " + DID[j] + " " +
                 R[j] + " " + " " + S[j] + " " + RID[j]+"\n")
    ff.close()
    print("Finish making Whole text files into one text file")

def Definerq(MaxValue, NumOfTopic, UserId, Query, pr, IPV, state):
    '''
    This function should be make Rq matrix for specific user, queray
    Input: MV(max_value), NumOfTopic, UserId, Query, pr, IPV, state(0 means QTSPR, 1 means PTSPR)
    Output: rq.
    '''
    #define rq
    rq = np.zeros((max_value+1, 1))
    temp = np.zeros((max_value+1, NumOfTopic))

    UID, Q = [], []
    if state == 0:
        f = open(Dir_query_topic_distro, 'r')
    else:
        f = open(Dir_user_topic_distro, 'r')

    lines = f.readlines()
    for line in lines:
        result = line.split()
        UID.append(result[0])
        Q.append(int(result[1])-1)  # text index begins 1
    f.close()

    for m in range(0, NumOfUser):
        if int(UID[m]) - UserId == 0 and int(Q[m]) - Query == -1:
            for i in range(0, NumOfTopic):
                temp[:, i] = pr[m, i]*IPV[:, i]
            for j in range(0, max_value+1):
                rq[j] = sum(temp[j, :]).reshape(-1)
            break
    return rq

def MakeTreeEvalFileforGPR(text_dir, IPV, method):
    '''
    This function should be make TreeEval file for GPR algorithm.
    Input: text_dir(Wholetext.txt), IPV(r vector), method(0: NS, 1:WS, 2:CM)
    '''

    #file open and save the list
    UID_Q, Index, v = [], [], []
    f4 = open(text_dir, 'r')
    lines = f4.readlines()
    for line in lines:
        result = line.split()
        UID_Q.append(result[0])
        Index.append(int(result[2]))  # text index begins 1
        v.append(float(result[4]))
    f4.close()

    init = UID_Q[0]
    UID_index = []
    count = 0
    for i in range(0, len(UID_Q)):
        if init == UID_Q[i]:
            count = count + 1
            if i == len(UID_Q)-1:
                UID_index.append(count)
        else:
            init = UID_Q[i]
            UID_index.append(count)
            count = 1

    UID, Q = [], []
    for i in range(0, len(UID_Q)):
        UID.append(int(UID_Q[i].split('-')[0]))
        Q.append(int(UID_Q[i].split('-')[1])-1)


    if  method == 0:
        ff = open("./GPR-NS.txt", "w")
    elif method == 1:
        ff = open("./GPR-WS.txt", "w")
    else:
        ff = open("./GPR-CM.txt", "w")
    
    m = 0
    c = 0
    while m < len(UID_Q):
        Temp = np.zeros((UID_index[c], 1))
        T_Index, T_v = [], []
        T_index = Index[m:m+UID_index[c]]
        T_v = v[m:m+UID_index[c]]
        for j in range(0, UID_index[c]):
            if method == 0:
                Temp[j,0] = IPV[T_index[j]-1,0]
            elif method == 1:
                Temp[j,0] = 0.9*(IPV[T_index[j]-1,0]) + 0.1*(T_v[j])
            else:
                Temp[j,0] = 0.5*math.log(IPV[T_index[j]-1,0]) + 0.5*(T_v[j]) 
        
        Temp = Temp.reshape(-1)
        Temp = np.array(Temp)
        T_index = np.array(T_index)

        #IPV sort
        S_T_IPV, S_T_Index = [], []
        Sorted_Index = np.argsort(Temp)[::-1]
        S_T_IPV = Temp[Sorted_Index]
        S_T_Index = T_index[Sorted_Index]

        for k in range(0, UID_index[c]):
            ff.write(str(UID[m]) + "-" + str(int(Q[m])+1) + " " + "Q0" + " " + str(S_T_Index[k]) + " " +
                                        str(k+1) + " " + str(S_T_IPV[k]) + " " + "run-1"+"\n")
        m = m + UID_index[c]
        c = c + 1

    if  method == 0:
        ff.close()
        print("Finish making the GPR_NS.txt")
    elif method == 1:
        ff.close()
        print("Finish making the GPR_WS.txt")
    else:
        ff.close()
        print("Finish making the GPR_CM.txt")

def MakeTreeEvalFileforTSPR(text_dir, pr, IPV, state, method):
    '''
    This function should be make TreeEval file about each PageRank algorithm.
    Input: text_dir(Wholetext.txt), pr(probabilty of 38 users), IPV(r vector), method(0: NS, 1:WS, 2:CM)
    '''

    #file open and save the list
    UID_Q, Index, v = [], [], []
    f4 = open(text_dir, 'r')
    lines = f4.readlines()
    for line in lines:
        result = line.split()
        UID_Q.append(result[0])
        Index.append(int(result[2]))  # text index begins 1
        v.append(float(result[4]))
    f4.close()

    init = UID_Q[0]
    UID_index = []
    count = 0
    for i in range(0, len(UID_Q)):
        if init == UID_Q[i]:
            count = count + 1
            if i == len(UID_Q)-1:
                UID_index.append(count)
        else:
            init = UID_Q[i]
            UID_index.append(count)
            count = 1

    UID, Q = [], []
    for i in range(0, len(UID_Q)):
        UID.append(int(UID_Q[i].split('-')[0]))
        Q.append(int(UID_Q[i].split('-')[1])-1)

    txt_UID, txt_Q = [], []
    if state == 0:
        f = open(Dir_query_topic_distro, 'r')
    else:
        f = open(Dir_user_topic_distro, 'r')

    lines = f.readlines()
    for line in lines:
        result = line.split()
        txt_UID.append(int(result[0]))
        txt_Q.append(int(result[1]))  # text index begins 1
    f.close()

    S_rq, S_Index = [], []
    m = 0
    c = 0
    if state == 0 and method == 0:
        ff = open("./QTSPR-NS.txt", "w")
    if state == 0 and method == 1:
        ff = open("./QTSPR-WS.txt", "w")
    if state == 0 and method == 2:
        ff = open("./QTSPR-CM.txt", "w")
    if state == 1 and method == 0:
        ff = open("./PTSPR-NS.txt", "w")
    if state == 1 and method == 1:
        ff = open("./PTSPR-WS.txt", "w")
    if state == 1 and method == 2:
        ff = open("./PTSPR-CM.txt", "w")

    while m < len(UID_Q):
        for i in range(0, NumOfUser):
            if UID[m] - txt_UID[i] == 0 and Q[m] - txt_Q[i] == -1:
                rq = Definerq(max_value, NumOfTopic,
                                txt_UID[i], txt_Q[i], pr, IPV, state)

                Temp = np.zeros((UID_index[c], 1))
                T_Index, T_v = [], []
                T_index = Index[m:m+UID_index[c]]
                T_v = v[m:m+UID_index[c]]
                for j in range(0, UID_index[c]):
                    if method == 0:
                        Temp[j, 0] = rq[T_index[j]-1,0]
                    elif method == 1:
                        Temp[j, 0] = 0.9*(rq[T_index[j]-1,0]) + 0.1*(T_v[j])
                    else:
                        Temp[j, 0] = 0.5*math.log(rq[T_index[j]-1,0]) + 0.5*(T_v[j])
                    
                Temp = Temp.reshape(-1)
                Temp = np.array(Temp)
                T_index = np.array(T_index)

                #rq sort
                S_T_rq, S_T_Index = [], []
                Sorted_Index = np.argsort(Temp)[::-1]
                S_T_rq = Temp[Sorted_Index]
                S_T_Index = T_index[Sorted_Index]
                
                for j in range(0, UID_index[c]):
                    ff.write(str(UID[m]) + "-" + str(int(Q[m])+1) + " " + "Q0" + " " + str(T_index[j]) + " " +
                                        str(j+1) + " " + str(S_T_rq[j]) + " " + "run-1"+"\n")
        m = m + UID_index[c]
        c = c + 1

    if state == 0 and method == 0:
        ff.close()
        print("Finish making the QTSPR_NS.txt")
    if state == 0 and method == 1:
        ff.close()
        print("Finish making the QTSPR_WS.txt")
    if state == 0 and method == 2:
        ff.close()
        print("Finish making the QTSPR_CM.txt")
    if state == 1 and method == 0:
        ff.close()
        print("Finish making the PTSPR_NS.txt")
    if state == 1 and method == 1:
        ff.close()
        print("Finish making the PTSPR_WS.txt")
    if state == 1 and method == 2:
        ff.close()
        print("Finish making the PTSPR_CM.txt")
    
if __name__ == '__main__':
    #directory setting
    Dir_doc_topics = "./data/doc_topics.txt"
    Dir_query_topic_distro = "./data/query-topic-distro.txt"
    Dir_transition = "./data/transition.txt"
    Dir_user_topic_distro = "./data/user-topic-distro.txt"
    Dir_Wholetext = "./data/Wholetext.txt"

    MakeWholeTxt()

    startSetting = time.time()
    #############################################initial seting####################################################
    #setting for the max number of topics
    Topic = []
    f = open(Dir_doc_topics, 'r')
    lines = f.readlines()
    for line in lines:
        result = line.split()
        Topic.append(int(result[1])-1)  # text index begins 1
    f.close()
    NumOfTopic = max(Topic)+1

    #setting for the number of User
    U = []
    f = open(Dir_query_topic_distro, 'r')
    lines = f.readlines()
    for line in lines:
        result = line.split()
        U.append(int(result[0])-1)  # text index begins 1
    f.close()
    NumOfUser = len(U)

    #setting for the max value between row and col
    T_row, T_col, data = [], [], []  # transition i, j, k factors
    f = open(Dir_transition, 'r')
    lines = f.readlines()
    for line in lines:
        result = line.split()
        T_row.append(int(result[0])-1)  # text index begins 1
        T_col.append(int(result[1])-1)  # text index begins 1
        data.append(float(result[2]))  # 1 means connect, 0 means unconnect
    f.close()

    T_row_max = max(T_row)
    T_col_max = max(T_col)
    max_value = max(T_row_max, T_col_max)

    T_row = np.array(T_row)
    T_col = np.array(T_col)
    data = np.array(data)

    O = csc_matrix((data, (T_row, T_col)), shape=(
        max_value+1, max_value+1), dtype=float)

    ni = O.sum(axis=1)[:]
    ni = np.array(ni)
    ni = ni[:, 0]

    #T_row sort
    T_Sorted_row_idx = np.argsort(T_row)
    T_Sorted_row = T_row[T_Sorted_row_idx]
    T_Sorted_col = T_col[T_Sorted_row_idx]
    T_Sorted_data = data[T_Sorted_row_idx]

    #data normalization for data value is one
    v = 0
    for i in range(0, len(T_row)):
        if T_Sorted_row[i] != v:
            v = T_Sorted_row[i]
        if T_Sorted_row[i] == v:
            if ni[v] > 0:
                T_Sorted_data[i] = T_Sorted_data[i]/ni[v]

    O = csc_matrix((T_Sorted_data, (T_Sorted_row, T_Sorted_col)),
                   shape=(max_value+1, max_value+1), dtype=float)
    O = O.transpose()
    zero = []
    for i in range(0, max_value+1):
        if ni[i] == 0:
            zero.append(i)

    #p0 setting(p0 == IPV)
    p0 = (1/float(max_value+1))*np.ones((max_value+1, 1))
    GIPV = p0
    #B.tranpose()*r
    Ori_Zt = np.zeros((max_value+1, 1))  # Zero tranpose
    WS = 0
    for i in range(0, len(zero)):
        WS = WS + GIPV[zero[i]]
    WS = WS/float(max_value+1)
    Ori_Zt[:] = WS
    ###############################################################################################################
    Setting_time = time.time()-startSetting

    GPR(GIPV, Ori_Zt, 0.2)
    TSPR(Dir_query_topic_distro, 0.8, 0.1, 0.1)
    TSPR(Dir_user_topic_distro,0.8, 0.1, 0.1)