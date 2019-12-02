import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

def knn_q1(input, feature_matrix, K, type='',distance=''):
    if type is 'user':
        matrix = feature_matrix.T
    elif type is 'movie':
        matrix = feature_matrix

    sim = np.zeros((np.size(matrix, axis=0)))

    if distance is 'dot':
       sim = matrix.dot(matrix[input, :].T)
    elif distance is 'cos':
        sim = cosine_similarity(matrix,matrix[input, :].reshape(-1, 1).T)
        sim = sim.flatten()

    sim_index = np.argsort(sim)[::-1]
    for i in range(1, K+1):
        print(sim_index[i],sep =' ')

def knn_q2(movie_input, user_input, feature_matrix, K, type='',method='' ,distance=''):
    if type is 'user':
        matrix = feature_matrix.T
        input = user_input
        mul = movie_input
        #uni = uni_UserID
        #uni_mul =uni_MovieID

    elif type is 'movie':
        matrix = feature_matrix
        input = movie_input
        mul = user_input
        #uni = uni_MovieID
        #uni_mul = uni_UserID

    sort_idx = np.argsort(input)
    #input = input[sort_idx]
    #mul = mul[sort_idx]
    input_sort = input[sort_idx]
    mul_sort = mul[sort_idx]

    sim = np.zeros((np.size(matrix, axis=0)))
    KNN_query = np.zeros((np.size(matrix, axis=0)))
    new_matrix = np.zeros((K,(np.size(matrix, axis=1))))
    #result = []
    result = [0]*30000
    temp = 0
    c = 0
    idx = 0
    for In in input_sort:
        #if c % 1000 == 0:
            #print(c)

        if (temp != In and c is not 0) or (c is 0 and idx is 0):
            #index = np.where(uni == In)[0][0]
            index = In
            #index = uni.index(In)
            if distance is 'dot':
                sim = matrix.dot(matrix[index, :].T)
            elif distance is 'cos':
                sim = cosine_similarity(matrix, matrix[input, :].reshape(-1, 1).T)
                #sim = np.inner(matrix, matrix[index, :])/ ((norm(matrix))*norm(matrix[index, :])+1e-8)

            sim_index = np.argsort(sim)[::-1]
            new_matrix = matrix[sim_index[1:K + 1], :]
            new_sim = sim[sim_index[1:K + 1]]

            if method is 'weight_mean':
                weight = new_sim / (abs(new_sim).sum() + 1e-8)
                weight = weight.reshape(-1,1)
                new_matrix = np.multiply(new_matrix,weight)

            KNN_query = np.sum(new_matrix, axis=0) /K

        #print(KNN_query.shape)
            '''if mul_sort[c] not in uni_mul:
                result[sort_idx[idx]] = 3
                #rfesult.append(3)
                c += 1
                idx += 1
            else:'''
        result[sort_idx[idx]] = KNN_query[mul_sort[c]] + 3

        c += 1
        idx += 1
        temp = In
    return result

def knn_q2_3(movie_input, user_input, feature_matrix, K, type='',method='' ,distance=''):
    #print(feature_matrix)
    if type is 'user':
        matrix = feature_matrix.T
        input = user_input
        mul = movie_input


    elif type is 'movie':
        matrix = feature_matrix
        input = movie_input
        mul = user_input
    matrix_cal = matrix
    mul_matrix_mean = (np.sum(matrix, axis=0) / (np.count_nonzero(matrix, axis=0)+1e-10)).reshape(-1, 1)
    #print(np.count_nonzero(matrix, axis=1))
    matrix_mean = (np.sum(matrix,axis=1)/ (np.size(matrix, axis=1))).reshape(-1,1)
    #print(matrix_mean)
    matrix = matrix - matrix_mean
    matrix_norm = norm(matrix,axis=1).reshape(-1,1)

    if matrix_norm is 0:
        matrix /=  1e-8
    if matrix_norm is 0:
        matrix /= matrix_norm

    sort_idx = np.argsort(input)
    input_sort = input[sort_idx]
    mul_sort = mul[sort_idx]

    sim = np.zeros((np.size(matrix, axis=0)))
    KNN_query = np.zeros((np.size(matrix, axis=0)))
    new_matrix = np.zeros((K,(np.size(matrix, axis=1))))
    In_mean = 0
    In_norm = 0

    result = [0]*30000
    temp = 0
    c = 0
    idx = 0
    for In in input_sort:
        #if c % 1000 == 0:
            #print(c)

        if (temp != In and c is not 0) or (c is 0 and idx is 0):
            #index = np.where(uni == In)[0][0]
            index = In
            In_row = matrix[index, :]
            In_row_cal = matrix_cal[index, :]
            In_mean = (np.sum(In_row_cal, axis=0)/(np.size(In_row_cal, axis=0)))
            In_norm = norm(In_row_cal)


            #index = uni.index(In)
            if distance is 'dot':
                sim = matrix.dot(In_row.T)
            elif distance is 'cos':
                sim = cosine_similarity(matrix, matrix[input, :].reshape(-1, 1).T)
                #sim = np.inner(matrix, In_row)/ ((norm(matrix))*norm(In_row)+1e-8)

            sim_index = np.argsort(sim)[::-1]
            new_matrix = matrix[sim_index[1:K + 1], :]
            new_sim = sim[sim_index[1:K + 1]]

            if method is 'weight_mean':
                if abs(new_sim).sum() == 0:
                    weight = new_sim / (1e-10)
                else:
                    weight = new_sim / (abs(new_sim).sum())
                weight = weight.reshape(-1,1)
                new_matrix = np.multiply(new_matrix,weight)

            KNN_query = np.sum(new_matrix, axis=0).reshape(-1,1) /K

            KNN_query_mean = (np.sum(KNN_query, axis=0) / (np.size(In_row, axis=0)))[0]
            KNN_query -= KNN_query_mean
            if norm(KNN_query,axis=0) == 0:
                KNN_query /= 1e-10
            else:
                KNN_query /= (norm(KNN_query, axis=0).reshape(-1, 1))
        #print(In_norm, In_mean)
        result[sort_idx[idx]] = (0.6*((KNN_query[mul_sort[c]]) * In_norm + In_mean) +0.4*mul_matrix_mean[mul_sort[c]]+3)[0]
        #print((KNN_query[mul_sort[c]])[0])
        #result[sort_idx[idx]] = (KNN_query[mul_sort[c]])[0]
        c += 1
        idx += 1
        temp = In
    return result





