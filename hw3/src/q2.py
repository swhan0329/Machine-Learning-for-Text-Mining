from KNN_utils import *
import time

def q2_1(MovieID_dev,UserID_dev,movie_features_matrix):
    print("==================Question 2.2. User-user similarity==================")
    # q2.1
    print("uu_MD10")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev,  movie_features_matrix, 10, distance='dot',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_MD10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_MD100")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 100, distance='dot',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_MD100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_MD500")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 500, distance='dot',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_MD500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_MC10")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 10, distance='cos',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_MC10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_MC100")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 100, distance='cos',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_MC100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_MC500")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 500, distance='cos',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_MC500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_WC10")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev,  movie_features_matrix, 10, distance='cos',
                    type='user', method='weight_mean')
    print("time :", time.time() - start)
    f = open("uu_WC10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_WC100")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 100, distance='cos',
                    type='user', method='weight_mean')
    print("time :", time.time() - start)
    f = open("uu_WC100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_WC500")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev,  movie_features_matrix, 500, distance='cos',
                    type='user', method='weight_mean')
    print("time :", time.time() - start)
    f = open("uu_WC500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

def q2_2(MovieID_dev,UserID_dev,movie_features_matrix):
    print("==================Question 2.2. Movie-movie similarity==================")
    # q2.2
    print("mm_MD10")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 10, distance='dot',
                    type='movie', method='mean')
    print("time :", time.time() - start)
    f = open("mm_MD10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("mm_MD100")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev,movie_features_matrix, 100, distance='dot',
                    type='movie', method='mean')
    print("time :", time.time() - start)
    f = open("mm_MD100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("mm_MD500")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 500, distance='dot',
                    type='movie', method='mean')
    print("time :", time.time() - start)
    f = open("mm_MD500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("mm_MC10")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev,  movie_features_matrix, 10, distance='cos',
                    type='movie', method='mean')
    print("time :", time.time() - start)
    f = open("mm_MC10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("mm_MC100")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev,  movie_features_matrix, 100, distance='cos',
                    type='movie', method='mean')
    print("time :", time.time() - start)
    f = open("mm_MC100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("mm_MC500")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev,  movie_features_matrix, 500, distance='cos',
                    type='movie', method='mean')
    print("time :", time.time() - start)
    f = open("mm_MC500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("mm_WC10")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev,  movie_features_matrix, 10, distance='cos',
                    type='movie', method='weight_mean')
    print("time :", time.time() - start)
    f = open("mm_WC10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("mm_WC100")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 100, distance='cos',
                    type='movie', method='weight_mean')
    print("time :", time.time() - start)
    f = open("mm_WC100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("mm_WC500")
    start = time.time()
    result = knn_q2(MovieID_dev, UserID_dev, movie_features_matrix, 500, distance='cos',
                    type='movie', method='weight_mean')
    print("time :", time.time() - start)
    f = open("mm_WC500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

def q2_3(MovieID_dev,UserID_dev,movie_features_matrix):
    print("==================Question 2.3. PCC==================")
    # q2.3
    print("uu_pccMD10")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev, movie_features_matrix, 10, distance='dot',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_pccMD10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_pccMD100")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev, movie_features_matrix, 100, distance='dot',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_pccMD100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_pccMD500")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev,  movie_features_matrix, 500, distance='dot',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_pccMD500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_pccMC10")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev, movie_features_matrix, 10, distance='cos',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_pccMC10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_pccMC100")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev,  movie_features_matrix, 100, distance='cos',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_pccMC100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_pccMC500")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev, movie_features_matrix, 500, distance='cos',
                    type='user', method='mean')
    print("time :", time.time() - start)
    f = open("uu_pccMC500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_pccWC10")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev, movie_features_matrix, 10, distance='cos',
                    type='user', method='weight_mean')
    print("time :", time.time() - start)
    f = open("uu_pccWC10.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_pccWC100")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev, movie_features_matrix, 100, distance='cos',
                    type='user', method='weight_mean')
    print("time :", time.time() - start)
    f = open("uu_pccWC100.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()

    print("uu_pccWC500")
    start = time.time()
    result = knn_q2_3(MovieID_dev, UserID_dev, movie_features_matrix, 500, distance='cos',
                    type='user', method='weight_mean')
    print("time :", time.time() - start)
    f = open("uu_pccWC500.txt", 'w')
    f.write('\n'.join(map(str, result)))
    f.close()