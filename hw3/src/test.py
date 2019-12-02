from q2_4 import *

def test(MovieID_test, UserID_test, I, movie_features_csr_matrix,MovieID_max,UserID_max,num,option,test,eval):
    q2_4(MovieID_test, UserID_test, I, movie_features_csr_matrix.T,MovieID_max+1,UserID_max+1,num,option,test,eval)


def dev(MovieID_dev, UserID_dev, I, movie_features_csr_matrix,MovieID_max,UserID_max,num,option,test,eval):
    q2_4(MovieID_dev, UserID_dev, I, movie_features_csr_matrix.T,MovieID_max+1,UserID_max+1,num,option,test,eval)
