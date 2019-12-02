import pandas as pd
from test import *
from q1 import *
from q2 import *
from q2_4 import *

train_csv = "/data/train.csv"
dev_csv = "/data/dev.csv"
test_csv = "/data/test.csv"

# load the train csv dataset
data = open(train_csv).read()
MovieID,UserID,Rating, RatingDate = [], [], [], []
for i, line in enumerate(data.split("\n")):
    if not line:
        break
    content = line.split(',')
    MovieID.append(int(content[0]))
    UserID.append(int(content[1]))
    Rating.append(float(content[2]))
    RatingDate.append(content[3])

# create a dataframe using MovieID,UserID,Rating and RatingDate
trainDF = pd.DataFrame()
trainDF['MovieID'] = MovieID
trainDF['UserID'] = UserID
trainDF['Rating'] = Rating
trainDF['RatingDate'] = RatingDate


UserID_max = max(UserID)
MovieID_max = max(MovieID)

# load the dev csv dataset
data_dev = open(dev_csv).read()
MovieID_dev,UserID_dev = [], []
for i, line in enumerate(data_dev.split("\n")):
    if not line:
        break
    content_dev = line.split(',')
    MovieID_dev.append(int(content_dev[0]))
    UserID_dev.append(int(content_dev[1]))

# load the test csv dataset
data_test = open(test_csv).read()
MovieID_test,UserID_test = [], []
for i, line in enumerate(data_test.split("\n")):
    if not line:
        break
    content_test = line.split(',')
    MovieID_test.append(int(content_test[0]))
    UserID_test.append(int(content_test[1]))
'''
#q1
q1(MovieID, UserID, Rating)'''

# pivot ratings into movie features
movie_features_pivot = trainDF.pivot(
    index='MovieID',
    columns='UserID',
    values='Rating'
).fillna(0)

movie_features_csr_matrix = csr_matrix((trainDF['Rating'],(trainDF['MovieID'],trainDF['UserID'])),shape=(MovieID_max+1, UserID_max+1))

I = csr_matrix((np.ones((trainDF['Rating']).shape[0]),(trainDF['UserID'],trainDF['MovieID']))).todense()

# q2.4
print("==================Question 2.4. Matrix Factorization==================")
q2_4(MovieID_dev, UserID_dev, I, movie_features_csr_matrix.T,MovieID_max+1,UserID_max+1,0,0,0,0)

for i in range(len(Rating)):
    Rating[i] -= 3
trainDF['Rating'] = Rating

# pivot ratings into movie features
movie_features_pivot = trainDF.pivot(
    index='MovieID',
    columns='UserID',
    values='Rating'
).fillna(0)
movie_features_csr_matrix = csr_matrix((trainDF['Rating'],(trainDF['MovieID'],trainDF['UserID'])),shape=(MovieID_max+1, UserID_max+1))
movie_features_matrix = movie_features_csr_matrix.toarray()

MovieID_dev = np.asarray(MovieID_dev)
UserID_dev = np.asarray(UserID_dev)
MovieID_test = np.asarray(MovieID_test)
UserID_test = np.asarray(UserID_test)

#KNN
print("==================Question 1. Top 5 NNs==================")
K=5
print("user 4321 dot")
knn_q1(4321,movie_features_matrix, K, type='user', distance='dot')
print("user 4321 cos")
knn_q1(4321, movie_features_matrix, K, type='user', distance='cos')
print("movie 3 dot")
knn_q1(3,  movie_features_matrix, K, type='movie', distance='dot')
print("movie 3 cos")
knn_q1(3, movie_features_matrix, K, type='movie', distance='cos')


#q2.1
q2_1(MovieID_dev,UserID_dev,movie_features_matrix)

#q2.2
q2_2(MovieID_dev,UserID_dev,movie_features_matrix)

#q2.3
q2_3(MovieID_dev,UserID_dev,movie_features_matrix)

print("==================Test predictions==================\n")
test(MovieID_test, UserID_test, I, movie_features_csr_matrix,MovieID_max,UserID_max,5,1,1,0)
print("==================dev predixctions==================")
dev(MovieID_dev, UserID_dev, I, movie_features_csr_matrix,MovieID_max,UserID_max,5,1,0,1)

