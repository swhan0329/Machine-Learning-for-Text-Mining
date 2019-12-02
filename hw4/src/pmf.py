import numpy as np

class PMF(object):
    """PMF

    :param object:
    """

    def __init__(self, num_factors, num_users, num_movies):
        """__init__

        :param num_factors:
        :param num_users:
        :param num_movies:
        """
        # note that you should not modify this function
        np.random.seed(11)
        self.U = np.random.normal(size=(num_factors, num_users))
        self.V = np.random.normal(size=(num_factors, num_movies))
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_factors = num_factors
    def predict(self, user, movie):
        """predict

        :param user:
        :param movie:
        """
        # note that you should not modify this function
        return (self.U[:, user] * self.V[:, movie]).sum()

    def train(self, users, movies, ratings, alpha, lambda_u, lambda_v,
              batch_size, num_iterations):
        """train

        :param users: np.array of shape [N], type = np.int64
        :param movies: np.array of shape [N], type = np.int64
        :param ratings: np.array of shape [N], type = np.float32
        :param alpha: learning rate
        :param lambda_u:
        :param lambda_v:
        :param batch_size:
        :param num_iterations: how many SGD iterations to run
        """
        # modify this function to implement mini-batch SGD
        # for the i-th training instance,
        # user `users[i]` rates the movie `movies[i]`
        # with a rating `ratings[i]`.

        total_training_cases = users.shape[0]

        for i in range(num_iterations):
            start_idx = (i * batch_size) % total_training_cases
            users_batch = users[start_idx:start_idx + batch_size]
            movies_batch = movies[start_idx:start_idx + batch_size]
            ratings_batch = ratings[start_idx:start_idx + batch_size]
            curr_size = ratings_batch.shape[0]

            # TODO: implement your SGD here!!
            self.calLossSGD(users_batch, movies_batch, ratings_batch, alpha,lambda_u, lambda_v)

            '''total_predict = np.dot(self.U.T, self.V)[users,movies]
            loss = 0.5 * (np.sum((ratings - total_predict) ** 2)) \
                   + (lambda_u / 2) * (np.sum(self.U ** 2)) \
                   + (lambda_v / 2) * (np.sum(self.V ** 2))

            #print("loss:", loss)'''

    def calLossSGD(self, users, movies, ratings, alpha,lambda_u, lambda_v):
        total_predict = np.dot(self.U.T, self.V)[users,movies]

        set_user = set(users)
        for i in set_user:
            user_idx = np.where(users==i)[0]
            movie_idx = movies[user_idx]
            dU = np.matmul(self.V[:,movie_idx],total_predict[user_idx]-ratings[user_idx]) + lambda_u * self.U[:, i]
            self.U[:, i] -= alpha * dU

        set_movie = set(movies)
        for k in set_movie:
            movie_idx = np.where(movies==k)[0]
            user_idx = users[movie_idx]
            dV = np.matmul(self.U[:, user_idx], total_predict[movie_idx]-ratings[movie_idx])+lambda_v * self.V[:, k]
            self.V[:, k] -= alpha * dV





