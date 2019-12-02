import numpy as np
import time

class RMLR(object):
    def __init__(self, lam, lr):
        self.W = None
        self.lambda_ = lam
        self.learningrate = lr
        self.iter = 50
        self.bs = 1000

        sigma = 1
        self.W = sigma * np.random.randn(2000,5)

    def train(self, x, y, x_tst, y_tst,xdev,xtest):
        n = x.shape[0]

        for i in range(self.iter):
            #start = time.time()
            if i % 10 ==0:
                print(i,"/",self.iter)
            xBatch = None
            yBatch = None

            # creating batch
            for j in range(0, n, self.bs):
                xBatch = x[j:j + self.bs,:]
                yBatch = y[j:j + self.bs]

                XW = xBatch.dot(self.W)
                e_XW = np.exp(XW)
                sum_e_XW = np.sum(e_XW, axis=1)
                prob_y = e_XW / np.reshape(sum_e_XW, (-1, 1))
                prob_sum = np.sum(prob_y * yBatch, axis=1)
                loss = np.sum(np.log(prob_sum)) + (self.lambda_ / 2) * np.sum(self.W * self.W)

                dW_0 = np.array(xBatch.multiply(np.reshape(yBatch[:, 0] - (e_XW[:, 0] / sum_e_XW), (-1, 1))).sum(axis=0))[0, :]
                dW_0 = dW_0.reshape(-1, 1)
                dW_1 = np.array(xBatch.multiply(np.reshape(yBatch[:, 1] - (e_XW[:, 1] / sum_e_XW), (-1, 1))).sum(axis=0))[0, :]
                dW_1 = dW_1.reshape(-1, 1)
                dW_2 = np.array(xBatch.multiply(np.reshape(yBatch[:, 2] - (e_XW[:, 2] / sum_e_XW), (-1, 1))).sum(axis=0))[0, :]
                dW_2 = dW_2.reshape(-1, 1)
                dW_3 = np.array(xBatch.multiply(np.reshape(yBatch[:, 3] - (e_XW[:, 3] / sum_e_XW), (-1, 1))).sum(axis=0))[0, :]
                dW_3 = dW_3.reshape(-1, 1)
                dW_4 = np.array(xBatch.multiply(np.reshape(yBatch[:, 4] - (e_XW[:, 4] / sum_e_XW), (-1, 1))).sum(axis=0))[0, :]
                dW_4 = dW_4.reshape(-1, 1)
                dW = np.concatenate((dW_0, dW_1, dW_2, dW_3, dW_4), axis=1) + self.lambda_ * self.W

                self.W += self.learningrate * dW

            val=np.nonzero(y_tst)[1]

        rmse = self.calRMSE(x_tst,val)
        acc  = self.calAccuracy(x_tst, val)
        print("loss,acc,rmse:", loss, acc, rmse)
        # print("time :", time.time() - start)

        #FOR SVM
        '''g = open("DF_predict_dev.txt", 'r')
        lines = g.readlines()
        #y_hard = self.hard_predict(xdev)
        y_soft = self.soft_predict(xdev)
        f = open("dev-predictions.txt", 'w')
        for i in range(len(y_soft)):
            temp = str(lines[i]).rstrip('\n') + " " + str(y_soft[i] + 1)
            f.write(temp)
            f.write("\n")
        f.close()'''

        #FOR MY ALGORITHM
        y_hard = self.hard_predict(xdev)
        y_soft = self.soft_predict(xdev)
        f = open("dev-predictions.txt", 'w')
        for i in range(len(y_hard)):
            temp = str(y_hard[i] + 1) + " " + str(y_soft[i] + 1)
            f.write(temp)
            f.write("\n")
        f.close()

        y_hard = self.hard_predict(xtest)
        y_soft = self.soft_predict(xtest)
        f = open("test-predictions.txt", 'w')
        for i in range(len(y_hard)):
            temp = str(y_hard[i]+1) + " " + str(y_soft[i]+1)
            f.write(temp)
            f.write("\n")
        f.close()


    def hard_predict(self, x):
        XW = x.dot(self.W)
        e_XW = np.exp(XW)
        sum_e_XW = np.sum(e_XW, axis=1)
        prob_y = e_XW / np.reshape(sum_e_XW, (-1, 1))

        y = np.argmax(prob_y,axis=1)

        return y

    def soft_predict(self, x):
        y = 0
        XW = x.dot(self.W)
        e_XW = np.exp(XW)
        sum_e_XW = np.sum(e_XW, axis=1)

        for i in range(5):
            y += i*(np.exp(x.dot(self.W[:,i]))/sum_e_XW)

        return y

    def calAccuracy(self, x, y):
        acc = 0
        yPred = self.hard_predict(x)

        acc = np.mean(y == yPred) * 100
        return acc

    def calRMSE(self, x, y):
        rmse = 0
        yPred = self.soft_predict(x)

        rmse = np.sqrt(np.sum((y - yPred)*(y - yPred),axis=0)/np.size(y,axis=0))
        return rmse
