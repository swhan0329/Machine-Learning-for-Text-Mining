import numpy as np
import conjugateGradient as cg
from scipy.sparse import csr_matrix
import time

class SVM(object):

    def __init__(self, inputDim, cartegory = 'realsim'):
        self.W = None
        self.newiter = 5

        sigma = 1
        self.W = sigma * np.random.randn(inputDim)

        if cartegory == "realsim":
            print("Dataset: realsim")
            self.W_star = 669.664812
            self.lam = 7230.875
            self.beta = 0.2
            self.lr = 0.0003
            self.iter = 100
            self.bs = 200
        if cartegory == "covtype":
            print("Dataset: covtype")
            self.W_star = 2541.664519
            self.lam = 3631.3203125
            self.beta = 0.2
            self.lr = 0.00003
            self.iter = 50
            self.bs = 1000
    def trainSGD(self, x, y, x_test, y_test):
        print("START SGD")
        start = time.time()
        n = x.shape[0]
        lossHistory, normHistory, accHistory, timeHistory = [], [], [], []
        rt = 0
        x = x.toarray()

        for i in range(self.iter):
            if i % 10 == 0:
                print(i,"/",self.iter)
            xBatch = None
            yBatch = None

            rt = self.lr / (1 + self.beta * i)
            # creating batch
            for j in range(0, n, self.bs):
                xBatch = x[j:j + self.bs,:]
                yBatch = y[j:j + self.bs]

                xx = 1 - yBatch * xBatch.dot(self.W)
                I = np.squeeze((xx > 0))
                XI = xBatch[I, :]

                dW = self.W + 2 * self.lam / self.bs * np.dot(XI.T, (np.dot(XI, self.W) - yBatch[I]))

                self.W = self.W - rt * dW
                timeHistory.append(time.time() - start)

                norm = np.linalg.norm(dW,2)

                loss = 0.5 * np.dot(self.W.T, self.W) + self.lam / self.bs * np.dot(np.maximum(xx, 0).T, np.maximum(xx, 0))
                rf = (loss - self.W_star) / self.W_star
                acc = self.calAccuracy(x_test, y_test)

                normHistory.append(norm)
                lossHistory.append(rf)
                accHistory.append(acc)
        print("SGD_accuracy:",accHistory[-1])
        print("SGD_total time:",timeHistory[-1])
        print("FINISH SGD")
        return  lossHistory, normHistory,accHistory, timeHistory

    def trainNEW(self, x, y,x_test, y_test):
        print("START NM")
        start = time.time()
        n = x.get_shape()[0]

        lossHistory, normHistory, accHistory, timeHistory = [], [], [], []
        for i in range(self.newiter):
            print(i,"/",self.newiter)
            xW = x.dot(self.W)
            yxW = y * xW

            I = np.nonzero((np.ones_like(yxW) - yxW) > 0)[0]

            XI = x.toarray()
            XI = XI[I, :]
            XI = csr_matrix(XI)

            loss_seg = XI.dot(self.W) - y[I]
            dW = self.W + 2 * self.lam / n * XI.transpose().dot(loss_seg)
            d, _ = cg.conjugateGradient(x, I, dW, self.lam)

            self.W = self.W + d

            timeHistory.append(time.time() - start)

            xx = 1 - y * x.dot(self.W)
            loss = 0.5 * np.dot(self.W.T, self.W) + self.lam / n * np.dot(np.maximum(xx,0).T,np.maximum(xx,0))
            rf = (loss - self.W_star)/self.W_star
            norm = np.linalg.norm(dW, 2)
            acc = self.calAccuracy(x_test, y_test)

            normHistory.append(norm)
            lossHistory.append(rf)
            accHistory.append(acc)
        print("NM_accuracy:", accHistory[-1])
        print("NM_total time:", timeHistory[-1])
        print("FINISH NM")
        return lossHistory, normHistory, accHistory, timeHistory

    def predict(self, x, ):
        yPred = np.zeros(x.shape[0])

        s = x.dot(self.W)
        for i in range(np.size(x,axis=0)):
            if s[i] >= 0:
                yPred[i] = 1
            else:
                yPred[i] = -1
        return yPred

    def calAccuracy(self, x, y):
        acc = 0
        yPred = self.predict(x)
        acc = np.mean(y == yPred) * 100
        return acc
