import sys
from scipy import sparse
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from svm import *

def main():
    # read the train file from first arugment
    train_file = sys.argv[1]

    # read the test file from second argument
    test_file = sys.argv[2]

    car = train_file[8:15]

    X_train, y_train= load_svmlight_file(train_file)
    X_test, y_test= load_svmlight_file(test_file)

    X_train = X_train.toarray()
    X_test= X_test.toarray()

    X_train = np.column_stack((X_train,np.ones(np.size(X_train,axis=0))))
    X_test = np.column_stack((X_test, np.ones(np.size(X_test, axis=0))))

    n = np.size(X_train, axis=1)

    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)

    svm = SVM(n,cartegory = car)
    losshistory_NEW, normHistory_NEW, accHistory_New, timeHistory_NEW = svm.trainNEW(X_train, y_train,X_test, y_test)
    svm = SVM(n,cartegory=car)
    losshistory_SGD, normHistory_SGD, accHistory_SGD, timeHistory_SGD = svm.trainSGD(X_train, y_train, X_test, y_test)
    '''
    plt.figure(1)
    plt.title('Relative function value diﬀerence versus training time')
    plt.plot(timeHistory_NEW, losshistory_NEW,"-.b", linewidth=3,label='Newton method')
    plt.plot(timeHistory_SGD, losshistory_SGD, ':r', linewidth=1,label='Stochastic gradient method')

    plt.legend(loc='upper right')

    plt.figure(2)
    plt.title('gradient norm versus training time')
    plt.plot(timeHistory_NEW, normHistory_NEW,"-.b",linewidth=3, label='Newton method')
    plt.plot(timeHistory_SGD, normHistory_SGD, ':r',linewidth=1, label='Stochastic gradient method')

    plt.legend(loc='upper right')

    plt.figure(3)
    plt.title('test set accuracies versus training time')
    plt.plot(timeHistory_NEW, accHistory_New, "-.b", linewidth=3,label='Newton method')
    plt.plot(timeHistory_SGD, accHistory_SGD, ":r",linewidth=1, label='Stochastic gradient method')

    plt.legend(loc='lower right')
    plt.show()'''

# Main entry point to the program
if __name__ == '__main__':
    main()
