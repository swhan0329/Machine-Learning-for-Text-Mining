import RMLR
import pickle
import numpy as np

def CTF_final():
    with open('CTF_text_dict_trn.pickle', 'rb') as handle:
        clean_text_trn = pickle.load(handle)
    with open('CTF_text_dict_dev.pickle', 'rb') as handle:
        clean_text_dev = pickle.load(handle)
    with open('CTF_text_dict_test.pickle', 'rb') as handle:
        clean_text_test = pickle.load(handle)
    with open('star_trn.pickle', 'rb') as handle:
        star = pickle.load(handle)

    split = int(0.8 * clean_text_trn.shape[0])

    Xtrain = clean_text_trn[:split,:]
    Xdev = clean_text_trn[split:,:]
    Ytrain = star[:split,:]
    Ydev = star[split:,:]

    rmlr = RMLR.RMLR(lam=0.001, lr=0.001)
    rmlr.train(Xtrain,Ytrain,Xdev,Ydev,clean_text_dev,clean_text_test)

def DF_final():
    with open('DF_text_dict_trn.pickle', 'rb') as handle:
        clean_text_trn = pickle.load(handle)
    with open('DF_text_dict_dev.pickle', 'rb') as handle:
        clean_text_dev = pickle.load(handle)
    with open('DF_text_dict_test.pickle', 'rb') as handle:
        clean_text_test = pickle.load(handle)
    with open('star_trn.pickle', 'rb') as handle:
        star = pickle.load(handle)

    split = int(0.8 * clean_text_trn.shape[0])

    Xtrain = clean_text_trn[:split,:]
    Xdev = clean_text_trn[split:,:]
    Ytrain = star[:split,:]
    Ydev = star[split:,:]

    rmlr = RMLR.RMLR(lam=0.001, lr=0.001)
    rmlr.train(Xtrain,Ytrain,Xdev,Ydev,clean_text_dev,clean_text_test)