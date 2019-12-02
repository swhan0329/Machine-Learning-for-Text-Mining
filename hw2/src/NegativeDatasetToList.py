import os
import pandas

def NegativeTrainDatasetToList(train_path):
    file_list = []
    for file in os.listdir(train_path):
        file_list.append(file)

    # load the train dataset
    labels, texts, leng_neg = [],[], []
    for i in range(0, len(file_list)):
        data = open(train_path+str(file_list[i])).read()
        for j, line in enumerate(data.split("\n")):
            content = line.split()
            leng_neg.append(len(content))
            labels.append('0')
            texts.append(" ".join(content[0:]))
    total_leng= sum(leng_neg)
    max_leng = max(leng_neg)
    #print("neg total length of doc: ",total_leng)
    #print("neg max length of document in T:",max_leng)
    #create a dataframe using texts and lables
    trainDF=pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    return trainDF,  total_leng, max_leng

def NegativeTestDatasetToList(test_path):
    file_list = []
    for file in os.listdir(test_path):
        file_list.append(file)

    # load the test dataset
    labels, texts = [],[]
    for i in range(0, len(file_list)):
        data = open(test_path+str(file_list[i])).read()
        for j, line in enumerate(data.split("\n")):
            content = line.split()

            labels.append('0')
            texts.append(" ".join(content[0:]))

    #create a dataframe using texts and lables
    testDF=pandas.DataFrame()
    testDF['text'] = texts
    testDF['label'] = labels

    return testDF