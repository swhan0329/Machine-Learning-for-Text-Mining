import pandas, os

def PositiveTrainDatasetToList(train_path):
    file_list = []
    for file in os.listdir(train_path):
        file_list.append(file)

    # load the train dataset
    labels, texts, leng_pos = [],[],[]
    for i in range(0, len(file_list)):
        data = open(train_path+str(file_list[i])).read()
        for j, line in enumerate(data.split("\n")):
            content = line.split()
            leng_pos.append(len(content))
            labels.append('1')
            texts.append(" ".join(content[0:]))
    total_leng = sum(leng_pos)
    max_leng = max(leng_pos)
    #print("pos total length of doc: ",total_leng)
    #print("pos max length of document in T:",max_leng)
    #create a dataframe using texts and lables
    trainDF=pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    return trainDF, total_leng, max_leng

def PositiveTestDatasetToList(test_path):
    file_list = []
    for file in os.listdir(test_path):
        file_list.append(file)

    # load the test dataset
    labels, texts = [],[]
    for i in range(0, len(file_list)):
        data = open(test_path+str(file_list[i])).read()
        for j, line in enumerate(data.split("\n")):
            content = line.split()
            labels.append('1')
            texts.append(" ".join(content[0:]))

    #create a dataframe using texts and lables
    testDF=pandas.DataFrame()
    testDF['text'] = texts
    testDF['label'] = labels

    return testDF