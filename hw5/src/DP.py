import json
import string
import pickle
from string import digits
import numpy as np

def train():
    print("DATA_PREPROCESSING_TRAIN")
    ifile = open('../data/yelp_reviews_train.json')
    train_len = 0

    stars_list = []
    text_dict = {}

    print("loading data....")
    for i, line in enumerate(ifile):
        if i%100000==0:
            print(i)
        #if i == 10:
         #   break
        # convert the json on this line to a dict
        data = json.loads(line)
        # extract what we want
        stars = data['stars']
        text_dict[i] = data['text']

        # add to the data collected so far
        stars_list.append(stars)
        train_len += 1

    def remove_digits(list):
        remove_digit = str.maketrans('', '', digits)
        list = [i.translate(remove_digit) for i in list]
        return list

    with open("../data/stopword.list",'r') as file:
        stopwords = [line.strip() for line in file]

    table = str.maketrans('', '', string.punctuation+"\n")

    print("cleaning text....")
    for i in range(train_len):
        if i % 100000 == 0:
            print(i, "/", train_len)
        text_dict[i] = text_dict[i].lower()
        text_dict[i] = text_dict[i].split()
        text_dict[i] = [w.translate(table) for w in text_dict[i]]  # Remove all the punctuation
        text_dict[i] = [w for w in text_dict[i] if w.isalpha()]
        text_dict[i] = remove_digits(text_dict[i])  # Remove all tokens that are contain numbers
        text_dict[i] = [word for word in text_dict[i] if word not in stopwords]
        text_dict[i] = list(filter(None, text_dict[i]))  # Remove all tokens that are empty.

    with open('clean_text_trn.pickle', 'wb') as handle:
        pickle.dump(text_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("getting frequence words....")
    freq_words={}
    for i in range(train_len):
        if i % 100000 == 0:
            print(i,"/",train_len)
        for item in text_dict[i]:
            if (item in freq_words):
                freq_words[item] += 1
            else:
                freq_words[item] = 1

    freq_words_sorted = sorted(freq_words.items() ,  key=lambda x: x[1],reverse=True)

    with open('CTF_freq_words_sorted.pickle', 'wb') as handle:
        pickle.dump(freq_words_sorted, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #####Data Preprocessing(1)#####
    for i in range(9):
        print(freq_words_sorted[i])

    #####Data Preprocessing(2)#####
    num_one = stars_list.count(1)
    num_two = stars_list.count(2)
    num_three = stars_list.count(3)
    num_four = stars_list.count(4)
    num_five = stars_list.count(5)

    print(num_one,num_two,num_three,num_four,num_five)
    print(num_one/train_len*100,num_two/train_len*100,num_three/train_len*100,num_four/train_len*100,num_five/train_len*100)

    with open('star_ori_trn.pickle', 'wb') as handle:
        pickle.dump(stars_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    stars_list = np.asarray(stars_list)
    stars_list = stars_list.T
    n_values = np.max(stars_list)
    matrix = np.eye(n_values)[stars_list-1]
    with open('star_trn.pickle', 'wb') as handle:
        pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

def dev():
    print("DATA_PREPROCESSING_DEV")
    ifile = open('../data/yelp_reviews_dev.json')
    dev_len = 0

    text_dict = {}

    print("loading data....")
    for i, line in enumerate(ifile):
        if i % 100000 == 0:
            print(i)
        # if i == 10:
        #   break
        # convert the json on this line to a dict
        data = json.loads(line)
        # extract what we want
        text_dict[i] = data['text']

        # add to the data collected so far
        dev_len += 1

    def remove_digits(list):
        remove_digit = str.maketrans('', '', digits)
        list = [i.translate(remove_digit) for i in list]
        return list

    with open("../data/stopword.list", 'r') as file:
        stopwords = [line.strip() for line in file]

    table = str.maketrans('', '', string.punctuation+"\n")

    print("cleaning text....")
    for i in range(dev_len):
        if i % 100000 == 0:
            print(i, "/", dev_len)
        text_dict[i] = text_dict[i].lower()
        text_dict[i] = text_dict[i].split()
        text_dict[i] = [w.translate(table) for w in text_dict[i]]  # Remove all the punctuation
        text_dict[i] = [w for w in text_dict[i] if w.isalpha()]
        text_dict[i] = remove_digits(text_dict[i])  # Remove all tokens that are contain numbers
        text_dict[i] = [word for word in text_dict[i] if word not in stopwords]
        text_dict[i] = list(filter(None, text_dict[i]))  # Remove all tokens that are empty.

    with open('clean_text_dev.pickle', 'wb') as handle:
        pickle.dump(text_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test():
    print("DATA_PREPROCESSING_TEST")
    ifile = open('../data/yelp_reviews_test.json')
    test_len = 0

    text_dict = {}

    print("loading data....")
    for i, line in enumerate(ifile):
        if i % 100000 == 0:
            print(i)
        # if i == 10:
        #   break
        # convert the json on this line to a dict
        data = json.loads(line)
        # extract what we want
        text_dict[i] = data['text']

        # add to the data collected so far
        test_len += 1

    def remove_digits(list):
        remove_digit = str.maketrans('', '', digits)
        list = [i.translate(remove_digit) for i in list]
        return list

    with open("../data/stopword.list", 'r') as file:
        stopwords = [line.strip() for line in file]

    table = str.maketrans('', '', string.punctuation+"\n")

    print("cleaning text....")
    for i in range(test_len):
        if i % 100000 == 0:
            print(i, "/", test_len)
        text_dict[i] = text_dict[i].lower()
        text_dict[i] = text_dict[i].split()
        text_dict[i] = [w.translate(table) for w in text_dict[i]]  # Remove all the punctuation
        text_dict[i] = [w for w in text_dict[i] if w.isalpha()]
        text_dict[i] = remove_digits(text_dict[i])  # Remove all tokens that are contain numbers
        text_dict[i] = [word for word in text_dict[i] if word not in stopwords]
        text_dict[i] = list(filter(None, text_dict[i]))  # Remove all tokens that are empty.

    with open('clean_text_test.pickle', 'wb') as handle:
        pickle.dump(text_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
