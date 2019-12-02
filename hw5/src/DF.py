import pickle
from scipy.sparse import csr_matrix

def train():
    print("DF_TRAIN")
    with open('clean_text_trn.pickle', 'rb') as handle:
        clean_text = pickle.load(handle)

    clean_text_len = len(clean_text)

    freq_words={}
    for i in range(clean_text_len):
        if i % 10000 == 0:
            print(i)
        set_text_dict = set(clean_text[i])
        for item in set_text_dict:
            if (item in freq_words):
                freq_words[item] += 1
            else:
                freq_words[item] = 1

    freq_words_sorted = sorted(freq_words.items() ,  key=lambda x: x[1],reverse=True)

    with open('DF_freq_words_sorted.pickle', 'wb') as handle:
        pickle.dump(freq_words_sorted, handle, protocol=pickle.HIGHEST_PROTOCOL)

    top2000_dict = freq_words_sorted[:2000]
    top2000_dict = dict(top2000_dict)
    clean_text_len = len(clean_text)
    row,col,data=[],[],[]
    top2000_list=list(top2000_dict.keys())
    print("making 2000 matrix....")
    for i in range(clean_text_len):
        if i % 100000 == 0:
            print(i,"/",clean_text_len)
        text_top2000_idx = [top2000_list.index(word) for word in clean_text[i] if
                            word in top2000_list]
        tdata = [clean_text[i].count(top2000_list[top2000_idx]) for top2000_idx in
                 set(text_top2000_idx)]
        tcol = list(set(text_top2000_idx))
        trow = [i] * len(tcol)
        col.extend(tcol)
        row.extend(trow)
        data.extend(tdata)
    matrix = csr_matrix((data,(row,col)))

    with open('DF_text_dict_trn.pickle', 'wb') as handle:
        pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

def dev():
    print("DF_DEV")
    with open('clean_text_dev.pickle', 'rb') as handle:
        clean_text = pickle.load(handle)

    with open('DF_freq_words_sorted.pickle', 'rb') as handle:
        DF_freq_words_sorted = pickle.load(handle)

    top2000_dict = DF_freq_words_sorted[:2000]
    top2000_dict = dict(top2000_dict)
    clean_text_len = len(clean_text)
    row,col,data=[],[],[]
    top2000_list=list(top2000_dict.keys())
    print("making 2000 matrix....")
    for i in range(clean_text_len):
        if i % 100000 == 0:
            print(i,"/",clean_text_len)
        text_top2000_idx = [top2000_list.index(word) for word in clean_text[i] if
                            word in top2000_list]
        tdata = [clean_text[i].count(top2000_list[top2000_idx]) for top2000_idx in
                 set(text_top2000_idx)]
        tcol = list(set(text_top2000_idx))
        trow = [i] * len(tcol)
        col.extend(tcol)
        row.extend(trow)
        data.extend(tdata)
    matrix = csr_matrix((data,(row,col)))
    with open('DF_text_dict_dev.pickle', 'wb') as handle:
        pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test():
    print("DF_TEST")
    with open('clean_text_test.pickle', 'rb') as handle:
        clean_text = pickle.load(handle)
    with open('DF_freq_words_sorted.pickle', 'rb') as handle:
        DF_freq_words_sorted = pickle.load(handle)

    top2000_dict = DF_freq_words_sorted[:2000]
    top2000_dict = dict(top2000_dict)
    clean_text_len = len(clean_text)
    row,col,data=[],[],[]
    top2000_list=list(top2000_dict.keys())

    print("making 2000 matrix....")
    for i in range(clean_text_len):
        if i % 100000 == 0:
            print(i,"/",clean_text_len)
        text_top2000_idx = [top2000_list.index(word) for word in clean_text[i] if
                            word in top2000_list]
        tdata = [clean_text[i].count(top2000_list[top2000_idx]) for top2000_idx in
                 set(text_top2000_idx)]
        tcol = list(set(text_top2000_idx))
        trow = [i] * len(tcol)
        col.extend(tcol)
        row.extend(trow)
        data.extend(tdata)
    matrix = csr_matrix((data,(row,col)))

    with open('DF_text_dict_test.pickle', 'wb') as handle:
        pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
