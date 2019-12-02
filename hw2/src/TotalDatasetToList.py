import PositiveDatasetToList as PDTL
import NegativeDatasetToList as NDTL
import pandas

Positive_train_dataset_dir = "../data/train/positive/"
Negative_train_dataset_dir = "../data/train/negative/"
Positive_test_dataset_dir = "../data/test/positive/"
Negative_test_dataset_dir = "../data/test/negative/"

def TotalDatasetToList():
    Pos_train_DF,  total_leng_pos, max_leng_pos = PDTL.PositiveTrainDatasetToList(Positive_train_dataset_dir)
    pos_ex = len(Pos_train_DF['text'])
    Pos_test_DF = PDTL.PositiveTestDatasetToList(Positive_test_dataset_dir)
    Neg_train_DF,  total_leng_neg, max_leng_neg = NDTL.NegativeTrainDatasetToList(Negative_train_dataset_dir)
    neg_ex = len(Pos_test_DF['text'])
    print("the ratio of positive examples to negative examples in T: ", pos_ex/neg_ex)
    Neg_test_DF = NDTL.NegativeTestDatasetToList(Negative_test_dataset_dir)
    print("the average length of document in T: ", (total_leng_pos+total_leng_neg)/2000)
    if max_leng_pos > max_leng_neg:
        max = max_leng_pos
    else:
        max = max_leng_neg
    print("the max length of document in T: ", max)
    train_DF = pandas.concat([Pos_train_DF, Neg_train_DF])
    print("the total number of training examples in T: ", len(train_DF['text']))
    test_DF = pandas.concat([Pos_test_DF, Neg_test_DF])

    return test_DF, train_DF
