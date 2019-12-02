import pickle

print("SVM_CTF_TRAIN")
with open('CTF_text_dict_trn.pickle', 'rb') as handle:
    matrix = pickle.load(handle)


with open('star_ori_trn.pickle', 'rb') as handle:
        star = pickle.load(handle)
handle.close()

print("SVM train set making....")
f = open("SVM_train.txt", 'w')
for i in range(matrix.shape[0]):
    if i % 100000 == 0:
        print(i)

    f.write(str(star[i]))
    f.write(" ")

    for count, feature in zip(matrix[i].data,matrix[i].nonzero()[1]):
        temp = str(feature+1) + ":" + str(count)
        f.write(temp)
        f.write(" ")
    f.write("\n")
f.close()

print("SVM_CTF_DEV")
with open('CTF_text_dict_dev.pickle', 'rb') as handle:
    matrix = pickle.load(handle)

handle.close()

print("SVM dev set making....")
f = open("SVM_dev.txt", 'w')
for i in range(matrix.shape[0]):
    if i % 100000 == 0:
        print(i)

    f.write("0")
    f.write(" ")

    for count, feature in zip(matrix[i].data,matrix[i].nonzero()[1]):
        temp = str(feature+1) + ":" + str(count)
        f.write(temp)
        f.write(" ")
    f.write("\n")
f.close()

print("SVM_CTF_TEST")
with open('CTF_text_dict_test.pickle', 'rb') as handle:
    matrix = pickle.load(handle)

handle.close()

print("SVM test set making....")
f = open("SVM_test.txt", 'w')
for i in range(matrix.shape[0]):
    if i % 100000 == 0:
        print(i)

    f.write("0")
    f.write(" ")

    for count, feature in zip(matrix[i].data,matrix[i].nonzero()[1]):
        temp = str(feature+1) + ":" + str(count)
        f.write(temp)
        f.write(" ")
    f.write("\n")
f.close()
print("SVM_DF_TRAIN")
with open('DF_text_dict_trn.pickle', 'rb') as handle:
    matrix = pickle.load(handle)

with open('star_ori_trn.pickle', 'rb') as handle:
        star = pickle.load(handle)
handle.close()

print("SVM train set making....")
f = open("SVM_train.txt", 'w')
for i in range(matrix.shape[0]):
    if i % 100000 == 0:
        print(i)

    f.write(str(star[i]))
    f.write(" ")

    for count, feature in zip(matrix[i].data,matrix[i].nonzero()[1]):
        temp = str(feature+1) + ":" + str(count)
        f.write(temp)
        f.write(" ")
    f.write("\n")
f.close()

print("SVM_DF_DEV")
with open('DF_text_dict_dev.pickle', 'rb') as handle:
    matrix = pickle.load(handle)

handle.close()

print("SVM dev set making....")
f = open("SVM_dev.txt", 'w')
for i in range(matrix.shape[0]):
    if i % 100000 == 0:
        print(i)

    f.write("0")
    f.write(" ")

    for count, feature in zip(matrix[i].data,matrix[i].nonzero()[1]):
        temp = str(feature+1) + ":" + str(count)
        f.write(temp)
        f.write(" ")
    f.write("\n")
f.close()

print("SVM_DF_TEST")
with open('DF_text_dict_test.pickle', 'rb') as handle:
    matrix = pickle.load(handle)

handle.close()

print("SVM test set making....")
f = open("SVM_test.txt", 'w')
for i in range(matrix.shape[0]):
    if i % 100000 == 0:
        print(i)

    f.write("0")
    f.write(" ")

    for count, feature in zip(matrix[i].data,matrix[i].nonzero()[1]):
        temp = str(feature+1) + ":" + str(count)
        f.write(temp)
        f.write(" ")
    f.write("\n")
f.close()