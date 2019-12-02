from KNN_utils import *
def q1(MovieID, UserID, Rating):
    print("==================Question 1.Corpus Exploration==================")
    uni_MovieID = set(MovieID)
    uni_UserID = set(UserID)
    # print(uni_MovieID)
    # print(uni_UserID)
    print("the total number of movies:", len(uni_MovieID))
    print("the total number of users:", len(uni_UserID))
    print("the number of times any movie was rated '1':", Rating.count(1))
    print("the number of times any movie was rated '3':", Rating.count(3))
    print("the number of times any movie was rated '5':", Rating.count(5))
    print("the average movie rating across all users and movies:", sum(Rating) / len(UserID))

    print("\nFor user ID 4321")
    print("the number of movies rated:", UserID.count(4321))
    temp_4321_index = [i for i, x in enumerate(UserID) if x == 4321]
    # print(temp_4321_index)
    total_4321 = 0
    count_1, count_3, count_5 = 0, 0, 0
    for idx in temp_4321_index:
        if Rating[idx] == 1:
            count_1 += 1
        elif Rating[idx] == 3:
            count_3 += 1
        elif Rating[idx] == 5:
            count_5 += 1
        total_4321 += Rating[idx]
    print("the number of times the user gave a '1' rating:", count_1)
    print("the number of times the user gave a '3' rating:", count_3)
    print("the number of times the user gave a '5' rating:", count_5)
    print("the average rating for this user:", total_4321 / len(temp_4321_index))

    print("\nFor movie ID 3")
    print("the number of users rating this movie:", MovieID.count(3))
    temp_3_index = [i for i, x in enumerate(MovieID) if x == 3]
    # print(temp_4321_index)
    total_3 = 0
    count_1, count_3, count_5 = 0, 0, 0
    for idx in temp_3_index:
        if Rating[idx] == 1:
            count_1 += 1
        elif Rating[idx] == 3:
            count_3 += 1
        elif Rating[idx] == 5:
            count_5 += 1
        total_3 += Rating[idx]
    print("the number of times the user gave a '1' rating:", count_1)
    print("the number of times the user gave a '3' rating:", count_3)
    print("the number of times the user gave a '5' rating:", count_5)
    print("the average rating for this movie:", total_3 / len(temp_3_index))
