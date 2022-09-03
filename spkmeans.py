import sys
import numpy
import pandas as pd
import mykmeanssp

numpy.random.seed(0)

def invalidInput():
    print("Invalid Input!")
    sys.exit()


def errorOccurred():
    print("An Error Has Occurred")
    sys.exit()


def euclidean_norm_squared(vector1, vector2):
    # this function determines the euclidean distance of two vectors, WITHOUT taking a sqrt at the end.

    return numpy.square(numpy.linalg.norm(vector1-vector2))


# input validations

n = len(sys.argv)  # num of cmd line arguments, helps us determine where our different inputs are located


if (n < 3 or n > 4):  # invalid number of cmd line args
    invalidInput()
if (n == 3):
    goal = sys.argv[1]
    file_name = sys.argv[2]

if (n == 4):
    k = sys.argv[1]
    if not k.isdigit():  # k is not an integer
        invalidInput()
    k = int(k)
    if k < 0:
        invalidInput()
    goal = sys.argv[2]
    file_name = sys.argv[3]
if(goal != "spk" and goal != "wam" and goal != "ddg" and goal != "lnorm" and goal != "jacobi"):
    invalidInput()
    

# after receiving and validating cmd line args, preforming inner join using pandas:

try:
    
    print ("inside try")
    with open(file_name) as file:
        list_of_all_vectors = [[float(num) for num in line.split(",")] for line in file]
        vector_table = numpy.array(list_of_all_vectors)

        print (vector_table)
        print (type(vector_table))
        print (list_of_all_vectors)
        print (type(list_of_all_vectors))
    

    num_of_vectors = len(list_of_all_vectors)  # counting the number of vectors provided
    print (num_of_vectors)

    # determining dimension of vectors
    if len(list_of_all_vectors) != 0:
        dimension = len(list_of_all_vectors[
                            0])  # dimension of first vector in the list (assuming input is valid, so all vectors
        # have the same dimension)
        print ("dimension is: ")
        print (dimension)
    else:  # there are no vectors in the list!
        errorOccurred()

    index_list = []  # index list: will be returned to the user at the end of the run

    curr_miu_index = numpy.random.choice([ i for i in range (0, num_of_vectors)])
    print ("curr miu")
    print (curr_miu_index)

    index_list.append(curr_miu_index)

    miu_list = [vector_table[curr_miu_index]]  # miu_list now holds the first miu that has been chosen randomly
    print (miu_list)
    curr_miu = miu_list[0]
    prob_list = [0 for i in range(0, num_of_vectors)]
    d_list = [numpy.inf for i in range(0, num_of_vectors)]  # for x_l, d_list[l] is D_l

    # len(miu_list) = i, and it is 1 right now

    while len(miu_list) < k:
        print (k)
        for i in range(0, num_of_vectors):  # for every vector in the list:
            curr_vector = vector_table[i]

            for miu in miu_list:  # for every miu in miu list, we check the min. euclidean distance from curr_vector,
                # and update d_list accordingly
                curr_norm = euclidean_norm_squared(curr_vector, miu)
                if curr_norm < d_list[i]:
                    d_list[i] = curr_norm

        d_sum = sum(d_list)  # sum of all D's (sigma (i from 0 to num_of_vectors) : D_i)

        for i in range(0, num_of_vectors):
            prob_list[i] = d_list[i] / d_sum  # for x_l :  D_l/(sigma (i from 0 to num_of_vectors) : D_i)

        curr_miu_index = numpy.random.choice(vector_table.shape[0], size=1, replace=False, p=prob_list)[
            0]  # choosing a vector randomly, with probabilities (prob_list) taken to account
        index_list.append(curr_miu_index)
        curr_miu = vector_table[curr_miu_index]
        miu_list = numpy.vstack([miu_list, [curr_miu]])

    # CALLING CLUSTERING METHOD FROM HW1

    print (file_name)
    print (index_list)
    print(k)
    print (num_of_vectors)
    print(dimension)
    print (goal)


    cent_list = spkmeansmodule.fit(goal, file_name, index_list, k, num_of_vectors, dimension)
    print ("hello")
    # retrieving original indices of chosen k vectors

    for i in range(0, len(index_list)):  # printing indices (with commas)
        if i == (len(index_list) - 1):
            print(index_list[i])
        else:
            print(index_list[i], ",", sep='', end='')

    for i in range(0, len(cent_list)):  # printing centroids (by coordinates with commas)
        for j in range(0, len(cent_list[i])):
            if j == (len(cent_list[i]) - 1):
                print("%.4f" % cent_list[i][j])

            else:
                print("%.4f" % cent_list[i][j], ",", sep='', end='')

except:
    errorOccurred()
