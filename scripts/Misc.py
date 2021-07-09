# from cv2 import cv2
# import numpy as np
# import os

# from tensorflow.python.keras.backend import dtype


# imagesFalse = []
# count = 0
# for i in os.listdir(f'./class_false/'):
#     if count == 7:
#         imagesFalse.append(cv2.resize(cv2.imread(f'./class_false/{i}'), (224, 224)))
#         count = 0
#     count+=1

# np.save("./false", np.array(imagesFalse, dtype='float32'))


def countUnique(s):
    h = {}
    for i in s:
        if i not in h:
            h[i] = 0
        h[i] += 1
    return h


def check(h1, h2):
    for i in h1:
        if i not in h2:
            return False
        if h1[i] > h2[i]:
            return False
    return True


def getDifference(h1, h2):
    h3 = {}
    for i in h1:
        if i not in h2:
            h3[i] = h1[i]
        else:
            h3[i] = h1[i] - h2[i]
    return h3


def MinWindowSubstring2(strArr):
    s1 = strArr[0]
    s2 = strArr[1]
    s2Map, s2Len = countUnique(s2)
    for i in range(s2Len, len(s1) + 1):
        for j in range(0, len(s1) - i + 1):
            currStr = s1[j : j + i]
            # print(i, j, currStr)
            if check(s2Map, countUnique(currStr)[0]):
                return currStr

    return -1


def MinWindowSubstring(strArr):
    s1 = strArr[0]
    s2 = strArr[1]
    print(s1, s2)
    s2Map = countUnique(s2)

    beg = 0
    end = len(s1) - 1

    currStr = s1[beg : beg + end + 1]
    currStrMap = countUnique(currStr)

    diff = getDifference(s2Map, currStrMap)


    while end > 0:
        if s1[end] not in diff:
            end -= 1
        elif diff[s1[end]] < 0:
            diff[s1[end]] += 1
            end -= 1
        else:
            break
        print(end, s1[beg : beg + end + 1], diff)

    while beg < end:
        if s1[beg] not in diff:
            beg += 1
        elif diff[s1[beg]] < 0:
            diff[s1[beg]] += 1
            beg += 1
        else:
            return s1[beg : end + 1]
        print(beg, s1[beg : end + 1], diff)

    currStr = s1[beg : end + 1]
    return currStr


# keep this function call here
print(MinWindowSubstring(input().split(" ")))