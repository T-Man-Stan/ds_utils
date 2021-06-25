import math

n = int(input())
arr = list(map(int, input().split(' ')))

def mean(arr):
    s = 0
    for i in arr:
        s += i
    
    return(s/len(arr))


def std(arr):
    mu = mean(arr)
    num = 0
    for i in arr:
        num += (i - mu)**2

    return (round(math.sqrt(num/len(arr)), 1))

