# *** weighted mean ***

n = int(input())
d = list(map(int, input().split(' ')))
w = list(map(int, input().split(' ')))

def weighted_mean(d, w, n):

    if len(d) != len(w):
        raise Exception("Weight and Data vectors are differnt lengths.")
   
    s = 0
    s_denominator = 0
    for i in range(len(d)):
        s += d[i]*w[i]
        s_denominator += w[i]

    return(s/s_denominator)

# result = weighted_mean(d, w, n)