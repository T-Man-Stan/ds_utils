import statistics

# statistics.median(arr)

# Implement from scratch:
def median(data):

    data = sorted(data)
    n = len(data)
    if n == 0:
        raise Exception("Test")
    if n%2 == 1:
        return data[n//2]
    else:
        i = n//2
        return (data[i - 1] + data[i])/2
          
# median(arr)