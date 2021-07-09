rootDir = './'

# Shuffles the passed arrays while maintaining correspondance
# uses numpy permutation generator
def shuffle(a, b):
    import numpy as np
    p = np.random.permutation(len(a))
    return a[p], b[p]