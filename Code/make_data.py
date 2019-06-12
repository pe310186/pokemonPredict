from data import *
from joblib import dump, load
import numpy as np
import random

if __name__ == '__main__':
    real_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 148, 149]
    X, y = get_data_sparse()
    id_list = [[] for i in range(151)]
    for i in range(X.shape[0]):
        id_list[y[i]].append(i)
    # 10k per file 10 file
    
    for i in range(10):
        x_buf = []
        y_buf = []
        for k in range(10000):
            idx = random.choice(id_list[random.choice(real_idx)])
            x_buf.append(X[idx])
            y_buf.append(y[idx])
        dump(np.asarray(x_buf),'X-{}.jl'.format(i)) 
        dump(np.asarray(y_buf),'y-{}.jl'.format(i))