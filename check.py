import numpy as np

with open("./results/prediction/testset_2014_12min.npy", 'rb') as f:
    test_log = np.load(f)

print(len(test_log))