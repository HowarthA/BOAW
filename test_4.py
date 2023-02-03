import numpy as np
from sklearn.metrics import pairwise_distances

rep1 = np.array([[1,2,],[3,4],[5,6]])

rep2 = np.array([[1,2,],[8,9]])

W = pairwise_distances(rep1, rep2)


print(W[:, :, np.newaxis])

#W = np.array([[2,2],[4,4],[8,8]])

residual2 = np.sum(np.abs(rep1)) + np.sum(np.abs(rep2))

temp = np.abs(rep1[:, np.newaxis, :] + rep2) - 0.5* np.abs(rep1[:, np.newaxis, :] - rep2)

residual2 -= np.sum(W[:, :, None] * temp)


print(residual2)
print()

residual = np.sum(abs(rep1)) + np.sum(abs(rep2))

for i, b1 in enumerate(rep1):

    for j, b2 in enumerate(rep2):

        print(W[i, j])

        residual -= np.sum(W[i, j] * (abs(b1 + b2) - 0.5 * abs(b1 - b2)))

        print( (abs(b1 + b2) - 0.5 * abs(b1 - b2)))

        print(W[i, j] * (abs(b1 + b2) - 0.5 * abs(b1 - b2)))


print(residual)