import numpy as np


a = np.array([[1,2,3],[5,5,6]])

un_centre = np.mean(a,axis= 0)

b= np.array([7,8,9])

pos_v = un_centre - b

pos_v = pos_v / np.linalg.norm(pos_v)

print(pos_v)