import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Vandermonde(x, d):
    N = x.size
    A = np.zeros((N, d+1))

    for i in range(d+1):
        A[:, i] = x ** i

    return A

def least_squares_solver(A, y):
    inv_gram = np.linalg.inv(np.transpose(A) @ A)
    theta = inv_gram @ np.transpose(A) @ y

    solution = A @ theta
    return solution