from sympy import symbols, solve
import numpy as np


# def func(n):
#     x = symbols('x')
#     if n == 6:
#         return 0
#     if n == 1:
#         expr = x - (1 + (func(2) + x) / 2)
#         print(expr)
#
#         return solve(expr)
#     expr = 1 + (x + (func(n + 1))) / 2
#     print(expr, n)
#     return expr







state_vector = np.array([0] * 6)
state_vector[0] = 1
transition_matrix = np.array(
    [[1 / 2] + [1 / 2 if j == i else 0 for j in range(5)] for i in range(5)] + [[0, 0, 0, 0, 0, 1]])
print(state_vector)
print(transition_matrix)
for i in range(62):
    state_vector = np.dot(state_vector, transition_matrix)
    print(state_vector)
print(func(1))
