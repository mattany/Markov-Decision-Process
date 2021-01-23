import numpy as np
from MDP import *

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


states = [State("", 0, 1),
          State("A", 1, 0),
          State("AB", 2, 0),
          State("ABB", 3, 0),
          State("ABBA", 4, 0),
          ]

transitions = [Transition(0, 1, 0.4),
               Transition(0, 0, 0.6),
               Transition(1, 1, 0.4),
               Transition(1, 2, 0.6),
               Transition(2, 1, 0.4),
               Transition(2, 3, 0.6),
               Transition(3, 4, 0.4),
               Transition(3, 0, 0.6),
               Transition(4, 4, 1)]

markov_process = MDP(states, transitions)
print(markov_process)
markov_process.show_transition_matrix()
markov_process.progress_k_steps(50)
print(markov_process)
markov_process.reset()
print(markov_process)
print(markov_process.expected_number_of_steps_between_states("", "ABBA"))


# state_vector = np.array([0] * 6)
# state_vector[0] = 1
# transition_matrix = np.array(
#     [[1 / 2] + [1 / 2 if j == i else 0 for j in range(5)] for i in range(5)] + [[0, 0, 0, 0, 0, 1]])
# print(state_vector)
# print(transition_matrix)
# for i in range(62):
#     state_vector = np.dot(state_vector, transition_matrix)
#     print(state_vector)
