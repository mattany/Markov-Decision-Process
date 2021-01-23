import numpy as np
from Markov_Chain import *

# from state


# Dice
# events = [Event("1", 1 / 6), Event("2", 1 / 6), Event("3", 1 /6), Event("4", 1 / 6), Event("5", 1 / 6), Event("6", 1 / 6)]
# goal_states = ["13", "22", "31"]
# markov_chain = Markov_Chain.get_chain_from_events(goal_states, events)
# e = markov_chain.get_expected_number_of_steps_to_goal("_", "13")
# markov_chain.draw_transitions()


# events = [Event("A", 0.6), Event("B", 0.4)]
# goal_states = ["ABBA", "ABAB"]
# goal = "ABAB"

# events = [Event("H", 1 / 2), Event("T", 1 / 2)]
# goal_states = ["HHHHH"]
# goal = "HHHH"

# events = [Event(chr(_), 1/26) for _ in range(ord('a'), ord('z') + 1)]
# goal_states = ["whatamidoin"]
# goal = "what"


events = [Event("H", 1 / 2), Event("T", 1 / 2)]
goal_states = ["HHT", "HTT"]
goal = "_"


markov_chain = Markov_Chain.get_chain_from_events(goal_states, events)
# e = markov_process.get_expected_number_of_steps_to_goal("_", "ABAB")
# print(e)
markov_chain.progress_k_steps(10000)
print(markov_chain.get_state_vector())
markov_chain.reset()
e = markov_chain.get_expected_number_of_steps_to_goal("_", goal)
print(e)
markov_chain.draw_transitions()