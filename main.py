import numpy as np
from Markov_Chain import *
# from state

events = [Event("A",0.4), Event("B", 0.6)]
goal_states = ["ABA", "BAB"]
markov_chain = Markov_Chain.get_chain_from_events(goal_states, events)
markov_chain.draw_transitions()

# markov_process.show_transition_matrix()
# e = markov_process.get_expected_number_of_steps_to_goal("", "ABBA")
# print(e)
# goal_state = "ABAB"
# markov_process = Markov_Chain.get_chain_from_events(goal_state, events)
# e = markov_process.get_expected_number_of_steps_to_goal("", "ABAB")
# print(e)


# events = [Event("H", 0.5), Event("T", 0.5)]
# goal_state = "HHHHH"
# markov_chain = Markov_Chain.get_chain_from_events(goal_state, events)
# # markov_chain.draw_transitions()
# #
# #
