from typing import List, NamedTuple
import numpy as np


class State(NamedTuple):
    name: str
    id: int
    initial_probability: float


class Transition(NamedTuple):
    source: State
    sink: State
    probability: float


class MDP(object):
    def __init__(self, states: List[State], transitions: List[Transition]):
        self.state_amount = len(states)
        for i, state in enumerate(states):
            for j in range(i + 1, self.state_amount):
                state_2 = states[j]
                assert(state.id != state_2.id and state.name != state_2.name)

        self.states = sorted(states, key=lambda s: s.id)
        self.transitions = transitions
        self.transition_matrix = None
        self.state_vector = None

        self.construct_matrix()
        self.construct_vector()

    def construct_matrix(self):
        self.transition_matrix = np.array(self.state_amount * [self.state_amount * [0]])
        for t in self.transitions:
            self.transition_matrix[t.source.id, t.sink.id] = t.probability
        assert(sum(t) == 1 for t in self.transition_matrix)

    def construct_vector(self):
        self.state_vector = np.array()
