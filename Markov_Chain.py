import copy
from typing import List, NamedTuple, Tuple
import numpy as np


class State(NamedTuple):
    name: str
    id: int
    initial_probability: float


class Transition(NamedTuple):
    source_id: int
    sink_id: int
    probability: float

class Event(NamedTuple):
    sign: str
    probability: float

class Markov_Chain(object):
    def __init__(self, states: List[State], transitions: List[Transition]):
        self.state_amount = len(states)
        for i, state in enumerate(states):
            for j in range(i + 1, self.state_amount):
                assert (state.name != states[j].name)
        self.states = sorted(states, key=lambda s: s.id)
        assert (self.states[i].id == i for i in range(self.state_amount))
        self.transitions = transitions
        self.transition_matrix = None
        self.state_vector = None
        self.steps = 0
        self.construct_matrix()
        self.construct_vector()

    def construct_matrix(self):
        self.transition_matrix = np.zeros([self.state_amount, self.state_amount], dtype=float)
        for t in self.transitions:
            assert 0 <= t.source_id < self.state_amount and 0 <= t.sink_id < self.state_amount
            self.transition_matrix[t.source_id, t.sink_id] = t.probability
        assert (sum(t) == 1 for t in self.transition_matrix)

    def construct_vector(self):
        self.state_vector = np.array([s.initial_probability for s in self.states])

    def reset(self):
        self.construct_matrix()
        self.construct_vector()
        self.steps = 0

    def progress_k_steps(self, k):
        for i in range(k):
            self.state_vector = np.dot(self.state_vector, self.transition_matrix)
        self.steps += k

    def expected_number_of_steps_between_states(self, source_state: str, dest_state: str):
        src, dest = None, None
        for s in self.states:
            if s.name == source_state:
                src = s
            elif s.name == dest_state:
                dest = s
        assert (src and dest)
        linear_equation = np.copy(self.transition_matrix)
        for i, row in enumerate(linear_equation):
            if i != dest.id:
                linear_equation[i, i] -= 1
        inverse = np.linalg.inv(linear_equation)
        b_vector = np.full((self.state_amount), -1)
        b_vector[dest.id] = 0
        solution_vector = np.dot(inverse, b_vector)
        return solution_vector[src.id]

    def __repr__(self):
        out_str = f"After {self.steps} steps.\n"
        for i, prob in enumerate(self.state_vector):
            out_str += f"State:{self.states[i].name}, Probability: {prob}\n"
        return out_str

    def show_transition_matrix(self):
        print(self.transition_matrix)


def state_generator(goal_event_sequence: str,  events: Tuple[Event]):
    assert sum(e.probability for e in events) == 1
    stack = [State("", 0, 1)]
    ret_val = list()
    i, id = 0, 1
    while stack:
        cur = stack.pop()
        ret_val.append(cur)
        for e in events:
            if goal_event_sequence[i] == e.sign:
                stack.append(State(goal_event_sequence[:i + 1], id, ))