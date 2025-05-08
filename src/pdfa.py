import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PDFATransition:
    symbol: int
    target: int
    frequency: int


class PDFAState:

    id: int
    final_frequency: int
    sink: Optional[int]
    transitions: Dict[int, PDFATransition]

    def __init__(self, id: int, final_frequency: int = 0, sink: Optional[int] = None) -> None:
        self.id = id
        self.final_frequency = final_frequency
        self.sink = sink
        self.transitions = {}


    def add_transition(self, symbol: int, target: int, frequency: int) -> None:
        self.transitions[symbol] = PDFATransition(symbol, target, frequency)


    def total_frequency(self) -> int:
        return self.final_frequency + sum(t.frequency for t in self.transitions.values())


    """Returns either (symbol, next_id) if transition or (sink_type, None) if reached sink"""
    def choose_next(self) -> Tuple[int, Optional[int]]:

        total = self.total_frequency()
        if total == 0:
            raise ValueError("Ended up in state with 0 total count")

        r = random.randint(1, total)
        current = self.final_frequency
        if r <= current:
            if self.sink is None:
                raise ValueError(f"Node {self.id} with non zero final frequency is not a sink")
            return self.sink, None # Accepted (final state)

        for symbol, t in self.transitions.items():
            current += t.frequency
            if r <= current:
                return symbol, t.target 
        raise ValueError("Something went very wrong")


Trace = Tuple[int, List[int]]

class PDFA:

    name: str
    states: Dict[int, PDFAState]
    start_state: int
    alphabet_size: int

    def __init__(self, name:str, alphabet_size: int, states: List[PDFAState] = [], start_state: int = -1) -> None:
        self.name = name
        self.alphabet_size = alphabet_size
        self.states = {state.id: state for state in states}
        self.start_state = start_state


    def add_state(self, id: int) -> PDFAState:
        self.states[id] = PDFAState(id)
        return self.states[id]

    def add_sink(self, id: int, sink: int) -> PDFAState:
        self.states[id] = PDFAState(id, sink=sink)
        return self.states[id]


    def add_edge(self, origin: int, label: int, target: int, frequency: int) -> None:
        self.states[origin].add_transition(label, target, frequency)


    def generate_trace(self) -> Trace:
        trace: List[int] = []
        final_label: int = 0
        state: PDFAState = self.states[self.start_state]

        while True:
            symbol, next_state_id = state.choose_next()
            if next_state_id is None:
                final_label = symbol
                break  # Accepted
            trace.append(symbol)
            state = self.states[next_state_id]
        return final_label, trace


    def generate_dataset(self, num_traces: int) -> List[Trace]:
        return [self.generate_trace() for _ in range(num_traces)]


    def write_dataset(self, num_traces: int, out_path: str) -> None:
        traces = self.generate_dataset(num_traces)
        with open(out_path, "w") as f:
            f.write(f"{num_traces} {self.alphabet_size}\n")
            for label, trace in traces:
                f.write(f"{label} {len(trace)} {' '.join(map(str, trace))}\n")
