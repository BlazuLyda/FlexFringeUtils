from datetime import datetime
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PDFATransition:
    symbol: int
    target: int
    frequency: int

@dataclass(frozen=True)
class Choice:
    symbol: int
    target: Optional[int]
    prob: float


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
    def choose_next(self) -> Choice:

        total = self.total_frequency()
        if total == 0:
            raise ValueError("Ended up in state with 0 total count")

        r = random.randint(1, total)
        current = self.final_frequency
        if r <= current:
            if self.sink is None:
                raise ValueError(f"Node {self.id} with non zero final frequency is not a sink")
            return Choice(self.sink, None, self.final_frequency / total) # Accepted (final state)

        for symbol, t in self.transitions.items():
            current += t.frequency
            if r <= current:
                return Choice(symbol, t.target, t.frequency / total) 
        raise ValueError("Something went very wrong")


@dataclass(frozen=True)
class Trace:
    sink: int
    symbols: List[int] 
    prob: float

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

    def add_sink(self, id: int, sink: int = 1) -> PDFAState:
        self.states[id] = PDFAState(id, sink=sink)
        return self.states[id]


    def add_edge(self, origin: int, label: int, target: int, frequency: int) -> None:
        self.states[origin].add_transition(label, target, frequency)


    def generate_trace(self) -> Trace:

        symbols: List[int] = []
        sink: int = 0
        prob: float = 1
        state: PDFAState = self.states[self.start_state]

        while True:
            choice = state.choose_next()
            prob *= choice.prob
            if choice.target is None:
                sink = choice.symbol
                break  # Accepted
            symbols.append(choice.symbol)
            state = self.states[choice.target]
        return Trace(sink, symbols, prob)


    def generate_dataset(self, num_traces: int) -> List[Trace]:
        """
        Performs num_traces random walk through the pdfa generating traces. The relative frequencies of the traces
        reflect the real pdfa distribution. 
        """
        return [self.generate_trace() for _ in range(num_traces)]

    def generate_testset(self, num_traces: int) -> List[Trace]:
        """
        Generates num_traces unique traces. The probabilities of the traces divided by the sum of all probabilities
        in this set should reflect the real pdfa distribution. 
        """
        seen: set[tuple[int, ...]] = set()
        unique_traces: list[Trace] = []

        while len(unique_traces) < num_traces:
            trace = self.generate_trace()
            key = tuple(trace.symbols)
            if key not in seen:
                seen.add(key)
                unique_traces.append(trace)

        return unique_traces


    def write_trainset(self, num_traces: int, out_path: str, append_to: List[Trace] = []) -> List[Trace]:
        """
        Generates and writes a set of traces to a file in abbadingo format.
        Length of 'append_to' should be smaller equal than 'num_traces'.
        """
        traces: List[Trace] = append_to + self.generate_dataset(num_traces - len(append_to))
        # Write the traces
        with open(out_path, "w") as f:
            f.write(f"{num_traces} {self.alphabet_size}\n")
            for trace in traces:
                f.write(f"{trace.sink} {len(trace.symbols)} {' '.join(map(str, trace.symbols))}\n")
        return traces


    def write_testset(self, test_size: int, traces_out_path: str, solutions_out_path: str) -> List[Trace]:
        """
        Generates and writes a test set of traces in abbadingo format. Additionaly,
        writes the traces probabilities to solutions files.
        """
        traces = self.generate_testset(test_size)
        # Write the unique traces
        with open(traces_out_path, "w") as f:
            f.write(f"{test_size} {self.alphabet_size}\n")
            for trace in traces:
                f.write(f"{trace.sink} {len(trace.symbols)} {' '.join(map(str, trace.symbols))}\n")
        # Write the solutions
        with open(solutions_out_path, "w") as f:
            f.write(f"{test_size}\n")
            for trace in traces:
                f.write(f"{trace.prob}\n")
        return traces
