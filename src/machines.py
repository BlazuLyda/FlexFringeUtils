from typing import Callable
from pdfa import PDFAState, PDFA
from export import export_pdfa_dot

def simple_machine() -> PDFA:
    # Build a sample PDFA manually
    s0 = PDFAState(id=0, final_frequency=1)
    s1 = PDFAState(id=1, final_frequency=3)
    s2 = PDFAState(id=2, final_frequency=6)

    s0.add_transition(0, 1, 4)
    s1.add_transition(1, 2, 6)
    s2.add_transition(0, 0, 2)

    return PDFA(name="simple_machine", states=[s0, s1, s2], start_state=0, alphabet_size=2)


def machine_1() -> PDFA:
    pdfa = PDFA(name="machine_1", alphabet_size=2)

    # Add states
    for state_id in range(4):
        pdfa.add_state(state_id)

    # Set frequencies
    pdfa.states[0].final_frequency = 0
    pdfa.states[1].final_frequency = 0
    pdfa.states[2].final_frequency = 0
    pdfa.states[3].final_frequency = 10  # Accepting state

    # Define transitions
    pdfa.add_edge(0, 0, 1, 5)  # From 0 with label 0 to 1
    pdfa.add_edge(0, 1, 2, 5)  # From 0 with label 1 to 2
    pdfa.add_edge(1, 0, 3, 3)  # From 1 with label 0 to accepting state 3
    pdfa.add_edge(2, 1, 3, 3)  # From 2 with label 1 to accepting state 3

    # Initial state is 0
    pdfa.start_state = 0

    return pdfa


MachineMaker = Callable[[], PDFA]

MACHINE: MachineMaker = machine_1
NUM_TRACES = 50

if __name__ == "__main__":

    pdfa = MACHINE()
    pdfa.write_dataset(num_traces=NUM_TRACES, out_path=f"data/{pdfa.name}_training.txt")
    export_pdfa_dot(pdfa, f"data/{pdfa.name}")
