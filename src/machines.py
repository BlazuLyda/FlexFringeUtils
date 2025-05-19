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


def parity_machine() -> PDFA:
    pdfa = PDFA(name="parity", alphabet_size=2)

    # Add states
    pdfa.add_sink(0)
    pdfa.add_state(1)

    # Set frequencies
    pdfa.states[0].final_frequency = 1

    # Define transitions
    pdfa.add_edge(0, 0, 0, 1)
    pdfa.add_edge(0, 1, 1, 1)
    pdfa.add_edge(1, 0, 0, 1)
    pdfa.add_edge(1, 1, 1, 1)

    # Initial state is 0
    pdfa.start_state = 0

    return pdfa


# From "Learning Stochastic Regular Languages..."
def reber() -> PDFA:
    pdfa = PDFA(name="reber_grammar", alphabet_size=7)

    # Alphabet:
    B=0; T=1; P=2; S=3; X=4; E=5; V=6

    # Add states
    pdfa.add_state(0)
    pdfa.add_state(1)
    pdfa.add_state(2)
    pdfa.add_state(3)
    pdfa.add_state(4)
    pdfa.add_state(5)
    pdfa.add_state(6)
    pdfa.add_sink(7)
    pdfa.states[7].final_frequency = 1

    # Define transitions 
    pdfa.add_edge(0, B, 1, 10)

    pdfa.add_edge(1, T, 2, 5)
    pdfa.add_edge(1, P, 3, 5)

    pdfa.add_edge(2, S, 2, 6)
    pdfa.add_edge(2, X, 4, 4)
    
    pdfa.add_edge(3, T, 3, 7)
    pdfa.add_edge(3, V, 5, 3)

    pdfa.add_edge(4, X, 3, 5)
    pdfa.add_edge(4, S, 6, 5)

    pdfa.add_edge(5, P, 4, 5)
    pdfa.add_edge(5, V, 6, 5)

    pdfa.add_edge(6, E, 7, 10)

    # Initial state is 0
    pdfa.start_state = 0

    return pdfa


MachineMaker = Callable[[], PDFA]

MACHINE: MachineMaker = reber
TRAIN_SIZE = 10000
TEST_SIZE = 1000

if __name__ == "__main__":

    pdfa = MACHINE()
    pdfa.write_trainset(num_traces=TRAIN_SIZE, out_path=f"data/{pdfa.name}_training.txt")
    pdfa.write_testset(test_size=TEST_SIZE, traces_out_path=f"data/{pdfa.name}_test.txt", solutions_out_path=f"data/{pdfa.name}_solution.txt")
    export_pdfa_dot(pdfa, f"data/{pdfa.name}")
