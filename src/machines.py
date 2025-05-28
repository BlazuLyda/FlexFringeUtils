from typing import Callable
import random
from pdfa import PDFAState, PDFA
from export import export_pdfa_dot, export_pdfa_json, import_pdfa_json

def simple_machine() -> PDFA:
    # Build a sample PDFA manually
    s0 = PDFAState(id=0, final_frequency=1)
    s1 = PDFAState(id=1, final_frequency=3)
    s2 = PDFAState(id=2, final_frequency=6)

    s0.add_transition(0, 1, 4)
    s1.add_transition(1, 2, 6)
    s2.add_transition(0, 0, 2)

    return PDFA(name="simple_machine", states=[s0, s1, s2], start_state=0, alphabet_size=2)


# Machine that accepts strings ending with a "0", for example "00", "110", "01110"
def parity_machine() -> PDFA:
    pdfa = PDFA(name="parity", alphabet_size=2)

    # Add states
    pdfa.add_sink(0)
    pdfa.add_state(1)

    # Set frequencies
    pdfa.states[0].final_frequency = 1

    # Define transitions: node_from, symbol, node_to, frequency 
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

    # Define transitions: node_from, symbol, node_to, frequency 
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


def random_pdfa(
    name: str = "random",
    min_states: int = 5,
    max_states: int = 75,
    min_alpha: int = 4,
    max_alpha: int = 24,
    sym_sparsity_range=(0.2, 0.8),
    trans_sparsity_range=(0.0, 0.2),
    frequency_range=(1, 5)
) -> PDFA:
    """
    Generate a random Probabilistic DFA per PAutomaC spec (simplified):
      - |Q| uniformly in [min_states, max_states]
      - |Σ| uniformly in [min_alpha, max_alpha]
      - Symbol-sparsity: pick between 20-80% of state-symbol pairs
      - For each selected (q,a), add exactly one transition to random target
      - Transition-sparsity: add 0-20% more transitions chosen similarly
      - All edge frequencies set to 1; single sink state is accepting with freq=1.
    """
    # 1) Pick sizes
    n_states = random.randint(min_states, max_states)
    alpha    = random.randint(min_alpha, max_alpha)
    freq = lambda : random.randint(frequency_range[0], frequency_range[1])

    # 2) Create empty PDFA
    pdfa = PDFA(name=name, alphabet_size=alpha)
    for q in range(n_states):
        pdfa.add_state(q)
    sink = n_states
    pdfa.add_sink(sink)
    pdfa.states[sink].final_frequency = freq()  # make sink accepting
    pdfa.start_state = 0

    # 3) All possible (state, symbol) pairs
    all_pairs = [(q, a) for q in range(n_states) for a in range(alpha)]

    # 4) Symbol sparsity: sample some fraction of these pairs
    frac_sym = random.uniform(*sym_sparsity_range)
    k_sym    = int(len(all_pairs) * frac_sym)
    sym_pairs = random.sample(all_pairs, k_sym)

    for (q, a) in sym_pairs:
        tgt = random.randint(0, sink)
        pdfa.add_edge(q, a, tgt, frequency=freq())

    # 5) Transition sparsity: sample additional pairs from the remainder
    remainder = [p for p in all_pairs if p not in sym_pairs]
    frac_trans = random.uniform(*trans_sparsity_range)
    k_trans    = int(len(remainder) * frac_trans)
    extra_pairs = random.sample(remainder, k_trans)

    for (q, a) in extra_pairs:
        tgt = random.randint(0, sink)
        pdfa.add_edge(q, a, tgt, frequency=freq())

    # 6) Add transitions to states with 0 edge count
    not_dead_states = set([sink])
    for q in range(0, sink):
        if pdfa.states[q].total_frequency() > 0:
            not_dead_states.add(q)
    print("Not dead states: ", not_dead_states)

    for (q, a) in all_pairs:
        if q in not_dead_states:
            continue
        # Add transition to somewhere else than the state itself
        tgt = q
        while tgt == q:
            tgt = random.randint(0, sink)
        pdfa.add_edge(q, a, tgt, frequency=freq())
        not_dead_states.add(q)

    return pdfa


MachineMaker = Callable[[], PDFA]

MACHINE: MachineMaker = reber
TRAIN_SIZE = 10000
TEST_SIZE = 1000

if __name__ == "__main__":

    pdfa = random_pdfa(
        min_states = 20, 
        max_states=25, 
        min_alpha=12, 
        max_alpha=14, 
        sym_sparsity_range=(0.2, 0.3), 
        trans_sparsity_range=(0.0, 0.1)
    )
    export_pdfa_dot(pdfa, f"models/{pdfa.name}")
    export_pdfa_json(pdfa) 
