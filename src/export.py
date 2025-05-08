from graphviz import Digraph

from pdfa import PDFA

def export_pdfa_dot(pdfa: PDFA, output_file: str) -> None:

    dot = Digraph(name="PDFA", format="png")
    dot.attr(rankdir="LR", shape="circle")

    # Add nodes
    for state_id, state in pdfa.states.items():
        label = f"{state_id}\\n#{state.final_frequency}/{state.total_frequency()}"
        if state.final_frequency > 0:
            label += f" fin: {state.sink}"
        shape = "doublecircle" if state.final_frequency > 0 else "circle"
        style = "filled" if state == pdfa.start_state else "solid"
        fillcolor = "lightblue" if state == pdfa.start_state else "white"
        dot.node(str(state_id), label=label, shape=shape, style=style, fillcolor=fillcolor)

    # Add edges
    for state_id, state in pdfa.states.items():
        for t in state.transitions.values():
            label = f"{t.symbol} #{t.frequency}/{state.total_frequency()}"
            dot.edge(str(state_id), str(t.target), label=label)

    # Add initial dummy node pointing to start state
    dot.node("init", shape="point")
    dot.edge("init", str(pdfa.start_state))

    # Write to file
    print(dot)
    dot.render(output_file, cleanup=True)
