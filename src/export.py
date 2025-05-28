from graphviz import Digraph
import json

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


def export_pdfa_json(pdfa: PDFA, output_file: str = "") -> None:
    """
    Provide pdfa and optional output_file
    """
    if output_file == "":
        output_file = f"models/{pdfa.name}.json"

    data = {
        "name": pdfa.name,
        "alphabet_size": pdfa.alphabet_size,
        "start_state": pdfa.start_state,
        "states": [
            {
                "id": sid,
                "final_frequency": state.final_frequency,
                "transitions": [
                    {
                        "symbol": t.symbol,
                        "target": t.target,
                        "frequency": t.frequency
                    }
                    for t in state.transitions.values()
                ]
            }
            for sid, state in pdfa.states.items()
        ]
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def import_pdfa_json(pdfa_name: str = "", input_file: str = "") -> PDFA:
    """
    Provide either pdfa_name or input_file 
    """
    if input_file == "":
        input_file = f"models/{pdfa_name}.json"
    with open(input_file, "r") as f:
        data = json.load(f)

    pdfa = PDFA(name=data["name"], alphabet_size=data["alphabet_size"])
    
    # First, add all states
    for state in data["states"]:
        sid = state["id"]
        if state["final_frequency"] > 0:
            pdfa.add_sink(sid)
        else:
            pdfa.add_state(sid)
        pdfa.states[sid].final_frequency = state["final_frequency"]

    # Then, add all transitions
    for state in data["states"]:
        sid = state["id"]
        for t in state["transitions"]:
            pdfa.add_edge(sid, t["symbol"], t["target"], t["frequency"])

    pdfa.start_state = data["start_state"]
    return pdfa
