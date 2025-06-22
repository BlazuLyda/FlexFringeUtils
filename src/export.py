from graphviz import Digraph
import json
import os
import csv
from typing import Dict, List
import statistics

from pdfa import PDFA

import numpy as np

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


class ResultsManager:

    def __init__(self, base_dir: str):
        self.base_dir = base_dir


    def get_dataset_dir(self, experiment_name: str, dataset_name: str) -> str:
        path = os.path.join(self.base_dir, experiment_name, dataset_name)
        os.makedirs(path, exist_ok=True)
        return path


    def write_properties(self, experiment_name: str, dataset_name: str, properties: Dict[str, str], suffix: str = "") -> None:
        path = self.get_dataset_dir(experiment_name, dataset_name)
        file_path = os.path.join(path, f"params{suffix}.txt")
        with open(file_path, 'w') as f:
            for key, value in properties.items():
                f.write(f"{key} = {value}\n")


    def write_run_results(
            self,
            experiment_name: str,
            dataset_name: str,
            models: List[str],
            results: List[List[float]],
            suffix: str = ""
        ) -> None:
        path = self.get_dataset_dir(experiment_name, dataset_name)
        file_path = os.path.join(path, f"results{suffix}.csv")
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(models)
            writer.writerows(results)


    def read_properties(self, experiment_name: str, dataset_name: str, suffix: str = "") -> Dict[str, str]:
        path = self.get_dataset_dir(experiment_name, dataset_name)
        file_path = os.path.join(path, f'params{suffix}.txt')
        properties: Dict[str, str] = {}

        if not os.path.exists(file_path):
            return properties  # or raise an error if preferable

        with open(file_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.split('=', 1)
                    properties[key.strip()] = value.strip()
        return properties


    def read_run_results(self, experiment_name: str, dataset_name: str, suffix: str = "") -> Dict[str, List[float]]:
        path = self.get_dataset_dir(experiment_name, dataset_name)
        file_path = os.path.join(path, f"results{suffix}.csv")
        aggregated: Dict[str, List[float]] = {}

        if not os.path.exists(file_path):
            return aggregated  # or raise an error if that's preferable

        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            headers = next(reader)  # First line: model names
            for model_id in headers:
                aggregated[model_id] = []

            for row in reader:
                for model_id, score in zip(headers, row):
                    aggregated[model_id].append(float(score))
        return aggregated


    def write_latex_table(self, experiment_name: str, table: str, table_name: str = "table"):

        table_output = os.path.join(self.base_dir, experiment_name, f"{table_name}.tex")
        with open(table_output, "w") as f:
            f.write(table)


    def generate_latex_table_for_experiment(
        self,
        experiment_name: str,
        datasets: List[str],
        models: List[str],
    ) -> str:
        table_lines = []

        # Header
        header = ["Dataset", "Train Size"] + ["{" + m.replace("_", " ") + "}" for m in models]
        col_format = ["l", "l"] + ["S[table-format=1.2(3)]" for _ in range(len(models))]
        col_format_str = "|" + "|".join(col_format) + "|"
        table_lines.append("\\begin{tabular}{" + col_format_str + "}")
        table_lines.append("\\hline")
        table_lines.append(" & ".join(header) + " \\\\")
        table_lines.append("\\hline")

        # Assuming `models[0]` is always the baseline, e.g., "SINGLE"
        baseline_model = models[0]

        for dataset in datasets:
            # Read params and results from the dataset directory
            params = self.read_properties(experiment_name, dataset)
            results = self.read_run_results(experiment_name, dataset)

            # Format row: dataset name, train_size, scores (mean ± std)
            row: list[str] = [params["dataset"].replace("_", " "), params["trainset_size"]]

            # Fetch baseline scores
            baseline_scores = results.get(baseline_model)
            if not baseline_scores:
                row.extend(["N/A"] * len(models))
                table_lines.append(" & ".join(row) + r" \\")
                continue

            # Add baseline raw score
            mean_baseline = np.mean(baseline_scores)
            row.append(f"{mean_baseline:.1f}")

            for model in models[1:]:
                scores = results.get(model)
                if scores:
                    ratios = [m / b for m, b in zip(scores, baseline_scores)]
                    mean = np.mean(ratios)
                    std = np.std(ratios)
                    row.append(f"{mean:.2f} \\pm {std:.2f}")
                else:
                    row.append("N/A")
            table_lines.append(str.join(" & ", row) + r" \\")

        table_lines.append("\\hline")
        table_lines.append("\\end{tabular}")

        table = str.join("\n", table_lines)
        table_output = os.path.join(self.base_dir, experiment_name, "table.tex")
        with open(table_output, "w") as f:
            f.write(table)
        return table





