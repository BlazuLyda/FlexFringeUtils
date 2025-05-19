from dataclasses import dataclass
from enum import Enum

import subprocess
import os
from pathlib import Path
import platform
import numpy as np

from pdfa import PDFA, Trace


# === CONFIGURATION ===

# Root of FlexFringe-related files
FF_ROOT = Path("C:/users/blaze/Projects/FlexFringeEnsemble")  # Or Path("/home/user/FlexFringe") on Linux

# Executable name or path (can be .exe or just 'ensemble' on Linux)
ENSEMBLE_EXECUTABLE = "cmake-build-debug-visual-studio/flexfringe.exe" if platform.system() == "Windows" else "flexfringe"

# Full path to executable
ENSEMBLE_PATH = FF_ROOT / ENSEMBLE_EXECUTABLE


# === Ensemble runners ===

class Model(Enum):
    SINGLE = "single"
    ENSEMBLE = "ensemble"


def run_flexfringe(args, cwd=None) -> str:
    """Run the ensemble program with given arguments and return output."""
    cmd = [str(ENSEMBLE_PATH)] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            capture_output=True
        )
        # print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running command:", cmd)
        print(e.stdout, e.stderr)
        raise


def gen_model(model: Model, training_data_path: str, output_path: str, ini_path: str = "ini/alergia.ini", nrestimators: int = 10) -> str:
    args = [
        "--mode", "gen_" + model.value if model == Model.ENSEMBLE else "batch",
        "--ini", str(FF_ROOT / ini_path),
        "--outputfile", str(FF_ROOT / output_path),
        "--nrestimators", str(nrestimators),
        str(FF_ROOT / training_data_path)
    ]
    return run_flexfringe(args)


def pred_model(model: Model, apta_path: str, solution_path: str, test_data_path: str, ini_path: str = "ini/alergia.ini") -> str:
    args = [
        "--mode", "pred_" + model.value,
        "--ini", str(FF_ROOT / ini_path),
        "--aptafile", str(FF_ROOT / apta_path),
        "--solution", str(FF_ROOT / solution_path),
        str(FF_ROOT / test_data_path)
    ]
    return run_flexfringe(args)

def extract_perplexity_score(run_output) -> float:
    lines = run_output.splitlines()
    # Iterate from the end to be faster
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line.startswith("Final perplexity:"):
            perplexity_str = line.removeprefix("Final perplexity:").strip()
            return float(perplexity_str)
    raise ValueError("Perplexity score not found in output")


# === Evaluation / Experiments ===

@dataclass(frozen=True)
class TestConfig:
    name: str
    dir: str
    train_suffix: str
    model_suffix: str
    solution_suffix: str
    test_suffix: str

def create_standard_config(test_name: str) -> TestConfig:
    return TestConfig(
        name = test_name,
        dir = test_name,
        train_suffix = "_training.txt",
        model_suffix = "_training.txt",
        solution_suffix = "_solution.txt",
        test_suffix = "_test.txt"
    )

STANDARD_TEST = create_standard_config("test")

PAUTOMAC_TEST = TestConfig(
    name = "PAutomaC",
    dir = "PAutomaC-competition_sets",
    train_suffix = ".train.dat",
    model_suffix = ".train.dat.ff",
    solution_suffix = "_solution.txt",
    test_suffix = ".test.dat"
)

def train_model(config: TestConfig, model_type: Model, model_name: str, trainset_name: str = "", nrestimators: int = 10) -> None:
    if trainset_name == "":
        trainset_name = model_name
    # Ensure the out directory exists
    os.makedirs(f"{FF_ROOT}/data/{config.dir}/out", exist_ok = True)
    gen_model(
        model = model_type,
        training_data_path =    f"data/{config.dir}/{trainset_name}{config.train_suffix}",
        output_path =           f"data/{config.dir}/out/{model_name}{config.model_suffix}",
        nrestimators = nrestimators
    )

def test_model(config: TestConfig, model_type: Model, model_name: str, testset_name: str = "") -> float:
    if testset_name == "":
        testset_name = model_name
    output =  pred_model(
        model = model_type,
        apta_path =             f"data/{config.dir}/out/{model_name}{config.model_suffix}", 
        solution_path =         f"data/{config.dir}/{testset_name}{config.solution_suffix}", 
        test_data_path =        f"data/{config.dir}/{testset_name}{config.test_suffix}"
    )
    return extract_perplexity_score(output)


# === Writing test/datasets ===

def write_testset(config: TestConfig, pdfa: PDFA, testset_name: str, size: int) -> None:
    pdfa.write_testset(
        test_size = size, 
        traces_out_path =       f"{FF_ROOT}/data/{config.dir}/{testset_name}{config.test_suffix}", 
        solutions_out_path =    f"{FF_ROOT}/data/{config.dir}/{testset_name}{config.solution_suffix}"
    )

def write_trainset(config: TestConfig, pdfa: PDFA, trainset_name: str, size: int, append_to: list[Trace] = []) -> list[Trace]:
    return pdfa.write_trainset(
        num_traces = size, 
        out_path =              f"{FF_ROOT}/data/{config.dir}/{trainset_name}{config.train_suffix}",
        append_to = append_to
    )