from dataclasses import dataclass
from enum import Enum

import subprocess
import os
from pathlib import Path
import platform
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from pdfa import PDFA, Trace


# === CONFIGURATION ===

# Root of FlexFringe-related files
FF_ROOT = Path("C:/users/blaze/Projects/FlexFringeEnsemble")  # Or Path("/home/user/FlexFringe") on Linux

# Executable name or path (can be .exe or just 'ensemble' on Linux)
ENSEMBLE_EXECUTABLE = "cmake-build-debug-visual-studio/flexfringe.exe" if platform.system() == "Windows" else "flexfringe"

# Full path to executable
ENSEMBLE_PATH = FF_ROOT / ENSEMBLE_EXECUTABLE

# Number of CPU cores for parallel tasks
CORES = 12


# === Ensemble runners ===

class ModelType(Enum):
    SINGLE = "single"
    ENSEMBLE = "ensemble"

class EnsMode(Enum):
    RANDOM = "random"
    GREEDY = "greedy"

@dataclass
class Model:
    model_type: ModelType
    model_name: str
    nrestimators: int = 10
    random_factor: float = 0.0
    ens_mode: EnsMode = EnsMode.RANDOM

    @staticmethod
    def Single(model_name: str) -> "Model":
        return Model(
            model_type=ModelType.SINGLE,
            model_name=model_name,
            nrestimators=1
        )

    @staticmethod
    def Random(model_name: str, nrestimators: int = 10) -> "Model":
        return Model(
            model_type=ModelType.ENSEMBLE,
            ens_mode=EnsMode.RANDOM,
            model_name=model_name,
            nrestimators=nrestimators
        )

    @staticmethod
    def Greedy(model_name: str, nrestimators: int = 10, rand: float = 1.0) -> "Model":
        return Model(
            model_type=ModelType.ENSEMBLE,
            ens_mode=EnsMode.GREEDY,
            model_name=model_name,
            nrestimators=nrestimators,
            random_factor=rand
        )

    def into_run(self: "Model", run_name: str|None = None, trainset_name: str|None = None) -> "TrainRun":
        return TrainRun(
            model=self,
            run_name=f"{self.model_name}_{run_name}" if run_name else self.model_name,
            trainset_name=trainset_name if trainset_name else self.model_name
        )

@dataclass
class TrainRun:
    model: Model
    run_name: str
    trainset_name: str = ""

    def into_test(self: "TrainRun", testset_name: str|None = None) -> "TestRun":
        return TestRun(
            model=self.model,
            run_name=self.run_name,
            testset_name=testset_name if testset_name else self.model.model_name
        )

@dataclass
class TestRun:
    model: Model
    run_name: str
    testset_name: str = ""

    @staticmethod
    def from_train_runs(train_runs: list[TrainRun], testset_name: str|None = None) -> list["TestRun"]:
        return [run.into_test(testset_name) for run in train_runs]


def run_flexfringe(args, cwd=None) -> str:
    """Run the ensemble program with given arguments and return output."""
    cmd = [str(ENSEMBLE_PATH)] + args
    # print("Running command:", cmd)
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


def gen_model(model: ModelType, training_data_path: str, output_path: str, ini_path: str = "ini/alergia.ini", nrestimators: int = 10, random_factor: float = 0.0, ens_mode: EnsMode = EnsMode.RANDOM) -> str:
    args = [
        "--mode", "gen_" + model.value if model == ModelType.ENSEMBLE else "batch",
        "--ini", str(FF_ROOT / ini_path),
        "--outputfile", str(FF_ROOT / output_path),
        "--nrestimators", str(nrestimators),
        "--random", str(random_factor),
        "--ensmode", ens_mode.value,
        str(FF_ROOT / training_data_path)
    ]
    return run_flexfringe(args)


def pred_model(model: ModelType, apta_path: str, solution_path: str, test_data_path: str, ini_path: str = "ini/alergia.ini", nrestimators: int = 10) -> str:
    args = [
        "--mode", "pred_" + model.value,
        "--ini", str(FF_ROOT / ini_path),
        "--aptafile", str(FF_ROOT / apta_path),
        "--solution", str(FF_ROOT / solution_path),
        "--nrestimators", str(nrestimators),
        str(FF_ROOT / test_data_path)
    ]
    return run_flexfringe(args)


def inter_model(model_paths: list[str], sample_size: int, train_data_path: str, nrestimators: int) -> str:
    absolute_paths = [str(FF_ROOT / model_path) for model_path in model_paths]
    args = [
        "--mode", "inter_model",
        "--ini", str(FF_ROOT / "ini/alergia.ini"), # Just to fill required args
        "--samsize", str(sample_size),
        "--aptafile", str.join(";", absolute_paths),
        "--nrestimators", str(nrestimators),
        str(FF_ROOT / train_data_path)
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
    raise ValueError("Perplexity score not found in output", run_output)


def extract_diff_array(run_output) -> list[float]:
    lines = run_output.splitlines()
    # Iterate from the end to be faster
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line.startswith("diffs:"):
            diffs_str = line.removeprefix("diffs:").strip()
            values_as_str = str.split(diffs_str, ";")[:-1] # Leave empty string at the end
            return [float(value) for value in values_as_str]
    raise ValueError("Diff score not found in output", run_output)


# === Evaluation / Experiments ===

@dataclass(frozen=True)
class TestConfig:
    name: str
    dir: str
    train_suffix: str
    model_suffix: str
    solution_suffix: str
    test_suffix: str

def create_standard_config(test_name: str, dir: str|None = None) -> TestConfig:
    return TestConfig(
        name = test_name,
        dir = dir if dir else test_name,
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


def train_model(config: TestConfig, run: TrainRun) -> None:
    # Ensure the out directory exists
    model = run.model
    os.makedirs(f"{FF_ROOT}/data/{config.dir}/out_{config.name}", exist_ok = True)
    gen_model(
        model=model.model_type,
        training_data_path=     f"data/{config.dir}/{run.trainset_name}{config.train_suffix}",
        output_path=            f"data/{config.dir}/out_{config.name}/{run.run_name}{config.model_suffix}",
        nrestimators=model.nrestimators,
        random_factor=model.random_factor,
        ens_mode=model.ens_mode
    )


def test_model(config: TestConfig, run: TestRun) -> float:
    model = run.model
    output =  pred_model(
        model=model.model_type,
        apta_path =             f"data/{config.dir}/out_{config.name}/{run.run_name}{config.model_suffix}", 
        solution_path =         f"data/{config.dir}/{run.testset_name}{config.solution_suffix}", 
        test_data_path =        f"data/{config.dir}/{run.testset_name}{config.test_suffix}",
        nrestimators=model.nrestimators
    )
    return extract_perplexity_score(output)


def diff_models(config: TestConfig, trainset_name: str, model_names: list[str], nrestimators: int, sample_size: int = 200) -> list[float]:
    model_paths = [f"data/{config.dir}/out_{config.name}/{model_name}{config.model_suffix}" for model_name in model_names]
    output = inter_model(
        model_paths=model_paths,
        sample_size=sample_size,
        train_data_path=     f"data/{config.dir}/{trainset_name}{config.train_suffix}",
        nrestimators=nrestimators
    )
    return extract_diff_array(output)


def diff_model(config: TestConfig, trainset_name: str, model_name: str, nrestimators: int, sample_size: int = 200) -> list[float]:
    model_path = f"data/{config.dir}/out_{config.name}/{model_name}{config.model_suffix}"
    output = inter_model(
        model_paths=[model_path],
        sample_size=sample_size,
        train_data_path=     f"data/{config.dir}/{trainset_name}{config.train_suffix}",
        nrestimators=nrestimators
    )
    return extract_diff_array(output)


def train_models_batch(
    config: TestConfig,
    runs: list[TrainRun],
    number_of_cores: int = CORES
) -> None:
    """
    Runs flexfringe in gen_* mode for each (model_type, model_name, testset_name)
    in parallel, and returns a mapping from that tuple to the extracted perplexity.
    """
    with ThreadPoolExecutor(max_workers=number_of_cores) as exe:
        future_to_run = {
            exe.submit(train_model, config, run): id for id, run in enumerate(runs)
        }
        for fut in as_completed(future_to_run):
            key = future_to_run[fut]
            try:
                fut.result()
            except Exception as e:
                raise RuntimeError(f"Training failed for {runs[key]}: {e}")



def test_models_batch(
    config: TestConfig,
    runs: list[TestRun],
    number_of_cores: int = CORES
):
    """
    Runs flexfringe in gen_* mode for each (model_type, model_name, testset_name)
    in parallel, and returns a mapping from that tuple to the extracted perplexity.
    """
    results: list[float] = [-1.0 for _ in runs]

    with ThreadPoolExecutor(max_workers=number_of_cores) as exe:
        future_to_run = {
            exe.submit(test_model, config, run): id for id, run in enumerate(runs)
        }
        for fut in as_completed(future_to_run):
            key = future_to_run[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                raise RuntimeError(f"Testing failed for {runs[key]}: {e}")
    return results


def diff_models_batch(
    config: TestConfig,
    model_names: list[str],
    train_name: str,
    nrestimators: int,
    sample_size: int = 200,
    number_of_cores: int = CORES
):
    results: list[list[float]] = [[] for _ in model_names]

    with ThreadPoolExecutor(max_workers=number_of_cores) as exe:
        future_to_run = {
            exe.submit(diff_model, config, train_name, model_name, nrestimators, sample_size): id for id, model_name in enumerate(model_names)
        }
        for fut in as_completed(future_to_run):
            key = future_to_run[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                raise RuntimeError(f"Diff failed for {model_names[key]}: {e}")
    return results


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