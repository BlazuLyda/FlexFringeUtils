from dataclasses import dataclass, field
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

class VoteStrat(Enum):
    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    RANDOM = "random"
    PRECOMPUTED = "precomputed"

def add_suffix(base: str, suffix_1: str|None, suffix_2: str|None = None) -> str:
    if suffix_1 is not None and suffix_1 != "":
        return base + "_" + suffix_1
    if suffix_2 is not None and suffix_2 != "":
        return base + "_" + suffix_2
    return base

@dataclass
class Model:
    model_type: ModelType
    model_name: str
    model_file: str
    nrestimators: int = 10
    random_factor: float = 0.0
    ens_mode: EnsMode = EnsMode.RANDOM
    vote_strat: VoteStrat = VoteStrat.UNIFORM

    @staticmethod
    def Single(model_name: str, model_file: str|None = None) -> "Model":
        return Model(
            model_type=ModelType.SINGLE,
            model_name=model_name,
            model_file= model_file if model_file else model_name,
            nrestimators=1
        )

    @staticmethod
    def Random(model_name: str, model_file: str|None = None, nrestimators: int = 10) -> "Model":
        return Model(
            model_type=ModelType.ENSEMBLE,
            ens_mode=EnsMode.RANDOM,
            model_name=model_name,
            model_file= model_file if model_file else model_name,
            nrestimators=nrestimators
        )

    @staticmethod
    def Greedy(model_name: str, model_file: str|None = None, nrestimators: int = 10, rand: float = 1.0, vote_strat: VoteStrat = VoteStrat.UNIFORM) -> "Model":
        return Model(
            model_type=ModelType.ENSEMBLE,
            ens_mode=EnsMode.GREEDY,
            model_name=model_name,
            nrestimators=nrestimators,
            model_file= model_file if model_file else model_name,
            random_factor=rand,
            vote_strat=vote_strat
        )

    @staticmethod
    def Weighted(model_name: str, model_file: str|None = None, nrestimators: int = 10) -> "Model":
        return Model(
            model_type=ModelType.ENSEMBLE,
            ens_mode=EnsMode.GREEDY,
            model_name=model_name,
            nrestimators=nrestimators,
            model_file= model_file if model_file else model_name,
            vote_strat=VoteStrat.WEIGHTED
        )

    def into_run(
            self: "Model", 
            run_name: str|None = None,
            trainset_name: str|None = None, 
            continue_work: bool = False, 
            first_id: int = 0, 
            nrestimators: int|None = None
        ) -> "TrainRun":
        return TrainRun(
            model = self,
            run_name = add_suffix(self.model_file, run_name),
            trainset_name = trainset_name if trainset_name else self.model_file,
            continue_work=continue_work,
            first_id=first_id,
            nrestimators = nrestimators if nrestimators else self.nrestimators
        )

    def split_into_parallel_runs(
            self: "Model", 
            run_name: str|None = None, 
            trainset_name: str|None = None, 
            continue_work: bool = False, 
            parts: int|None = None
        ) -> list["TrainRun"]:

        tot_models = self.nrestimators
        if not parts or parts < tot_models:
            parts = tot_models

        unit_size = tot_models // parts
        last_unit_size = unit_size + tot_models % unit_size
        starting_ids = [i * unit_size for i in range(parts)]
        
        units = [
            self.into_run(run_name, trainset_name=trainset_name, continue_work=continue_work, first_id=i, nrestimators=unit_size)
            for i in starting_ids[:-1]
        ]
        units.append(
            self.into_run(run_name, trainset_name=trainset_name, continue_work=continue_work, first_id=starting_ids[-1], nrestimators=last_unit_size)
        )
        return units

    def into_test(
            self: "Model", 
            run_name: str|None = None,
            run_file: str|None = None,
            testset_name: str|None = None, 
            vote_strat: VoteStrat|None = None, 
            weights: list[float] = [], 
            ens_models: list[int] = [],
            nrestimators: int|None = None
        ) -> "TestRun":
        return TestRun(
            model = self,
            run_name= run_name if run_name else self.model_name,
            run_file= add_suffix(self.model_file, run_file, run_name),
            testset_name = testset_name if testset_name else self.model_file,
            vote_strat = vote_strat if vote_strat else self.vote_strat,
            weights=weights,
            ens_models=ens_models,
            nrestimators=nrestimators if nrestimators else self.nrestimators
        )

@dataclass
class TrainRun:
    model: Model
    run_name: str
    nrestimators: int
    trainset_name: str = ""
    first_id: int = 0
    continue_work: bool = False

    def into_test(self: "TrainRun", testset_name: str|None = None, run_name: str|None = None, run_file: str|None = None) -> "TestRun":
        return TestRun(
            model=self.model,
            run_name= run_name if run_name else self.run_name,
            run_file= add_suffix(self.run_name, run_file, run_name),
            testset_name=testset_name if testset_name else self.model.model_name,
            nrestimators=self.nrestimators
        )

@dataclass
class TestRun:
    model: Model
    run_name: str
    run_file: str
    nrestimators: int
    testset_name: str = ""
    vote_strat: VoteStrat = VoteStrat.UNIFORM
    weights: list[float] = field(default_factory=list)
    ens_models: list[int] = field(default_factory=list)

    @staticmethod
    def from_train_runs(train_runs: list[TrainRun], testset_name: str|None = None) -> list["TestRun"]:
        return [run.into_test(testset_name=testset_name) for run in train_runs]

    def get_output_file(self, config: "TestConfig") -> str:
        return f"{str(FF_ROOT)}/data/{config.dir}/out_{config.name}/{self.run_name}.result"


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
        # print(result.stdout, result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running command:", cmd)
        print(e.stdout, e.stderr)
        raise


def gen_model(
        model: ModelType, 
        training_data_path: str, 
        output_path: str, 
        ini_path: str = "ini/alergia.ini", 
        nrestimators: int = 10, 
        random_factor: float = 0.0, 
        ens_mode: EnsMode = EnsMode.RANDOM,
        sample_size: int = 200,
        first_id: int = 0,
        continue_work: bool = False,
    ) -> str:
    args = [
        "--mode", "gen_" + model.value if model == ModelType.ENSEMBLE else "batch",
        "--ini", str(FF_ROOT / ini_path),
        "--outputfile", str(FF_ROOT / output_path),
        "--nrestimators", str(nrestimators),
        "--random", str(random_factor),
        "--ensmode", ens_mode.value,
        "--samsize", str(sample_size),
        "--first-id", str(first_id),
        "--continue-work", "1" if continue_work else "0",
        str(FF_ROOT / training_data_path)
    ]
    return run_flexfringe(args)


def pred_model(
        model: ModelType, 
        apta_path: str, 
        solution_path: str|None, 
        test_data_path: str, 
        output_path: str, 
        ini_path: str = "ini/alergia.ini", 
        nrestimators: int = 10, 
        vote_strat: VoteStrat = VoteStrat.UNIFORM,
        weights: list[float] = [],
        ens_models: list[int] = []
    ) -> str:

    optional_args: list[str] = []
    if len(weights) > 0:
        optional_args.append("--weights")
        optional_args.append(str.join(";", [str(w) for w in weights]))
    if len(ens_models) > 0:
        optional_args.append("--ensmodels")
        optional_args.append(str.join(";", [str(w) for w in ens_models]))
        nrestimators = len(ens_models)
    if solution_path != None:
        optional_args.append("--solution")
        optional_args.append(str(FF_ROOT / solution_path))
    
    args = [
        "--mode", "pred_" + model.value,
        "--ini", str(FF_ROOT / ini_path),
        "--outputfile", str(FF_ROOT / output_path),
        "--aptafile", str(FF_ROOT / apta_path),
        "--nrestimators", str(nrestimators),
        "--votestrat", vote_strat.value,
        *optional_args,
        str(FF_ROOT / test_data_path)
    ]
    return run_flexfringe(args)


def inter_model(model_paths: list[str], sample_size: int, train_data_path: str, nrestimators: int, vote_strat: VoteStrat) -> str:
    absolute_paths = [str(FF_ROOT / model_path) for model_path in model_paths]
    args = [
        "--mode", "inter_model",
        "--ini", str(FF_ROOT / "ini/alergia.ini"), # Just to fill required args
        "--samsize", str(sample_size),
        "--aptafile", str.join(";", absolute_paths),
        "--nrestimators", str(nrestimators),
        "--votestrat", vote_strat.value,
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


def extract_float_array(run_output, marker = "diffs:", required = True) -> list[float]:
    lines = run_output.splitlines()
    # Iterate from the end to be faster
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line.startswith(marker):
            data_str = line.removeprefix(marker).strip()
            values_array = str.split(data_str, ";")[:-1] # Leave empty string at the end
            return [float(value) for value in values_array]
    if required:
        raise ValueError(f"Marker {marker} not found int the output", run_output)
    return []


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
        training_data_path=     f"data/{config.dir}/{run.trainset_name}{config.train_suffix}",
        output_path=            f"data/{config.dir}/out_{config.name}/{run.run_name}{config.model_suffix}",
        model=model.model_type,
        nrestimators=run.nrestimators,
        random_factor=model.random_factor,
        ens_mode=model.ens_mode,
        first_id=run.first_id,
        continue_work=run.continue_work
    )


def test_model(config: TestConfig, run: TestRun) -> float:
    # Just return the score, skip weights
    output = test_model_only_run(config, run)
    return extract_perplexity_score(output)

def test_model_no_solution(config: TestConfig, run: TestRun):
    test_model_only_run(config, run, with_solution=False)

def test_model_with_weights(config: TestConfig, run: TestRun) -> tuple[float, list[float]]:
    # Just return the score, skip weights
    output = test_model_only_run(config, run)
    # If Random, we want to know the weights used for the testing to compute IMV
    return extract_perplexity_score(output), extract_float_array(output, marker="weights:", required=True)


def test_model_only_run(config: TestConfig, run: TestRun, with_solution: bool = True) -> str:
    model = run.model
    return pred_model(
        apta_path =             f"data/{config.dir}/out_{config.name}/{run.run_file}{config.model_suffix}", 
        output_path =           f"data/{config.dir}/out_{config.name}/{run.run_name}.result",
        solution_path =         f"data/{config.dir}/{run.testset_name}{config.solution_suffix}" if with_solution else None, 
        test_data_path =        f"data/{config.dir}/{run.testset_name}{config.test_suffix}",
        model=model.model_type,
        nrestimators=run.nrestimators,
        vote_strat=run.vote_strat,
        weights=run.weights,
        ens_models=run.ens_models
    )


def diff_models(config: TestConfig, trainset_name: str, model_names: list[str], nrestimators: int, sample_size: int = 200) -> list[float]:
    model_paths = [f"data/{config.dir}/out_{config.name}/{model_name}{config.model_suffix}" for model_name in model_names]
    output = inter_model(
        model_paths=model_paths,
        sample_size=sample_size,
        train_data_path=     f"data/{config.dir}/{trainset_name}{config.train_suffix}",
        nrestimators=nrestimators,
        vote_strat=VoteStrat.WEIGHTED
    )
    return extract_float_array(output, marker="diffs:")


def diff_model(config: TestConfig, run: TrainRun, sample_size: int = 200) -> list[float]:
    model_path = f"data/{config.dir}/out_{config.name}/{run.run_name}{config.model_suffix}"
    output = inter_model(
        model_paths=[model_path],
        sample_size=sample_size,
        train_data_path=     f"data/{config.dir}/{run.trainset_name}{config.train_suffix}",
        nrestimators=run.nrestimators,
        vote_strat=run.model.vote_strat
    )
    return extract_float_array(output, marker="diffs:")


def train_models_batch(
    config: TestConfig,
    runs: list[TrainRun],
    number_of_cores: int = CORES
) -> None:
    """
    Runs flexfringe in gen_* mode for each (model_type, model_name, testset_name)
    in parallel, prioritizing models with more estimators.
    """

    # Attach original indices before sorting
    # indexed_runs = list(enumerate(runs))
    
    # Sort by nrestimators descending, so longest runs go first
    # indexed_runs.sort(key=lambda x: x[1].model.nrestimators, reverse=True)

    with ThreadPoolExecutor(max_workers=number_of_cores) as exe:
        future_to_run = {
            exe.submit(train_model, config, run): id for id, run in enumerate(runs[::-1]) # Usually the bigest models are at the end of the array
            # exe.submit(train_model, config, run): original_id
            # for original_id, run in indexed_runs
        }
        for fut in as_completed(future_to_run):
            key = future_to_run[fut]
            try:
                fut.result()
            except Exception as e:
                raise RuntimeError(f"Training failed for {runs[key]}: {e}")



def test_models_with_weights_batch(
    config: TestConfig,
    runs: list[TestRun],
    number_of_cores: int = CORES
):
    results: list[tuple[float, list[float]]] = [(-1.0, []) for _ in runs]

    with ThreadPoolExecutor(max_workers=number_of_cores) as exe:
        future_to_run = {
            exe.submit(test_model_with_weights, config, run): id for id, run in enumerate(runs)
        }
        for fut in as_completed(future_to_run):
            key = future_to_run[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                raise RuntimeError(f"Testing failed for {runs[key]}: {e}")
    return results


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


def test_models_raw_predictions(
    config: TestConfig,
    runs: list[TestRun],
    number_of_cores: int = CORES
):
    with ThreadPoolExecutor(max_workers=number_of_cores) as exe:
        future_to_run = {
            exe.submit(test_model_no_solution, config, run): id for id, run in enumerate(runs)
        }
        for fut in as_completed(future_to_run):
            key = future_to_run[fut]
            try:
                fut.result()
            except Exception as e:
                raise RuntimeError(f"Testing failed for {runs[key]}: {e}")


def diff_models_batch(
    config: TestConfig,
    runs: list[TrainRun],
    sample_size: int = 200,
    number_of_cores: int = CORES
):
    results: list[list[float]] = [[] for _ in runs]

    with ThreadPoolExecutor(max_workers=number_of_cores) as exe:
        future_to_run = {
            exe.submit(diff_model, config, run, sample_size): id for id, run in enumerate(runs)
        }
        for fut in as_completed(future_to_run):
            key = future_to_run[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                raise RuntimeError(f"Diff failed for {runs[key]}: {e}")
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