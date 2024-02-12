from pathlib import Path
import json

from collections.abc import Mapping
from typing import Tuple

def get_latest_experiment(
        experiment=None, experiment_path=None, 
        parent_path='data/experiments/one_shot_plan') \
        -> Tuple[Mapping, Path]:
    parent_path = Path(parent_path)
    if experiment is not None:
        experiment_path = parent_path / experiment
    if experiment_path is None:
        experiment_path = list(parent_path.iterdir())[-1]
        experiment_path = Path(experiment_path)

    exp_metadata_path = experiment_path / 'meta.json'
    exp_metadata = json.loads(exp_metadata_path.read_text())

    exp_data = experiment_path / 'data'
    return exp_metadata, exp_data
