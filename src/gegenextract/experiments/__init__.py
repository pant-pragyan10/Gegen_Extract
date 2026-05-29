from abc import ABC, abstractmethod
from typing import Dict, Optional
from ..schemas import ExperimentRun


class ExperimentTracker(ABC):
    @abstractmethod
    def start_run(self, experiment_id: str, config: Dict) -> ExperimentRun:
        pass

    @abstractmethod
    def end_run(self, run_id: str, metrics: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def log_metric(self, run_id: str, key: str, value: float) -> None:
        pass


class SQLiteExperimentTracker(ExperimentTracker):
    def __init__(self, db_url: str):
        self.db_url = db_url
        # Placeholder: initialize DB connections and tables

    def start_run(self, experiment_id: str, config: Dict) -> ExperimentRun:
        # create and return an ExperimentRun object (persistence to be implemented)
        run = ExperimentRun(id=experiment_id + "_run", experiment_id=experiment_id, config=config)
        return run

    def end_run(self, run_id: str, metrics: Dict[str, float]) -> None:
        # persist run end and metrics
        return None

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        # append metric to run history
        return None
