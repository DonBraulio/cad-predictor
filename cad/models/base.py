import torch
import numpy as np
import typing as T

from pathlib import Path
from importlib import import_module
from abc import abstractmethod, ABC

Array = T.TypeVar("Array", bound=T.Union[np.array, torch.tensor])
FilePath = T.TypeVar("FilePath", bound=T.Union[str, Path])
Predictor = T.TypeVar("Predictor", bound="CADPredictor")


class CADPredictor(ABC):
    def __init__(self, params_file: FilePath):
        self.featurizer = None
        self.load_from_params(params_file)

    @staticmethod
    def create_instance(model_path: str, params_file: FilePath) -> Predictor:
        # Find class and read params
        class_name = model_path.split(".")[-1]
        model_path = ".".join(model_path.split(".")[:-1])

        # Import and instantiate class
        Klass = getattr(import_module(model_path), class_name)
        return Klass(params_file)

    @abstractmethod
    def load_from_params(self, params_file: FilePath) -> None:
        """
        Read the params file and assign whatever variables are
        needed to initialize this model instance
        """

    @abstractmethod
    def predict(
        self, audio: Array, hop_s: float
    ) -> T.Tuple[np.array, np.array, np.array]:
        """
        Returns 3 arrays: t_preds, y_pred_labels, y_pred_scores
        Note: hop_s not necessarily matches t_preds' step, it
              will be a smaller interval that fits an integer
              number of feature frames
        """
