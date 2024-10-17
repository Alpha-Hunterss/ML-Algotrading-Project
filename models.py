import wandb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing_extensions import runtime_checkable
from typing import Protocol, Any, Type
import inspect

@runtime_checkable
class TrainableModel(Protocol):
    """
    This is the base protocol on which each ML model in this repo should based.
    """
    def fit(self, X: Any, y: Any) -> None:
        ...

    def predict(self, X: Any) -> Any:
        ...

    def predict_proba(self, X: Any) -> Any:
        ...

def set_model_parameters(model_class, all_params):
    # Check if the model class has the `get_params` method
    if hasattr(model_class(), 'get_params'):
        model_instance = model_class()
        valid_params = model_instance.get_params()
    else:
        # Fall back to inspecting the constructor's parameters
        model_signature = inspect.signature(model_class.__init__)
        model_params = model_signature.parameters
        valid_params = {param: None for param in model_params if param != 'self'}
    
    # Filter all_params to include only the valid parameters
    parameters = {param: value for param, value in all_params.items() if param in valid_params}
    
    return parameters

def model_func( 
        manual: bool,
        model: str,
        man_params: dict[str, Any],
        model_mapping: dict[str, Type[TrainableModel]] = None
    ):
    if model_mapping is None:
        model_mapping = {
            "RF": RandomForestClassifier,
            'XGB' : XGBClassifier,
            "LGBM": LGBMClassifier,
        }
    model_class = model_mapping.get(model)
    if not model_class or not issubclass(model_class, TrainableModel):
        raise TypeError("Make sure the model has the following methods: `fit`, `predict` and `predict_proba`")
    
    all_params = man_params["parameters"] if manual else dict(wandb.config)
    parameters = set_model_parameters(model_class, all_params)
    
    print(f'Final Parameters of the Model: {parameters}')
    clf = model_class(**parameters)

    return clf
