from numbers import Integral, Real
from typing import Protocol, Any, Type, Tuple, List
import inspect
import threading
import wandb
from sklearn import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, BaseEnsemble
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.parallel import Parallel, delayed
from scipy.sparse import issparse
from scipy.special import logsumexp
from scipy.optimize import minimize
from betacal import BetaCalibration
from venn_abers import VennAbersCalibrator
from joblib import effective_n_jobs
from warnings import catch_warnings, simplefilter
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing_extensions import runtime_checkable
import numpy as np
import pandas as pd


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used for XGB estimators' bootstrapping."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(
        0, n_samples, n_samples_bootstrap, dtype=np.int32
    )

    return sample_indices


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0.0, 1.0]`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, Integral):
        if max_samples > n_samples:
            msg = "`max_samples` must be <= n_samples={} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, Real):
        return max(round(n_samples * max_samples), 1)


def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


def is_not_trivial(tree):
    """Filters out an XGB if any of its trees have depth = 0.

    Args:
        tree: An XGBoost tree object.

    Returns:
        False if any tree is trivial (depth = 0), True otherwise.
    """
    booster = tree.get_booster()
    tree_dump = booster.get_dump(with_stats=True)

    # A tree is trivial (depth = 0) if it consists only of a single leaf node.
    for node in tree_dump:
        if "leaf=" in node and "yes=" not in node and "no=" not in node:
            return False  # Found a trivial tree

    return True


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel."""
    prediction = predict(X)

    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


def stratified_train_test_split(X, y, addi_y=None, use_cudf=False, test_size=0.2, random_state=None):
    """
    Randomly splits data into train and test sets using stratified sampling 
    to preserve label distribution.

    Args:
        X: Input features (numpy array or similar structure).
        y: Target labels (numpy array or similar structure).
        test_size: Proportion of the dataset to include in the test set (float).
        random_state: Seed for the random number generator (int).

    Returns:
        X_train, X_test, y_train, y_test: The split data.
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Input data X and y cannot be empty.")

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed=random_state)

    # Convert pandas objects to numpy if necessary
    if isinstance(X, (pd.DataFrame, pd.Series)) or use_cudf:
        X = X.reset_index(drop=True)  # Reset indices for consistency
    if isinstance(y, pd.Series) or use_cudf:
        y = y.reset_index(drop=True)

    # Unique classes and their indices
    if use_cudf:
        unique_classes, class_indices = np.unique(addi_y, return_inverse=True)
    else:
        unique_classes, class_indices = np.unique(y, return_inverse=True)
    test_indices = []

    # For each class, determine the test indices
    for cls in unique_classes:
        # Get all indices of the current class
        cls_indices = np.where(class_indices == cls)[0]
        # Shuffle class indices
        rng.shuffle(cls_indices)
        # Calculate the number of test samples for this class
        n_test_samples = max(1, int(len(cls_indices) * test_size))
        # Select test indices for the current class
        test_indices.extend(cls_indices[:n_test_samples])

    # Convert to numpy array and shuffle
    test_indices = np.array(test_indices)
    rng.shuffle(test_indices)

    # Get train indices by excluding test indices
    train_indices = np.setdiff1d(np.arange(len(X)), test_indices)

    # Use iloc for pandas and direct indexing for numpy
    if isinstance(X, (pd.DataFrame, pd.Series)) or use_cudf:
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        if use_cudf:
            y_train, y_test = y.iloc[train_indices], test_indices
        else:
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def _get_cudf_feature_names(X, cudf_type):
    """
    Get feature names from a CuDF DataFrame.

    Parameters
    ----------
    X : cudf.DataFrame of shape (n_samples, n_features)
        CuDF DataFrame to extract feature names.

        - CuDF DataFrame: The columns will be considered to be feature
          names. If the DataFrame contains non-string feature names, a `TypeError`
          will be raised.
        - All other input types will return `None`.

    Returns
    -------
    feature_names : ndarray or None
        Feature names of `X`. If input is not a CuDF DataFrame or feature names
        are mixed types, `None` is returned or an error is raised.
    """
    feature_names = None

    if isinstance(X, cudf_type):
        # Extract column names
        feature_names = np.asarray(X.columns, dtype=object)

        if feature_names is None or len(feature_names) == 0:
            return None

        # Check types of feature names
        types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))

        # Mixed types of string and non-string are not supported
        if len(types) > 1 and "str" in types:
            raise TypeError(
                "Feature names are only supported if all input features have string names, "
                f"but your input has {types} as feature name / column name types. "
                "If you want feature names to be stored and validated, you must convert "
                "them all to strings, by using X.columns = X.columns.astype(str) for "
                "example. Otherwise you can remove feature / column names from your input "
                "data, or convert them all to a non-string data type."
            )

        # Only feature names of all strings are supported
        if len(types) == 1 and types[0] == "str":
            return feature_names

    return None


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


class XGBForestClassifier(BaseEnsemble):
    """
    An ensemble classifier that builds a forest of XGBoost estimators.
    
    This classifier trains multiple `XGBClassifier` instances and combines their 
    predictions, similar to Random Forests. It supports bootstrapping, parallelism, 
    and advanced customization of estimator parameters.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of `XGBClassifier` estimators to train in the ensemble.

    bootstrap : bool, default=False
        Whether to use bootstrapping to sample training data for each estimator.

    max_samples : int, float, or None, default=None
        If bootstrap=True, the number of samples to draw for training each estimator.
        - If None, all samples are used.
        - If int, the absolute number of samples.
        - If float, the proportion of samples (0.0 < max_samples <= 1.0).

    n_jobs : int or None, default=None
        The number of jobs to run in parallel. If -1, use all available processors.

    p_strategy : {'threads', 'processes'}, default='threads'
        Parallelism strategy for training the estimators:
        - 'threads': Use multi-threading for parallel execution.
        - 'processes': Use multi-processing for parallel execution.

    use_loky : bool, default=False
        Whether to use the `loky` backend for parallelism, which is more memory-efficient 
        for large datasets but may introduce some overhead.

    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of bootstrapping and the individual estimator random states.

    verbose : int, default=0
        Controls verbosity during training:
        - 0: No output.
        - 1: Minimal output.
        - 2: Detailed output.

    class_weight : {dict, 'balanced', 'balanced_subsample', None}, default=None
        Class weights for handling imbalanced datasets. If 'balanced_subsample' is used, 
        weights are applied to each bootstrap sample.

    Other Parameters
    ----------------
    All additional parameters are passed directly to the underlying `XGBClassifier` instances.

    Attributes
    ----------
    estimators_ : list of XGBClassifier
        The collection of fitted base estimators.

    classes_ : array of shape (n_classes,)
        The unique class labels.

    n_classes_ : int
        The number of classes.

    n_samples_ : int
        The number of samples in the training dataset.

    n_samples_bootstrap : int
        The number of samples used for bootstrapping (if applicable).

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the ensemble of `XGBClassifier` estimators.

    predict(X)
        Predict class labels for the input samples.

    predict_proba(X)
        Predict class probabilities for the input samples.
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "bootstrap": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "max_samples": [
            None,
            Interval(RealNotInt, 0.0, 1.0, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "class_weight": [str, None],
        "p_strategy": [str],
        "use_loky": ["boolean"],
        "xgb_n_estimators": [Interval(Integral, 1, None, closed="left")],
        "objective": [str],
        "max_depth": [Interval(Integral, 0, None, closed="left")],
        "learning_rate": [Interval(Real, 0.0, 1.0, closed="right")],
        "subsample": [Interval(Real, 0.0, 1.0, closed="right")],
        "colsample_bytree": [Interval(Real, 0.0, 1.0, closed="right")],
        "scale_pos_weight": [Interval(Real, 0.0, None, closed="left")],
        "device": [str],
        "tree_method": [str],
        "booster": [str],
        "verbosity": ["verbose"],
        "use_rmm": ["boolean"],
        "seed": [Interval(Integral, 0, None, closed="left")],
        "sampling_method": [str],
        "colsample_bylevel": [Interval(Real, 0.0, 1.0, closed="right")],
        "colsample_bynode": [Interval(Real, 0.0, 1.0, closed="right")],
        "max_delta_step": [Interval(Integral, 0, None, closed="left")],
        "max_leaves": [Interval(Integral, 0, None, closed="left")],
        "max_bin": [Interval(Integral, 1, None, closed="left")],
        "num_parallel_tree": [Interval(Integral, 1, None, closed="left")],
        "refresh_leaf": [Interval(Integral, 0, 1, closed="both")],
        "process_type": [str],
        "early_stopping_rounds": [Interval(Integral, 1, None, closed="left"), None],
        "seed_per_iteration": ["boolean"],
        "multi_strategy": [str],
        "sample_type": [str],
        "one_drop": [Interval(Integral, 0, 1, closed="both")],
        "skip_drop": [Interval(Real, 0.0, 1.0, closed="both")],
        "normalize_type": [str],
        "rate_drop": [Interval(Real, 0.0, 1.0, closed="both")],
        "max_cached_hist_node": [Interval(Integral, 1, None, closed="left")],
        "grow_policy": [str],
        "min_child_weight": [Interval(Integral, 0, None, closed="left")],
        "reg_lambda": [Interval(Real, 0.0, None, closed="left")],
        "reg_alpha": [Interval(Real, 0.0, None, closed="left")],
        "gamma": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_samples=None,
        class_weight=None,
        p_strategy="threads",
        use_loky=False,
        xgb_n_estimators=100,
        objective="binary:logistic",
        nthread=-1,
        max_depth=6,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight=1.0,
        device="cpu",
        tree_method="hist",
        booster="gbtree",
        verbosity=0,
        use_rmm=False,
        seed=0,
        sampling_method="uniform",
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        max_delta_step=0,
        max_leaves=0,
        max_bin=256,
        num_parallel_tree=1,
        refresh_leaf=1,
        process_type="default",
        early_stopping_rounds=None,
        seed_per_iteration=False,
        multi_strategy="one_output_per_tree",
        sample_type="uniform",
        one_drop=0,
        skip_drop=0.0,
        normalize_type="tree",
        rate_drop=0.0,
        max_cached_hist_node=65536,
        grow_policy="depthwise",
        min_child_weight=1,
        reg_lambda=1,
        reg_alpha=0,
        gamma=0
    ):
        super().__init__(
            estimator=XGBClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "objective",
                "nthread",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "scale_pos_weight",
                "device",
                "tree_method",
                "booster",
                "verbosity",
                "use_rmm",
                "seed",
                "sampling_method",
                "colsample_bylevel",
                "colsample_bynode",
                "max_delta_step",
                "max_leaves",
                "max_bin",
                "num_parallel_tree",
                "refresh_leaf",
                "process_type",
                "random_state",
                "early_stopping_rounds",
                "seed_per_iteration",
                "multi_strategy",
                "sample_type",
                "one_drop",
                "skip_drop",
                "normalize_type",
                "rate_drop",
                "max_cached_hist_node",
                "grow_policy",
                "min_child_weight",
                "reg_lambda",
                "reg_alpha",
                "gamma",
            ),
        )

        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.p_strategy = p_strategy
        self.use_loky = use_loky
        self.xgb_n_estimators = xgb_n_estimators
        self.objective = objective
        self.nthread = nthread
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.device = device
        self.tree_method = tree_method
        self.booster = booster
        self.verbosity = verbosity
        self.use_rmm = use_rmm
        self.seed = seed
        self.sampling_method = sampling_method
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.max_delta_step = max_delta_step
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.num_parallel_tree = num_parallel_tree
        self.refresh_leaf = refresh_leaf
        self.process_type = process_type
        self.early_stopping_rounds = early_stopping_rounds
        self.seed_per_iteration = seed_per_iteration
        self.multi_strategy = multi_strategy
        self.sample_type = sample_type
        self.one_drop = one_drop
        self.skip_drop = skip_drop
        self.normalize_type = normalize_type
        self.rate_drop = rate_drop
        self.max_cached_hist_node = max_cached_hist_node
        self.grow_policy = grow_policy
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_samples = None
        self.n_samples_bootstrap = None
        self.use_cudf = False
        self.feature_names_in_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the XGBForestClassifier. samples_
        """
        self._validate_params()

        # Check for sparse `y`
        if issparse(y):
            raise ValueError("Sparse multilabel-indicator for y is not supported.")

        if self.device == 'cuda':
            import cudf

            if isinstance(X, cudf.DataFrame):
                self.use_cudf = True
                np_X = X.to_numpy()
                np_y = y.to_numpy()

        # Validate input data
        if not self.use_cudf:
            X, y = self._validate_data(
                X, y, multi_output=False, accept_sparse="csc", dtype=np.float32
            )
        else:
            self.feature_names_in_ = _get_cudf_feature_names(X, cudf.DataFrame)

        # Validate sample weights
        if sample_weight is not None:
            if self.use_cudf:
                sample_weight = _check_sample_weight(sample_weight, np_X)
            else:
                sample_weight = _check_sample_weight(sample_weight, X)

        # Transform `y` appropriately
        if not self.use_cudf:
            y = np.atleast_1d(y)
            if y.ndim == 2:
                if y.shape[1] > 1:
                    raise ValueError("XGBoost doesn't support multi-output target data.")
                if y.shape[1] == 1:
                    y = np.ravel(y)  # Flatten to (n_samples,)

        # Validate Poisson objective requirements
        if self.use_cudf:
            if self.objective == "count:poisson":
                if np.any(np_y < 0):
                    raise ValueError(
                        "y contains negative values, which are not allowed for Poisson regression."
                    )
                if np.sum(np_y) <= 0:
                    raise ValueError(
                        "Sum of y must be strictly positive for Poisson regression."
                    )
        else:
            if self.objective == "count:poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "y contains negative values, which are not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y must be strictly positive for Poisson regression."
                    )

        self.n_samples = y.shape[0]

        # Validate y data and class_weight
        if self.use_cudf:
            _, expanded_class_weight = self._validate_y_class_weight(np_y)
        else:
            y, expanded_class_weight = self._validate_y_class_weight(y)

        # Ensure `y` is contiguous and of correct dtype
        if not self.use_cudf:
            if not y.flags.contiguous or y.dtype != np.float32:
                y = np.ascontiguousarray(y, dtype=np.float32)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Bootstrap validation and sample size determination
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            self.n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=self.n_samples, max_samples=self.max_samples
            )
        else:
            self.n_samples_bootstrap = None

        self._validate_estimator()

        # Define the training function for each estimator
        def train_estimator(random_state, estimator_idx):
            """
            Train a single XGBClassifier with or without bootstrapping.
            """
            if self.verbose > 1:
                print(f"Training estimator {estimator_idx + 1} of {self.n_estimators}...")

            estimator = self._make_estimator(append=False, random_state=random_state)

            if self.bootstrap:
                if sample_weight is None:
                    curr_sample_weight = np.ones((self.n_samples,), dtype=np.float64)
                else:
                    curr_sample_weight = sample_weight.copy()

                indices = _generate_sample_indices(
                    random_state, self.n_samples, self.n_samples_bootstrap
                )
                sample_counts = np.bincount(indices, minlength=self.n_samples)
                curr_sample_weight *= sample_counts

                if self.class_weight == "subsample":
                    with catch_warnings():
                        simplefilter("ignore", DeprecationWarning)
                        if self.use_cudf:
                            curr_sample_weight *= compute_sample_weight("auto", np_y, indices=indices)
                        else:
                            curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
                elif self.class_weight == "balanced_subsample":
                    if self.use_cudf:
                        curr_sample_weight *= compute_sample_weight("balanced", np_y, indices=indices)
                    else:
                        curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

                estimator.fit(X, y, sample_weight=curr_sample_weight)
            else:
                estimator.fit(X, y)

            return estimator

        # Fit all estimators in parallel
        self.random_state = check_random_state(self.random_state)
        random_states = [
            self.random_state.randint(0, np.iinfo(np.int32).max) for _ in range(self.n_estimators)
        ]

        if self.p_strategy == "processes":
            prefer = "processes"
            backend = "loky" if self.use_loky else None
        elif self.p_strategy == "threads":
            prefer = "threads"
            backend = None
        else:
            raise ValueError(
                "Parallelism strategy should either be 'processes' or 'threads'."
                f"Given: {self.p_strategy}"
            )

        self.estimators_ = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, prefer=prefer, backend=backend
        )(
            delayed(train_estimator)(rs, idx)
            for idx, rs in enumerate(random_states)
        )

        # Delete intermediate arrays
        if self.use_cudf:
            del np_X
            del np_y
            del _

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_"):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        self.classes_ = []
        self.n_classes_ = []

        classes_k = np.unique(y)
        self.classes_.append(classes_k)
        self.n_classes_.append(classes_k.shape[0])

        if self.class_weight is not None:
            valid_presets = ("balanced", "balanced_subsample")
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError(
                        "Valid presets for class_weight include "
                        '"balanced" and "balanced_subsample".'
                        f'Given {self.class_weight}.'
                    )

            if self.class_weight != "balanced_subsample" or not self.bootstrap:
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight, y)

        return y, expanded_class_weight

    def _validate_params(self):
        super()._validate_params()

        # Custom validation for 'nthread'
        if self.nthread not in [-1] and not (
            isinstance(self.nthread, int) and self.nthread >= 1
        ):
            raise ValueError("nthread must be -1 or a positive integer.")

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.estimator_)

        # Create a dictionary of parameters to set, including 'n_estimators'
        params = {p: getattr(self, p) for p in self.estimator_params}
        params['n_estimators'] = self.xgb_n_estimators

        estimator.set_params(**params)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        X = self._validate_data(
            X,
            dtype=np.float32,
            accept_sparse="csr",
            reset=False,
            force_all_finite="allow-nan",
        )
        if issparse(X):
            if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                raise ValueError(
                    "Sparse matrices with np.int64 indices are not supported. "
                    "Convert the indices and indptr to np.int32 using the following: "
                    "X.indices = X.indices.astype(np.intc); "
                    "X.indptr = X.indptr.astype(np.intc)."
                )

        return X

    @property
    def feature_importances_(self):
        """
        The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees in the ensemble 
            are trivial (e.g., single-node trees or trees with no meaningful splits), 
            in which case it will be an array of zeros.
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(getattr)(tree, "feature_importances_")
            for tree in self.estimators_
            if len(tree.get_booster().get_dump()) > 1 and is_not_trivial(tree)
        )

        if not all_importances:

            return np.zeros(self.n_features_in_, dtype=np.float64)

        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)

        if np.sum(all_importances) == 0:
            return np.zeros_like(all_importances)

        return all_importances / np.sum(all_importances)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        if not self.use_cudf:
            X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_
        )

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:

            return all_proba[0]
        else:

            return all_proba

    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)

        return self.classes_.take(np.argmax(proba, axis=1), axis=0)


def _accumulate_prediction_stacked(predict, X, out, idx, lock):
    """
    Utility function for collecting predictions from each estimator into a 3D array.

    Parameters
    ----------
    predict : callable
        The prediction function of the estimator
    X : array-like
        Input data
    out : list of numpy.ndarray
        List containing the 3D array to store all predictions
    idx : int
        The index of the current estimator
    lock : threading.Lock
        Lock for thread-safe operations
    """
    prediction = predict(X)

    with lock:
        out[idx] = prediction


class StackedXGBForestClassifier(XGBForestClassifier):
    """
    A stacking variant of XGBForestClassifier that uses a meta-model to combine
    predictions from all base estimators.

    Parameters
    ----------
    stacked_model : estimator object, default=None
        The meta-model to be trained on the predictions of base estimators.
        Must implement fit and predict_proba methods.
    stacked_models_params : dictionary of parameters, default=None
        The meta-model's parameters with the dictionary type and string keys.
        Must contain valid meta-model's parameters.

    All other parameters are inherited from XGBForestClassifier.

    Attributes
    ----------
    All attributes from XGBForestClassifier plus:

    stacked_model_ : estimator object
        The fitted meta-model
    """

    def __init__(
        self,
        stacked_model=None,
        stacked_models_params=None,
        stacked_model_n_top_features=50,
        use_pca_stacked_model=False,
        pca_n_components=3,
        n_estimators=100,
        *,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_samples=None,
        class_weight=None,
        p_strategy="threads",
        use_loky=False,
        xgb_n_estimators=100,
        objective="binary:logistic",
        nthread=-1,
        max_depth=6,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight=1.0,
        device="cpu",
        tree_method="hist",
        booster="gbtree",
        verbosity=0,
        use_rmm=False,
        seed=0,
        sampling_method="uniform",
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        max_delta_step=0,
        max_leaves=0,
        max_bin=256,
        num_parallel_tree=1,
        refresh_leaf=1,
        process_type="default",
        early_stopping_rounds=None,
        seed_per_iteration=False,
        multi_strategy="one_output_per_tree",
        sample_type="uniform",
        one_drop=0,
        skip_drop=0.0,
        normalize_type="tree",
        rate_drop=0.0,
        max_cached_hist_node=65536,
        grow_policy="depthwise",
        min_child_weight=1,
        reg_lambda=1,
        reg_alpha=0,
        gamma=0
    ):
        super().__init__(
            n_estimators = n_estimators,
            bootstrap = bootstrap,
            n_jobs = n_jobs,
            random_state = random_state,
            verbose = verbose,
            max_samples = max_samples,
            class_weight = class_weight,
            p_strategy = p_strategy,
            use_loky = use_loky,
            xgb_n_estimators = xgb_n_estimators,
            objective = objective,
            nthread = nthread,
            max_depth = max_depth,
            learning_rate = learning_rate,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            scale_pos_weight = scale_pos_weight,
            device = device,
            tree_method = tree_method,
            booster = booster,
            verbosity = verbosity,
            use_rmm = use_rmm,
            seed = seed,
            sampling_method = sampling_method,
            colsample_bylevel = colsample_bylevel,
            colsample_bynode = colsample_bynode,
            max_delta_step = max_delta_step,
            max_leaves = max_leaves,
            max_bin = max_bin,
            num_parallel_tree = num_parallel_tree,
            refresh_leaf = refresh_leaf,
            process_type = process_type,
            early_stopping_rounds = early_stopping_rounds,
            seed_per_iteration = seed_per_iteration,
            multi_strategy = multi_strategy,
            sample_type = sample_type,
            one_drop = one_drop,
            skip_drop = skip_drop,
            normalize_type = normalize_type,
            rate_drop = rate_drop,
            max_cached_hist_node = max_cached_hist_node,
            grow_policy = grow_policy,
            min_child_weight = min_child_weight,
            reg_lambda = reg_lambda,
            reg_alpha = reg_alpha,
            gamma = gamma,
        )
        self.stacked_model = stacked_model
        self.stacked_models_params = stacked_models_params
        self.stacked_model_n_top_features = stacked_model_n_top_features
        self.use_pca_stacked_model = use_pca_stacked_model
        self.pca_n_components = pca_n_components

    def meta_feature_engineering(self, df, n_components):
        """
        This function calculates PCA for all astimators' output probs
        as an input for the stacked model.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        df = df.copy()
        model_columns = df.columns.tolist()
        col_prob = [col for col in model_columns if "th_est_pos_label_proba" in col]

        # Calculate statistics on probability columns
        df['mean'] = df[col_prob].mean(axis=1)
        df['std'] = df[col_prob].std(axis=1)
        df['variance'] = df[col_prob].var(axis=1)
        df['row_count_above_0.5'] = (df[col_prob] > 0.5).sum(axis=1)
        df['max_prob'] = df[col_prob].max(axis=1)
        df['min_prob'] = df[col_prob].min(axis=1)
        df['median_prob'] = df[col_prob].median(axis=1)

        # Standardize and perform PCA on probability columns
        if self.pca_model is not None :
            col_prob = self.pca_model['column']
            data_for_pca = df[col_prob].to_numpy()
            pca = self.pca_model['pca']
            scaler = self.pca_model['scaler']
            scaled_data = scaler.transform(data_for_pca)
            principal_components = pca.transform(scaled_data)
        else:
            data_for_pca = df[col_prob].to_numpy()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_for_pca)
            
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(scaled_data)

            self.pca_model = {
                'pca' : pca,
                'scaler' : scaler ,
                'column' : col_prob ,
            }

        for i in range(n_components):
            df[f"prob_PCA{i}"] = principal_components[:, i]

        df = df.drop(columns=col_prob)

        return df

    def predict_proba(self, X, y=None, stacked_model_trained=True):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed using
        the estimations derived from the stacked model that is trained on the
        output probabilities of the estimators, alongside the input X data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        # if not self.use_cudf:
        #     X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # storing the output of every estimator
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d([self.n_classes_]*len(self.estimators_))
        ]
        lock = threading.Lock()

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction_stacked)(e.predict_proba, X, all_proba, i, lock)
            for i, e in enumerate(self.estimators_)
        )

        feature_importances = self.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]
        top_features = X.columns[sorted_idx[:self.stacked_model_n_top_features]]
        X = X[top_features].copy()

        if not stacked_model_trained:
            if y is None:
                raise ValueError(
                    "When the stacked model is not trained yet,"
                    " y must be set."
                )

            if self.use_cudf:
                import cudf

                for idx, est_proba in enumerate(all_proba):
                    X[f"{idx}th_est_pos_label_proba"] = cudf.Series(
                        est_proba[:, 1],
                        index=X.index
                    )
            else:
                for idx, est_proba in enumerate(all_proba):
                    X.loc[:, f"{idx}th_est_pos_label_proba"] = est_proba[:, 1]

            if isinstance(self.stacked_models_params, dict):
                for key in self.stacked_models_params.keys():
                    if not isinstance(key, str):
                        raise ValueError(
                            "All of the keys of the 'stacked_models_params' "
                            "dictionary must be of type string."
                        )
            else:
                raise ValueError(
                    "'stacked_models_params' should be a dictionary "
                    "containing the stacked model's parameters. "
                    f"Given {self.stacked_models_params}."
                )

            if isinstance(self.stacked_model, str):
                if self.stacked_model not in ["RF", "XGB", "XGBF", "LGBM"]:
                    raise ValueError(
                        "Currently, the stacked model should either be one of "
                        f"'RF', 'XGB', 'XGBF' or 'LGBM'. Given {self.stacked_model}."
                    )

                if self.stacked_model == "RF":
                    model_class = RandomForestClassifier
                elif self.stacked_model == "XGB":
                    model_class = XGBClassifier
                elif self.stacked_model == "XGBF":
                    model_class = XGBForestClassifier
                else: # self.stacked_model == "LGBM"
                    model_class = LGBMClassifier

            elif isinstance(self.stacked_model, RandomForestClassifier):
                model_class = RandomForestClassifier

            elif isinstance(self.stacked_model, XGBClassifier):
                model_class = XGBClassifier

            elif isinstance(self.stacked_model, XGBForestClassifier):
                model_class = XGBForestClassifier

            elif isinstance(self.stacked_model, LGBMClassifier):
                model_class = LGBMClassifier

            model_instance = model_class()
            valid_params = model_instance.get_params()
            parameters = {
                param: value for param, value in self.stacked_models_params.items()
                if param in valid_params
            }
            self.stacked_model = model_class(**parameters)
            self.pca_model = None
            if self.use_pca_stacked_model:
                X = self.meta_feature_engineering(df=X, n_components=self.pca_n_components)

            self.stacked_model.fit(X, y)

            return
        else:
            if self.use_cudf:
                import cudf

                for idx, est_proba in enumerate(all_proba):
                    X[f"{idx}th_est_pos_label_proba"] = cudf.Series(
                        est_proba[:, 1],
                        index=X.index
                    )
            else:
                for idx, est_proba in enumerate(all_proba):
                    X[f"{idx}th_est_pos_label_proba"] = est_proba[:, 1]

            if self.use_pca_stacked_model:
                X = self.meta_feature_engineering(df=X, n_components=self.pca_n_components)

            return self.stacked_model.predict_proba(X)


# Temperature Scaling calibration
class TemperatureScalingCalibrator:

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        def loss_fn(temp):
            scaled_logits = logits / temp[0]
            log_probs = scaled_logits - logsumexp(scaled_logits, axis=1, keepdims=True)
            loss = -np.mean(log_probs[np.arange(len(y_true)), y_true])

            return loss

        res = minimize(loss_fn, [self.temperature], bounds=[(1e-2, 1e2)], method="L-BFGS-B")
        self.temperature = res.x[0]

    def predict_proba(self, logits):
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(
            scaled_logits - logsumexp(scaled_logits, axis=1, keepdims=True)
        )

        return exp_logits


class ClassificationConformalPredictor:

    def __init__(
        self,
        model=None,
        meta_model=None,
        meta_models_params=None,
        calibrate_meta=False,
        n_top_features=50,
        use_valid_as_calib=False,
        prob_estimator=None,
        retrain=False,
        use_cv=False,
        n_folds=1,
        calibration_size=0.2,
        calibration_method=None,
        zero_cls_scale=1,
        apply_calibs=False,
        random_state=42,
        use_meta_labeling=False,
    ):
        """
        Parameters:
        model: Any sklearn-compatible classifier with predict_proba
        alpha: Significance level (1 - confidence)
        """
        self.model = model
        self.meta_model = meta_model
        self.meta_models_params = meta_models_params
        self.calibrate_meta = calibrate_meta
        self.n_top_features = n_top_features
        self.use_valid_as_calib = use_valid_as_calib
        self.prob_estimator = prob_estimator
        self.retrain = retrain
        self.use_cv = use_cv
        self.n_folds = n_folds
        self.calibration_size = calibration_size
        self.calibration_method = calibration_method
        self.zero_cls_scale = zero_cls_scale
        self.apply_calibs = apply_calibs
        self.random_state = random_state
        self.use_meta_labeling = use_meta_labeling
        self.calibration_scores = {}
        self.use_cudf = False
        self.classes_ = None
        self.calibrator = None
        self.meta_calibrator = None
        self.meta_pos_label_perc = None

    def _apply_calibration(self, prob_pred, y_calib, model=None):
        """
        Applies the selected calibration method to the model's probabilities.
        """
        if self.calibration_method == "temperature_scaling":
            if model == "meta":
                self.meta_calibrator = TemperatureScalingCalibrator()
                logits = np.log(prob_pred + 1e-15)  # Convert probabilities to logits
                self.meta_calibrator.fit(logits, y_calib)
            else:
                self.calibrator = TemperatureScalingCalibrator()
                logits = np.log(prob_pred + 1e-15)  # Convert probabilities to logits
                self.calibrator.fit(logits, y_calib)

        elif self.calibration_method == "beta_calibration":
            if model == "meta":
                self.meta_calibrator = {}
                for idx, cls in enumerate(self.classes_):
                    # For each class, fit a separate BetaCalibration instance
                    meta_calibrator = BetaCalibration()
                    meta_calibrator.fit(prob_pred[:, idx], (y_calib == cls).astype(int))
                    self.meta_calibrator[cls] = meta_calibrator
            else:
                self.calibrator = {}
                for idx, cls in enumerate(self.classes_):
                    # For each class, fit a separate BetaCalibration instance
                    calibrator = BetaCalibration()
                    calibrator.fit(prob_pred[:, idx], (y_calib == cls).astype(int))
                    self.calibrator[cls] = calibrator

        elif self.calibration_method == "beta_temp" or self.calibration_method == "temp_beta":
            if model == "meta":
                self.meta_calibrator = {
                    'temp': TemperatureScalingCalibrator(),
                }

                logits = np.log(prob_pred + 1e-15)  # Convert probabilities to logits
                self.meta_calibrator['temp'].fit(logits, y_calib)

                for idx, cls in enumerate(self.classes_):
                    # For each class, fit a separate BetaCalibration instance
                    meta_calibrator = BetaCalibration()
                    meta_calibrator.fit(prob_pred[:, idx], (y_calib == cls).astype(int))
                    self.meta_calibrator[cls] = meta_calibrator
            else:
                self.calibrator = {
                    'temp': TemperatureScalingCalibrator(),
                }

                logits = np.log(prob_pred + 1e-15)  # Convert probabilities to logits
                self.calibrator['temp'].fit(logits, y_calib)

                for idx, cls in enumerate(self.classes_):
                    # For each class, fit a separate BetaCalibration instance
                    calibrator = BetaCalibration()
                    calibrator.fit(prob_pred[:, idx], (y_calib == cls).astype(int))
                    self.calibrator[cls] = calibrator

        elif self.calibration_method == "venn_abers":
            if model == "meta":
                self.meta_calibrator = VennAbersCalibrator(
                    estimator=self.meta_model,
                    inductive=True,
                    cal_size=0.25,
                    random_state=self.random_state,
                    precision=2
                )
                self.meta_calibrator.fit(prob_pred, y_calib)
            else:
                self.calibrator = VennAbersCalibrator(
                    estimator=self.model,
                    inductive=True,
                    cal_size=0.25,
                    random_state=self.random_state,
                    precision=2
                )
                self.calibrator.fit(prob_pred, y_calib)

        elif self.calibration_method is None:
            self.calibrator = None

        else:
            raise ValueError(
                "'calibration_method' should either be one of "
                "'beta_calibration', 'temperature_scaling' or None. "
                f"Given {self.calibration_method} instead."
            )

    def _calibrate_proba(self, prob_pred, model=None):
        """
        Calibrates the probabilities using the chosen method if applicable.
        """
        if self.calibrator is not None or self.meta_calibrator is not None:
            if self.calibration_method == "temperature_scaling":
                logits = np.log(prob_pred + 1e-15)  # Convert probabilities to logits
                if model == "meta":
                    prob_pred = self.meta_calibrator.predict_proba(logits)
                else:
                    prob_pred = self.calibrator.predict_proba(logits)

            elif self.calibration_method == "beta_calibration":
                # Calibrate probabilities for each class separately
                calibrated_probs = np.zeros_like(prob_pred)
                if model == "meta":
                    for idx, cls in enumerate(self.classes_):
                        calibrated_probs[:, idx] = self.meta_calibrator[cls].predict(prob_pred[:, idx])
                else:
                    for idx, cls in enumerate(self.classes_):
                        calibrated_probs[:, idx] = self.calibrator[cls].predict(prob_pred[:, idx])
                prob_pred = calibrated_probs

            elif self.calibration_method == "beta_temp" or self.calibration_method == "temp_beta":
                # Calibrate probabilities for each class separately
                calibrated_probs = np.zeros_like(prob_pred)
                if model == "meta":
                    for idx, cls in enumerate(self.classes_):
                        calibrated_probs[:, idx] = self.meta_calibrator[cls].predict(prob_pred[:, idx])
                else:
                    for idx, cls in enumerate(self.classes_):
                        calibrated_probs[:, idx] = self.calibrator[cls].predict(prob_pred[:, idx])
                beta_prob_pred = calibrated_probs

                logits = np.log(prob_pred + 1e-15)  # Convert probabilities to logits
                if model == "meta":
                    prob_pred = self.meta_calibrator['temp'].predict_proba(logits)
                else:
                    prob_pred = self.calibrator['temp'].predict_proba(logits)
                temp_prob_pred = prob_pred

                return beta_prob_pred, temp_prob_pred

            elif self.calibration_method == "venn_abers":
                if model == "meta":
                    prob_pred = self.meta_calibrator.predict_proba(prob_pred)
                else:
                    prob_pred = self.calibrator.predict_proba(prob_pred)

        return prob_pred

    def get_params(self, deep=True):
        """
        Get parameters for this conformal predictor.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns the parameters of this predictor and
            the contained model.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "model": self.model,
            "meta_model": self.meta_model,
            "meta_models_params": self.meta_models_params,
            "calibrate_meta": self.calibrate_meta,
            "n_top_features": self.n_top_features,
            "use_valid_as_calib": self.use_valid_as_calib,
            "prob_estimator": self.prob_estimator,
            "retrain": self.retrain,
            "use_cv": self.use_cv,
            "n_folds": self.n_folds,
            "calibration_size": self.calibration_size,
            "calibration_method": self.calibration_method,
            "zero_cls_scale": self.zero_cls_scale,
            "apply_calibs": self.apply_calibs,
            "random_state": self.random_state,
            "use_meta_labeling": self.use_meta_labeling,
        }

        if deep and self.model is not None and hasattr(self.model, "get_params"):
            # Add nested model's parameters
            model_params = self.model.get_params(deep=True)
            params.update({f"model__{key}": value for key, value in model_params.items()})

        return params

    def fit(self, X_train, y_train, addi_X=None, addi_y=None, use_cudf=False):
        self.use_cudf = use_cudf
        if self.model is None:
            raise ValueError("Model cannot be None.")

        if len(np.unique(y_train)) == 1:
            raise ValueError("'y_train' cannot contain only one class.")

        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data X_train and y_train cannot be empty.")

        if self.n_folds < 1:
            raise ValueError("'n_folds' cannot be negative or zero.")

        if self.use_valid_as_calib:
            self.model.fit(X_train, y_train)
            self.classes_ = self.model.classes_

            if use_cudf and (addi_X is not None and addi_y is not None):
                if self.calibration_method == "venn_abers":
                    self._apply_calibration(addi_X, addi_y)
            else:
                if self.calibration_method == "venn_abers":
                    self._apply_calibration(X_train, y_train)
        else:
            if self.use_cv:
                kf = StratifiedKFold(
                    n_splits=self.n_folds, shuffle=True, random_state=self.random_state
                )

                if use_cudf and (addi_X is not None and addi_y is not None):
                    self.classes_ = np.unique(addi_y)
                    self.calibration_scores = {cls: [] for cls in self.classes_}

                    for train_idx, calib_idx in kf.split(addi_X, addi_y):
                        # Split data
                        X_proper_train = X_train.iloc[train_idx]
                        y_proper_train = y_train.iloc[train_idx]
                        X_calib = X_train.iloc[calib_idx]
                        addi_y_calib = addi_y.iloc[calib_idx]

                        # Train model
                        model_clone = clone(self.model)
                        model_clone.fit(X_proper_train, y_proper_train)
                        self.classes_ = model_clone.classes_

                        # Get probability predictions of the calibration data
                        prob_pred = model_clone.predict_proba(X_calib)

                        # Apply calibration
                        if self.calibration_method is not None and self.calibration_method != "venn_abers":
                            self._apply_calibration(prob_pred, addi_y_calib)
                            if self.calibration_method == 'beta_temp':
                                prob_pred, _ = self._calibrate_proba(prob_pred)
                            elif self.calibration_method == 'temp_beta':
                                _, prob_pred = self._calibrate_proba(prob_pred)
                            else:
                                prob_pred = self._calibrate_proba(prob_pred)

                        for cls in self.classes_:
                            cls_indices = np.where(addi_y_calib == cls)[0]
                            self.calibration_scores[cls].extend(
                                1 - prob_pred[
                                    cls_indices, np.where(self.classes_ == cls)[0][0]
                                ]
                            )
                else:
                    self.classes_ = np.unique(y_train)
                    self.calibration_scores = {cls: [] for cls in self.classes_}

                    for train_idx, calib_idx in kf.split(X_train, y_train):
                        # Split data
                        X_proper_train = X_train.iloc[train_idx]
                        y_proper_train = y_train.iloc[train_idx]
                        X_calib = X_train.iloc[calib_idx]
                        y_calib = y_train.iloc[calib_idx]

                        # Train model
                        model_clone = clone(self.model)
                        model_clone.fit(X_proper_train, y_proper_train)
                        self.classes_ = model_clone.classes_

                        # Get probability predictions of the calibration data
                        prob_pred = model_clone.predict_proba(X_calib)

                        # Apply calibration
                        if self.calibration_method is not None and self.calibration_method != "venn_abers":
                            self._apply_calibration(prob_pred, y_calib)
                            if self.calibration_method == 'beta_temp':
                                prob_pred, _ = self._calibrate_proba(prob_pred)
                            elif self.calibration_method == 'temp_beta':
                                _, prob_pred = self._calibrate_proba(prob_pred)
                            else:
                                prob_pred = self._calibrate_proba(prob_pred)

                        for cls in self.classes_:
                            cls_indices = np.where(y_calib == cls)[0]
                            self.calibration_scores[cls].extend(
                                1 - prob_pred[
                                    cls_indices, np.where(self.classes_ == cls)[0][0]
                                ]
                            )

                # Fit final model on all data
                self.model.fit(X_train, y_train)

                # Apply Venn-ABERS
                if use_cudf and (addi_X is not None and addi_y is not None):
                    if self.calibration_method == "venn_abers" and self.apply_calibs:
                        self._apply_calibration(addi_X, addi_y)
                else:
                    if self.calibration_method == "venn_abers" and self.apply_calibs:
                        self._apply_calibration(X_train, y_train)

            else:
                # Split data into proper training and calibration sets
                X_proper_train, X_calib, y_proper_train, y_calib = stratified_train_test_split(
                    X_train, y_train, addi_y=addi_y, use_cudf=use_cudf, test_size=self.calibration_size,
                    random_state=self.random_state
                )

                if use_cudf:
                    y_calib = addi_y.iloc[y_calib]

                # Fit the model on proper training set
                self.model.fit(X_proper_train, y_proper_train)
                self.classes_ = self.model.classes_

                # Get probability predictions of the calibration data
                prob_pred = self.model.predict_proba(X_calib)

                # Apply calibration
                if self.calibration_method is not None and self.calibration_method != "venn_abers":
                    self._apply_calibration(prob_pred, y_calib)
                    if self.calibration_method == 'beta_temp':
                        prob_pred, _ = self._calibrate_proba(prob_pred)
                    elif self.calibration_method == 'temp_beta':
                        _, prob_pred = self._calibrate_proba(prob_pred)
                    else:
                        prob_pred = self._calibrate_proba(prob_pred)

                # Calculate nonconformity scores per class
                self.calibration_scores = {cls: [] for cls in self.classes_}

                for cls in self.classes_:
                    cls_indices = np.where(y_calib == cls)[0]
                    self.calibration_scores[cls] = 1 - prob_pred[
                        cls_indices, np.where(self.classes_ == cls)[0][0]
                    ]

                if self.retrain:
                    # Fit final model on all data if needed
                    self.model.fit(X_train, y_train)

                # Apply Venn-ABERS
                if use_cudf and (addi_X is not None and addi_y is not None):
                    if self.calibration_method == "venn_abers" and self.apply_calibs:
                        self._apply_calibration(addi_X, addi_y)
                else:
                    if self.calibration_method == "venn_abers" and self.apply_calibs:
                        self._apply_calibration(X_train, y_train)

        return self

    def predict(
        self, X, y, set_name, addi_X=None, addi_y=None, confidence_level=None
    ) -> Tuple[np.ndarray, List[set] | None]:
        """
        Returns:
        - predictions: most likely class for each sample
        - confidence: confidence score for each prediction
        - prediction_sets: array of sets of possible classes for each sample 
            within the confidence level
        """
        if len(X) == 0:
            raise ValueError("Input data X cannot be empty.")

        if set_name not in ["train", "valid", "test"]:
            raise ValueError(
                "'set_name' should either be 'train', 'valid' or 'test'."
                f" Given {set_name}")

        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet.")

        if confidence_level is not None and (
            confidence_level <= 0 or confidence_level >= 1
        ):
            raise ValueError("Confidence level must be between 0 and 1.")

        if self.zero_cls_scale <= 0:
            raise ValueError("'zero_cls_scale' cannot be negative or zero.")

        if self.calibration_method is None and self.apply_calibs:
            raise ValueError(
                "Calibration probabilities cannot be applied "
                "when 'calibration_method' is set to None."
            )

        if self.use_valid_as_calib:
            if set_name == "train":
                # Get probability predictions
                prob_pred = self.model.predict_proba(X)

                # Get predictions
                predictions = self.classes_[np.argmax(prob_pred, axis=1)]
                prediction_sets = None

                # Fit calibration
                if self.use_cudf:
                    if self.calibration_method is not None and self.calibration_method != "venn_abers":
                        self._apply_calibration(prob_pred, addi_y)
                else:
                    if self.calibration_method is not None and self.calibration_method != "venn_abers":
                        self._apply_calibration(prob_pred, y)

            elif set_name == "valid":
                if len(y) == 0:
                    raise ValueError("Input data y cannot be empty.")

                # Get probability predictions
                prob_pred = self.model.predict_proba(X)

                # Get predictions
                predictions = self.classes_[np.argmax(prob_pred, axis=1)]
                prediction_sets = None

                # Apply calibration
                if self.use_cudf:
                    if self.calibration_method is not None:
                        if self.calibration_method == "venn_abers":
                            base_prob_pred = prob_pred.copy()
                            prob_pred = self._calibrate_proba(addi_X)
                        elif self.calibration_method == "beta_temp":
                            base_prob_pred = prob_pred.copy()
                            prob_pred, temp_prob_pred = self._calibrate_proba(prob_pred)
                        elif self.calibration_method == "temp_beta":
                            base_prob_pred = prob_pred.copy()
                            beta_prob_pred, prob_pred = self._calibrate_proba(prob_pred)
                        else:
                            base_prob_pred = prob_pred.copy()
                            prob_pred = self._calibrate_proba(prob_pred)
                else:
                    if self.calibration_method is not None:
                        if self.calibration_method == "venn_abers":
                            base_prob_pred = prob_pred.copy()
                            prob_pred = self._calibrate_proba(X)
                        elif self.calibration_method == "beta_temp":
                            base_prob_pred = prob_pred.copy()
                            prob_pred, temp_prob_pred = self._calibrate_proba(prob_pred)
                        elif self.calibration_method == "temp_beta":
                            base_prob_pred = prob_pred.copy()
                            beta_prob_pred, prob_pred = self._calibrate_proba(prob_pred)
                        else:
                            base_prob_pred = prob_pred.copy()
                            prob_pred = self._calibrate_proba(prob_pred)

                if self.prob_estimator == "meta":
                    if isinstance(self.meta_models_params, dict):
                        for key in self.meta_models_params.keys():
                            if not isinstance(key, str):
                                raise ValueError(
                                    "All of the keys of the 'meta_models_params' "
                                    "dictionary must be of type string."
                                )
                    else:
                        raise ValueError(
                            "'meta_models_params' should be a dictionary "
                            "containing the meta model's parameters. "
                            f"Given {self.meta_models_params}."
                        )

                    if isinstance(self.meta_model, str):
                        if self.meta_model not in ["RF", "XGB", "XGBF", "LGBM"]:
                            raise ValueError(
                                "Currently, the meta model should either be one of "
                                f"'RF', 'XGB', 'XGBF' or 'LGBM'. Given {self.meta_model}."
                            )

                        if self.meta_model == "RF":
                            model_class = RandomForestClassifier
                        elif self.meta_model == "XGB":
                            model_class = XGBClassifier
                        elif self.meta_model == "XGBF":
                            model_class = XGBForestClassifier
                        else: # self.meta_model == "LGBM"
                            model_class = LGBMClassifier

                    elif isinstance(self.meta_model, RandomForestClassifier):
                        model_class = RandomForestClassifier

                    elif isinstance(self.meta_model, XGBClassifier):
                        model_class = XGBClassifier

                    elif isinstance(self.meta_model, XGBForestClassifier):
                        model_class = XGBForestClassifier

                    elif isinstance(self.meta_model, LGBMClassifier):
                        model_class = LGBMClassifier

                    model_instance = model_class()
                    valid_params = model_instance.get_params()
                    parameters = {
                        param: value for param, value in self.meta_models_params.items()
                        if param in valid_params
                    }
                    self.meta_model = model_class(**parameters)

                    feature_importances = self.model.feature_importances_
                    sorted_idx = np.argsort(feature_importances)[::-1]
                    top_features = X.columns[sorted_idx[:self.n_top_features]]

                    X_meta = X[top_features].copy()
                    if self.use_cudf:
                        y_meta = (predictions == addi_y).astype(int)
                    else:
                        y_meta = (predictions == y).astype(int)
                    y_temp = y_meta.copy()

                    if self.use_meta_labeling:
                        X_meta = X_meta.loc[predictions == 1]
                        y_meta = y_meta[predictions == 1]
                        y_temp = y_temp[predictions == 1]
                        prob_pred = prob_pred[predictions == 1]

                        if self.calibration_method is not None:
                            base_prob_pred = base_prob_pred[predictions == 1]

                    if list(self.classes_) == [0, 1]:
                        self.meta_pos_label_perc = round((np.sum(y_meta) / y_meta.size) * 100, 2)

                    if self.use_cudf:
                        import cudf

                        y_meta = cudf.Series(y_meta)

                        if self.calibration_method == 'beta_temp':
                            X_meta["base_model's_probs"] = cudf.Series(
                                base_prob_pred[:, 1], index=X_meta.index
                            )
                            X_meta["base_model's_beta_probs"] = cudf.Series(
                                prob_pred[:, 1], index=X_meta.index
                            )
                            X_meta["base_model's_temp_probs"] = cudf.Series(
                                temp_prob_pred[:, 1], index=X_meta.index
                            )
                        elif self.calibration_method == 'temp_beta':
                            X_meta["base_model's_probs"] = cudf.Series(
                                base_prob_pred[:, 1], index=X_meta.index
                            )
                            X_meta["base_model's_beta_probs"] = cudf.Series(
                                beta_prob_pred[:, 1], index=X_meta.index
                            )
                            X_meta["base_model's_temp_probs"] = cudf.Series(
                                prob_pred[:, 1], index=X_meta.index
                            )
                        elif self.calibration_method is not None:
                            X_meta["base_model's_probs"] = cudf.Series(
                                base_prob_pred[:, 1], index=X_meta.index
                            )
                            X_meta["base_model's_calib_probs"] = cudf.Series(
                                prob_pred[:, 1], index=X_meta.index
                            )
                        else:
                            X_meta["base_model's_probs"] = cudf.Series(
                                prob_pred[:, 1], index=X_meta.index
                            )
                    else:
                        if self.calibration_method == 'beta_temp':
                            X_meta["base_model's_probs"] = base_prob_pred[:, 1]
                            X_meta["base_model's_beta_probs"] = prob_pred[:, 1]
                            X_meta["base_model's_temp_probs"] = temp_prob_pred[:, 1]
                        elif self.calibration_method == 'temp_beta':
                            X_meta["base_model's_probs"] = base_prob_pred[:, 1]
                            X_meta["base_model's_beta_probs"] = beta_prob_pred[:, 1]
                            X_meta["base_model's_temp_probs"] = prob_pred[:, 1]
                        elif self.calibration_method is not None:
                            X_meta["base_model's_probs"] = base_prob_pred[:, 1]
                            X_meta["base_model's_calib_probs"] = prob_pred[:, 1]
                        else:
                            X_meta["base_model's_probs"] = prob_pred[:, 1]

                    self.meta_model.fit(X_meta, y_meta)

                    if self.use_cudf:
                        if self.calibrate_meta and self.calibration_method == "venn_abers":
                            self._apply_calibration(X_meta.to_pandas(), y_temp, model="meta")
                    else:
                        if self.calibrate_meta and self.calibration_method == "venn_abers":
                            self._apply_calibration(X_meta, y_meta, model="meta")

                    del X_meta
                    del y_meta
                    del y_temp
                else:
                    # Calculate nonconformity scores per class
                    self.calibration_scores = {cls: [] for cls in self.classes_}

                    if self.use_cudf:
                        for cls in self.classes_:
                            cls_indices = np.where(addi_y == cls)[0]
                            self.calibration_scores[cls] = 1 - prob_pred[
                                cls_indices, np.where(self.classes_ == cls)[0][0]
                            ]
                    else:
                        for cls in self.classes_:
                            cls_indices = np.where(y == cls)[0]
                            self.calibration_scores[cls] = 1 - prob_pred[
                                cls_indices, np.where(self.classes_ == cls)[0][0]
                            ]

            else: # set_name == test
                if confidence_level is not None:
                    # Calculate per-class thresholds
                    thresholds = {
                        cls: np.quantile(self.calibration_scores[cls], confidence_level)
                        for cls in self.classes_
                    }

                    if self.zero_cls_scale != 1:
                        thresholds[0] *= self.zero_cls_scale

                if self.apply_calibs:
                    # Get probability predictions
                    if self.calibration_method == "venn_abers":
                        if self.use_cudf:
                            prob_pred = self._calibrate_proba(addi_X)
                        else:
                            prob_pred = self._calibrate_proba(X)
                    elif self.calibration_method == 'beta_temp':
                        prob_pred, _ = self._calibrate_proba(self.model.predict_proba(X))
                    elif self.calibration_method == 'temp_beta':
                        _, prob_pred = self._calibrate_proba(self.model.predict_proba(X))
                    else:
                        prob_pred = self._calibrate_proba(self.model.predict_proba(X))

                    # Get predictions
                    predictions = self.classes_[np.argmax(prob_pred, axis=1)]
                else:
                    # Get probability predictions
                    prob_pred = self.model.predict_proba(X)

                    # Get predictions
                    predictions = self.classes_[np.argmax(prob_pred, axis=1)]

                    # Calibrate probabilities
                    if self.calibration_method == "venn_abers":
                        if self.use_cudf:
                            prob_pred = self._calibrate_proba(addi_X)
                        else:
                            prob_pred = self._calibrate_proba(X)
                    elif self.calibration_method == 'beta_temp':
                        prob_pred, _ = self._calibrate_proba(prob_pred)
                    elif self.calibration_method == 'temp_beta':
                        _, prob_pred = self._calibrate_proba(prob_pred)
                    else:
                        prob_pred = self._calibrate_proba(prob_pred)

                if confidence_level is not None:
                    # Create an array of prediction sets, one set per sample
                    prediction_sets = [
                        set(
                            cls for cls, threshold in thresholds.items()
                            if 1 - probs[np.where(self.classes_ == cls)[0][0]] <= threshold
                        )
                        for probs in prob_pred
                    ]
                else:
                    prediction_sets = None

        else:
            if confidence_level is not None:
                # Calculate per-class thresholds
                thresholds = {
                    cls: np.quantile(self.calibration_scores[cls], confidence_level)
                    for cls in self.classes_
                }

                if self.zero_cls_scale != 1:
                    thresholds[0] *= self.zero_cls_scale

            if self.apply_calibs:
                # Get probability predictions
                if self.calibration_method == "venn_abers":
                    if self.use_cudf:
                        prob_pred = self._calibrate_proba(addi_X)
                    else:
                        prob_pred = self._calibrate_proba(X)
                elif self.calibration_method == 'beta_temp':
                    prob_pred, _ = self._calibrate_proba(self.model.predict_proba(X))
                elif self.calibration_method == 'temp_beta':
                    _, prob_pred = self._calibrate_proba(self.model.predict_proba(X))
                else:
                    prob_pred = self._calibrate_proba(self.model.predict_proba(X))

                # Get predictions
                predictions = self.classes_[np.argmax(prob_pred, axis=1)]

                if self.calibration_method == "venn_abers":
                    prob_pred = self.model.predict_proba(X)
            else:
                # Get probability predictions
                prob_pred = self.model.predict_proba(X)

                # Get predictions
                predictions = self.classes_[np.argmax(prob_pred, axis=1)]

                # Calibrate probabilities
                if self.calibration_method != "venn_abers":
                    if self.calibration_method == 'beta_temp':
                        prob_pred, _ = self._calibrate_proba(prob_pred)
                    elif self.calibration_method == 'temp_beta':
                        _, prob_pred = self._calibrate_proba(prob_pred)
                    else:
                        prob_pred = self._calibrate_proba(prob_pred)

            if confidence_level is not None:
                # Create an array of prediction sets, one set per sample
                prediction_sets = [
                    set(
                        cls for cls, threshold in thresholds.items()
                        if 1 - probs[np.where(self.classes_ == cls)[0][0]] <= threshold
                    )
                    for probs in prob_pred
                ]
            else:
                prediction_sets = None

        return predictions, prediction_sets

    def categorize_proba(
            self, X, y, confidence_levels: List[float], addi_X=None, addi_y=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Categorizes predictions based on the range of confidence levels at which they remain stable.

        Args:
            X: Input data to predict on.
            confidence_levels: List of confidence levels to evaluate, in descending order.

        Returns:
            A list of confidence levels indicating the range each prediction can be trusted upon.
        """
        if self.prob_estimator == "meta":
            # Get base model's probabilities
            prob_pred = self.model.predict_proba(X)

            # Get predictions
            predictions = self.classes_[np.argmax(prob_pred, axis=1)]

            # Apply calibration
            if self.calibration_method is not None:
                if self.calibration_method == "venn_abers":
                    base_prob_pred = prob_pred.copy()
                    if self.use_cudf:
                        prob_pred = self._calibrate_proba(addi_X)
                    else:
                        prob_pred = self._calibrate_proba(X)
                elif self.calibration_method == 'beta_temp':
                    base_prob_pred = prob_pred.copy()
                    prob_pred, temp_prob_pred = self._calibrate_proba(prob_pred)
                elif self.calibration_method == 'temp_beta':
                    base_prob_pred = prob_pred.copy()
                    beta_prob_pred, prob_pred = self._calibrate_proba(prob_pred)
                else:
                    base_prob_pred = prob_pred.copy()
                    prob_pred = self._calibrate_proba(prob_pred)

            feature_importances = self.model.feature_importances_
            sorted_idx = np.argsort(feature_importances)[::-1]
            top_features = X.columns[sorted_idx[:self.n_top_features]]

            X_meta = X[top_features].copy()

            if self.use_meta_labeling:
                X_meta = X_meta.loc[predictions == 1]
                prob_pred = prob_pred[predictions == 1]

                if self.calibration_method == 'beta_temp':
                    base_prob_pred = base_prob_pred[predictions == 1]
                    temp_prob_pred = temp_prob_pred[predictions == 1]
                elif self.calibration_method == 'temp_beta':
                    base_prob_pred = base_prob_pred[predictions == 1]
                    beta_prob_pred = beta_prob_pred[predictions == 1]
                elif self.calibration_method is not None:
                    base_prob_pred = base_prob_pred[predictions == 1]

            if self.use_cudf:
                import cudf

                if self.calibration_method == 'beta_temp':
                    X_meta["base_model's_probs"] = cudf.Series(
                        base_prob_pred[:, 1], index=X_meta.index
                    )
                    X_meta["base_model's_beta_probs"] = cudf.Series(
                        prob_pred[:, 1], index=X_meta.index
                    )
                    X_meta["base_model's_temp_probs"] = cudf.Series(
                        temp_prob_pred[:, 1], index=X_meta.index
                    )
                elif self.calibration_method == 'temp_beta':
                    X_meta["base_model's_probs"] = cudf.Series(
                        base_prob_pred[:, 1], index=X_meta.index
                    )
                    X_meta["base_model's_beta_probs"] = cudf.Series(
                        beta_prob_pred[:, 1], index=X_meta.index
                    )
                    X_meta["base_model's_temp_probs"] = cudf.Series(
                        prob_pred[:, 1], index=X_meta.index
                    )
                elif self.calibration_method is not None:
                    X_meta["base_model's_probs"] = cudf.Series(
                        base_prob_pred[:, 1], index=X_meta.index
                    )
                    X_meta["base_model's_calib_probs"] = cudf.Series(
                        prob_pred[:, 1], index=X_meta.index
                    )
                else:
                    X_meta["base_model's_probs"] = cudf.Series(
                        prob_pred[:, 1], index=X_meta.index
                    )
            else:
                if self.calibration_method == 'beta_temp':
                    X_meta["base_model's_probs"] = base_prob_pred[:, 1]
                    X_meta["base_model's_beta_probs"] = prob_pred[:, 1]
                    X_meta["base_model's_temp_probs"] = temp_prob_pred[:, 1]
                elif self.calibration_method == 'temp_beta':
                    X_meta["base_model's_probs"] = base_prob_pred[:, 1]
                    X_meta["base_model's_beta_probs"] = beta_prob_pred[:, 1]
                    X_meta["base_model's_temp_probs"] = prob_pred[:, 1]
                elif self.calibration_method is not None:
                    X_meta["base_model's_probs"] = base_prob_pred[:, 1]
                    X_meta["base_model's_calib_probs"] = prob_pred[:, 1]
                else:
                    X_meta["base_model's_probs"] = prob_pred[:, 1]

            meta_probs = self.meta_model.predict_proba(X_meta)

            if self.calibrate_meta and self.calibration_method is not None:
                if len(y) == 0:
                    raise ValueError("Input data y cannot be empty.")

                if self.calibration_method == "venn_abers":
                    if self.use_cudf:
                        meta_probs = self._calibrate_proba(X_meta.to_pandas(), model="meta")
                    else:
                        meta_probs = self._calibrate_proba(X_meta, model="meta")
                else:
                    if self.use_cudf:
                        self._apply_calibration(meta_probs, (predictions == addi_y).astype(int), model="meta")
                    else:
                        self._apply_calibration(meta_probs, (predictions == y).astype(int), model="meta")
                    if self.calibration_method == 'beta_temp':
                        meta_probs, _ = self._calibrate_proba(meta_probs, model="meta")
                    elif self.calibration_method == 'temp_beta':
                        _, meta_probs = self._calibrate_proba(meta_probs, model="meta")
                    else:
                        meta_probs = self._calibrate_proba(meta_probs, model="meta")

            del X_meta
            meta_pos_probs = meta_probs[:, 1]

            confidence_levels = np.array(confidence_levels)
            confidence_levels = np.sort(confidence_levels)[::-1]
            confidence_level_range = np.zeros_like(meta_pos_probs)

            for i, prob in enumerate(meta_pos_probs):
                valid_levels = confidence_levels[confidence_levels <= prob]
                confidence_level_range[i] = np.max(valid_levels) if valid_levels.size > 0 else 0.0

        else:
            # Ensure confidence levels are sorted in descending order and get its length
            confidence_levels = sorted(confidence_levels, reverse=True)
            confidence_levels_len = len(confidence_levels) - 1

            # Store prediction sets for each confidence level
            prediction_sets_per_level = []

            for cl in confidence_levels:
                predictions, prediction_sets = self.predict(
                    X, None, "test", addi_X=addi_X, addi_y=addi_y, confidence_level=cl
                )
                prediction_sets_per_level.append(prediction_sets)

            # Determine confidence range for each prediction
            confidence_level_range = []

            for i, pred in enumerate(predictions):
                for j, cl in enumerate(confidence_levels):
                    prediction_set = prediction_sets_per_level[j][i]
                    pred_set_len = len(prediction_set)
                    if pred not in prediction_set:
                        if pred_set_len > 0:
                            confidence_level_range.append(0.0)
                        else:
                            if j == 0:
                                confidence_level_range.append(0.0)
                            else:
                                confidence_level_range.append(cl)
                        break
                    if pred_set_len == 1:
                        confidence_level_range.append(cl)
                        break
                    if j == confidence_levels_len:
                        confidence_level_range.append(cl)

            confidence_level_range = np.array(confidence_level_range)

        return predictions, confidence_level_range

    def predict_proba(self, X, addi_X=None, y=None, stacked_model_trained=True):
        """
        Just for compatibility"""
        if self.apply_calibs:
            if self.calibration_method == "venn_abers":
                if self.use_cudf:
                    prob_pred = self._calibrate_proba(addi_X)
                else:
                    prob_pred = self._calibrate_proba(X)
            elif self.calibration_method == "beta_temp":
                prob_pred, _ = self._calibrate_proba(self.model.predict_proba(X))
            elif self.calibration_method == "temp_beta":
                _, prob_pred = self._calibrate_proba(self.model.predict_proba(X))
            else:
                prob_pred = self._calibrate_proba(self.model.predict_proba(X))
        else:
            if not stacked_model_trained:
                self.model.predict_proba(X, y=y, stacked_model_trained=False)

                return
            else:
                prob_pred = self.model.predict_proba(X)

        return prob_pred


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
    parameters = {
        param: value for param, value in all_params.items() if param in valid_params
    }

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
            "XGB" : XGBClassifier,
            "XGBF": XGBForestClassifier,
            "XGBF+": StackedXGBForestClassifier,
            "LGBM": LGBMClassifier,
            "CF-RF": ClassificationConformalPredictor,
            "CF-XGB": ClassificationConformalPredictor,
            "CF-XGBF": ClassificationConformalPredictor,
            "CF-XGBF+": ClassificationConformalPredictor,
            "CF-LGBM": ClassificationConformalPredictor,
        }

    model_class = model_mapping.get(model)
    if not model_class or not issubclass(model_class, TrainableModel):
        raise TypeError(
            "Make sure the model has the following methods: "
            "`fit`, `predict` and `predict_proba`"
        )

    # Conformal Prediction-based models
    if model.startswith("CF-"):
        all_params = man_params["parameters"] if manual else dict(wandb.config)
        parameters = set_model_parameters(model_class, all_params)
        sec_model_class = model_mapping.get(model.split('-')[1])

        if not sec_model_class or not issubclass(sec_model_class, TrainableModel):
            raise TypeError(
                "Make sure the secondary model has the following methods: "
                "`fit`, `predict` and `predict_proba`"
            )

        sec_parameters = set_model_parameters(sec_model_class, all_params)
        parameters["model"] = sec_model_class(**sec_parameters)
    else:
        all_params = man_params["parameters"] if manual else dict(wandb.config)
        parameters = set_model_parameters(model_class, all_params)

    print(f'Final Parameters of the Model: {parameters}')
    clf = model_class(**parameters)

    return clf
