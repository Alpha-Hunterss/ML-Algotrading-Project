from numbers import Integral, Real
from typing import Protocol, Any, Type
import inspect
import threading
import wandb
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, BaseEnsemble
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.parallel import Parallel, delayed
from scipy.sparse import issparse
from joblib import effective_n_jobs
from warnings import catch_warnings, simplefilter
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing_extensions import runtime_checkable
import numpy as np


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

    def fit(self, X, y, sample_weight=None):
        """
        Fit the XGBForestClassifier. samples_
        """
        self._validate_params()

        # Check for sparse `y`
        if issparse(y):
            raise ValueError("Sparse multilabel-indicator for y is not supported.")

        # Validate input data
        X, y = self._validate_data(
            X, y, multi_output=False, accept_sparse="csc", dtype=np.float32
        )

        # Validate sample weights
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Transform `y` appropriately
        y = np.atleast_1d(y)
        if y.ndim == 2:
            if y.shape[1] > 1:
                raise ValueError("XGBoost doesn't support multi-output target data.")
            if y.shape[1] == 1:
                y = np.ravel(y)  # Flatten to (n_samples,)

        # Validate Poisson objective requirements
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
        y, expanded_class_weight = self._validate_y_class_weight(y)

        # Ensure `y` is contiguous and of correct dtype
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
                        curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
                elif self.class_weight == "balanced_subsample":
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
            "XGB" : XGBClassifier,
            "XGBF": XGBForestClassifier,
            "LGBM": LGBMClassifier,
        }
    model_class = model_mapping.get(model)
    if not model_class or not issubclass(model_class, TrainableModel):
        raise TypeError(
            "Make sure the model has the following methods: `fit`, `predict` and `predict_proba`"
        )

    all_params = man_params["parameters"] if manual else dict(wandb.config)
    parameters = set_model_parameters(model_class, all_params)

    print(f'Final Parameters of the Model: {parameters}')
    clf = model_class(**parameters)

    return clf
