import pandas as pd
from models import TrainableModel

def train_model_to_save(
    df: pd.DataFrame,
    final_clf: TrainableModel,
    max_train_size: int,
    save_model_mode: str|None,
    non_feature_columns: list[str],
    eval_set_ratio: float,
    model_name: str,
):
    """
    This function is used for retrainig the model on specific set of data specifically the most recent historical data
        df: the historical dataset,
        final_clf: classification model,
        max_train_size: maximum number of training samples,
        save_model_mode: types of trainig set: 
            'sample_train_size': Use all historical data and with max_train_size as sample size 
            'last_train_size': Use the most recent max_train_size 
            'all_data': Use all historical data
            None: The function will return none
    # 
    """

    is_ensemble_xgbf_model = "XGBF+" in model_name

    if "confidence_levels" in df.columns:
        df = df.drop(columns=["confidence_levels"], errors="ignore")

    if "confidence_levels" in df.columns:
        raise ValueError(
            "The model's input dataframe contains the irrelevant column 'confidence_levels'."
        )

    if save_model_mode == "last_train_size":
        final_clf.fit(
            df.iloc[-max_train_size:].drop(
                columns=non_feature_columns
            ),
            df.iloc[-max_train_size:]["target"],
        )

    elif save_model_mode == "sample_train_size":
        sample_idx = df.sample(max_train_size).index
        final_clf.fit(
            df.loc[sample_idx].drop(
                columns=non_feature_columns
            ),
            df.loc[sample_idx]["target"],
        )

    elif save_model_mode == "all_data":
        if is_ensemble_xgbf_model:
            df.sort_values(["_time"])
            split_idx = int(len(df) * (1-eval_set_ratio))
            pre_eval_df = df.iloc[:split_idx]
            eval_df = df.iloc[split_idx:]

            final_clf.fit(
                pre_eval_df.drop(columns=non_feature_columns),
                pre_eval_df["target"],
            )

            final_clf.predict_proba(
                eval_df.drop(columns=non_feature_columns),
                eval_df["target"],
                stacked_model_trained=False,
            )
        else:
            final_clf.fit(
                df.drop(columns=non_feature_columns),
                df["target"],
            )
    else:
        final_clf = None

    return final_clf
