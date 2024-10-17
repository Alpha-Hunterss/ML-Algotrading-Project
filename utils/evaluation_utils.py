import pandas as pd
from typing import Any

def get_models_evaluations(objects):
    """get dict of objects and return evaluations dict"""
    evaluations = {}
    for key in objects.keys():
        evaluations[key] = objects[key]["obj"].evals
    return evaluations


def create_conf_matrix(s):
    if (s["pred_as_val"] == 1) and (s["target"] == 1):
        return "TP"
    elif (s["pred_as_val"] == 0) and (s["target"] == 0):
        return "TN"
    elif (s["pred_as_val"] == 1) and (s["target"] == 0):
        return "FP"
    elif (s["pred_as_val"] == 0) and (s["target"] == 1):
        return "FN"
    else:
        return -1

def cal_aggregated_evals(evals_df: pd.DataFrame, set_name: str):

    mask = ~((evals_df.precision == 0) & (evals_df.FP == 0))
    evals_df_ex0 = evals_df.loc[mask]

    set_eval_dict = {
        
        f"signal_count_median_{set_name}" : evals_df.TP.median() + evals_df.FP.median(),
        f"signal_count_mean_{set_name}" : evals_df.TP.mean() + evals_df.FP.mean(),
        f"signal_count_min_{set_name}" : evals_df.TP.min() + evals_df.FP.min(),
        f"zero_signal_fold_count_{set_name}": evals_df.loc[~mask].precision.count(),

        f"precision_median_ex0_{set_name}": evals_df_ex0.precision.median(),
        f"precision_mean_ex0_{set_name}": evals_df_ex0.precision.mean(),
        f"precision_min_ex0_{set_name}": evals_df_ex0.precision.min(),
        f"precision_min_{set_name}" : evals_df.precision.min(),
        f"precision_total_{set_name}": evals_df.TP.sum() / (evals_df.TP.sum() + evals_df.FP.sum()),

        f"f1_score_ex0_{set_name}": evals_df_ex0.f1_score.mean(),
        f"recall_{set_name}": evals_df.recall.mean(),
    }
    if set_name != 'train':
        backtest_eval_dict = {
        f"profit_percent_median_{set_name}": evals_df.profit_percent.median(),
        f"profit_percent_mean_{set_name}": evals_df.profit_percent.mean(),
        f"profit_percent_min_{set_name}": evals_df.profit_percent.min(),

        f"max_dd_median_{set_name}": evals_df.max_dd.median(),
        f"max_dd_mean_{set_name}": evals_df.max_dd.mean(),
        f"max_dd_min_{set_name}": evals_df.max_dd.min(),

        f"n_unique_days_median_{set_name}": evals_df.n_unique_days.median(),
        f"n_unique_days_mean_{set_name}": evals_df.n_unique_days.mean(),
        f"n_unique_days_min_{set_name}": evals_df.n_unique_days.min(),
        f"n_unique_days_max_{set_name}": evals_df.n_unique_days.max(),

        f"n_max_daily_sig_median_{set_name}": evals_df.n_max_daily_sig.median(),
        f"n_max_daily_sig_mean_{set_name}": evals_df.n_max_daily_sig.mean(),
        f"n_max_daily_sig_min_{set_name}": evals_df.n_max_daily_sig.min(),
        f"n_max_daily_sig_max_{set_name}": evals_df.n_max_daily_sig.max(),
        }
        set_eval_dict.update(backtest_eval_dict)

    elif set_name == 'train':
        set_eval_dict.update({f"train_duration_{set_name}": int(evals_df.train_duration.astype("float").mean())})

    return set_eval_dict
