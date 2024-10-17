import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def eval_summerize_dict(evals, cal_mode="exclude_zero_preds"):
    """
    get evals dataFrame and returns aggrigated dictionary of some columns
    """

    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    cols = ["precision", "recall", "TP", "FP", "profit_percent", "max_dd"]

    f = {}
    for col in cols:
        f[col] = ["mean", "std", q1, "median", q3, "min"]

    if cal_mode == "exclude_zero_preds":
        eval_grouped = (
            evals.loc[~((evals.precision == 0) & (evals.FP == 0))]
            .groupby(["dataset"])
            .agg(f)
        )

    else:
        eval_grouped = evals.groupby(["dataset"]).agg(f)

    eval_grouped.columns = [
        "_".join(col).strip() for col in eval_grouped.columns.values
    ]
    eval_grouped_dict = eval_grouped.to_dict("index")

    return eval_grouped_dict


def cal_eval(y_real, y_pred):
    """
    calculate and return evaluation of class 1

    """
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    clf_report = classification_report(
        y_real, y_pred, output_dict=True, digits=0, zero_division=0
    )

    if "1" not in clf_report.keys():
        print("NO CLASS1 PREDICTION !")
        clf_report["1"] = {"f1-score": 0.0, "precision": 0.0, "recall": 0.0}

    class_1_report = clf_report["1"]
    for k, v in class_1_report.items():
        class_1_report[k] = round(v, 2)

    eval_list = [
        class_1_report["f1-score"],
        class_1_report["precision"],
        class_1_report["recall"],
        tp,
        fp,
        tn,
        fn,
    ]
    return eval_list
