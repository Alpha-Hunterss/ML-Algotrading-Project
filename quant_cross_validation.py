from sklearn.model_selection import TimeSeriesSplit
import time
import pandas as pd
import numpy as np
import gc
import threading
from sklearn.utils import parallel_backend
from joblib import Parallel, delayed
from .utils.evaluation_utils import cal_eval
from .backtest_funcs import do_backtest

def split_time_series(
    df_all: pd.DataFrame,
    max_train_size: int,
    n_splits: int,
    test_size: int,
    train_test_gap: int,
    eval_set_ratio: float = 0.4,
):
    """
    Return a nested dictionary with fold number as key and train, valid, test dates as values.
    :param max_train_size: Maximum size for train
    :param n_splits: Number of cross-validation folds
    :param test_size: Size of test set
    :param train_test_gap: Gap between train and test sets
    :param eval_set_ratio: Ratio of train set used for validation
    """
    all_dates = df_all.index.unique()  # Assumes DatetimeIndex from ETL
    all_dates = all_dates.sort_values()  # Ensure chronological order
    tscv = TimeSeriesSplit(
        gap=train_test_gap,
        max_train_size=max_train_size,
        n_splits=n_splits,
        test_size=test_size * 2,  # Test + valid
    )
    folds = {}
    for i, (train_index, test_valid_index) in enumerate(tscv.split(all_dates)):
        train_dates = all_dates[train_index]
        split_idx = int(len(train_dates) * (1.0 - eval_set_ratio))  # Train vs. valid split
        folds[i] = {
            "pre_eval_dates": train_dates[:split_idx],  # Train
            "eval_dates": train_dates[split_idx:],      # Valid (no 10*276 skip)
            "test_dates": all_dates[test_valid_index[test_size:]],  # Test
        }
    return folds



def quant_CV(
    df: pd.DataFrame,
    folds: dict[int, pd.DatetimeIndex],
    model,
    model_name,
    target_symbol,
    use_cudf,
    cnf_levels,
    initial_balance: int,
    accounts_leverage: int,
    default_volume: float,
    default_spread: int,
    early_stopping_rounds: int | None,
    df_raw_backtest: pd.DataFrame,
    bt_column_name: str,
    non_feature_columns: list[str],
    swap_rate: float,
    stop_loss: int,
    use_money_management: bool,
    n_max_OP: int,
    max_floating_dd: float,
    max_daily_dd: float,
    use_floating_risk: bool,
    use_dynamic_sl: bool,
    max_strg_sl_dynamic_perc: int,
    trade_mode: str,
    close_positions_at_midnight: bool,
    use_perc_levels: bool,
):
    """
    This function runs Time Series CV with available embargo/purge.
    It also backtests model signals on each fold and the whole test and valid sets.
    """
    evals = pd.DataFrame(
        columns=[
            "dataset",
            "K",
            "f1_score",
            "precision",
            "recall",
            "TP",
            "FP",
            "TN",
            "FN",
            "Min_date",
            "Max_date",
            "train_duration",
            "profit_percent",
            "max_dd",
            "sortino",
            "win_rate(%)",
            "max_exp_daily_dd",
            "max_overall_dd",
            "n_unique_days",
            "n_max_daily_sig",
            "max_n_open_position",
            "max_vol_open_positions",
            "no_iters_exceeding_dd",
        ]
    )
    df["pred_as_val"] = -1
    df["pred_val_proba"] = -1
    df["pred_as_test"] = -1
    df["pred_test_proba"] = -1
    df["confidence_levels"] = 0.0
    df["K"] = -1

    input_cols = [col for col in df.columns if col not in non_feature_columns]
    expected_feature_count = len(df.columns) - len(non_feature_columns)
    print(f"quant_CV input_cols count: {len(input_cols)}")
    print(f"Expected feature count (total columns - non_feature_columns): {expected_feature_count}")
    if len(input_cols) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} input columns, got {len(input_cols)}"
        )

    feature_importances = {feature: [] for feature in input_cols}
    is_cf_model = model_name.startswith("CF-")
    is_ensemble_xgbf_model = "XGBF+" in model_name

    if "XGB" in model_name:
        if is_cf_model:
            if getattr(model.model, "device", None) != "cuda" and use_cudf:
                raise ValueError("CuDF dataframes are useful only if `device='cuda'`.")
        else:
            if getattr(model, "device", None) != "cuda" and use_cudf:
                raise ValueError("CuDF dataframes are useful only if `device='cuda'`.")
    else:
        if use_cudf:
            raise ValueError("Non-XGB models do not support CuDF dataframes.")

    if use_cudf:
        import cudf
        cudf_df = cudf.from_pandas(df)
        for col in cudf_df.columns:
            if cudf_df[col].dtype == "bool":
                cudf_df[col] = cudf_df[col].astype("int8")
        print(f"cudf_df index sample: {cudf_df.index[:5].to_arrow().to_pylist()}")
    else:
        cudf_df = df  # Use pandas if not cudf

    general_backtest_df = {}

    for i in list(folds.keys()):
        print(f"Fold {i}:")
        tic = time.time()

        # Pre-compute datasets with input_cols
        train_data = cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())][input_cols]
        eval_data = cudf_df.loc[cudf_df.index.isin(folds[i]["eval_dates"].to_list())][input_cols]
        test_data = cudf_df.loc[cudf_df.index.isin(folds[i]["test_dates"].to_list())][input_cols]

        print(f"--> fold train size: {train_data.shape}")
        print(f"--> fold valid size: {eval_data.shape}")
        print(f"--> fold test size: {test_data.shape}")

        if train_data.shape[0] == 0:
            raise ValueError(f"Fold {i}: train_data has 0 rows")
        if eval_data.shape[0] == 0:
            raise ValueError(f"Fold {i}: eval_data has 0 rows")
        if test_data.shape[0] == 0:
            raise ValueError(f"Fold {i}: test_data has 0 rows")
        if eval_data.shape[1] != expected_feature_count:
            raise ValueError(
                f"Fold {i}: eval_data has {eval_data.shape[1]} columns, expected {expected_feature_count}"
            )

        train_min_max = [folds[i]["pre_eval_dates"].min(), folds[i]["pre_eval_dates"].max()]
        valid_min_max = [folds[i]["eval_dates"].min(), folds[i]["eval_dates"].max()]
        test_min_max = [folds[i]["test_dates"].min(), folds[i]["test_dates"].max()]
        min_max_dates = {
            "pre_eval_dates": train_min_max,
            "eval_dates": valid_min_max,
            "test_dates": test_min_max,
        }

        if is_ensemble_xgbf_model:
            if use_cudf:
                if early_stopping_rounds is not None:
                    print("early_stopping_rounds: ", early_stopping_rounds)
                    eval_set = [(eval_data, cudf_df.loc[cudf_df.index.isin(folds[i]["eval_dates"].to_list())]["target"])]
                    if is_cf_model:
                        model.fit(
                            train_data,
                            cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())]["target"],
                            addi_X=df.loc[folds[i]["pre_eval_dates"]][input_cols],
                            addi_y=df.loc[folds[i]["pre_eval_dates"]]["target"],
                            use_cudf=use_cudf,
                        )
                    else:
                        model.fit(
                            train_data,
                            cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())]["target"],
                            eval_set=eval_set,
                            verbose=False,
                        )
                else:
                    if is_cf_model:
                        model.fit(
                            train_data,
                            cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())]["target"],
                            addi_X=df.loc[folds[i]["pre_eval_dates"]][input_cols],
                            addi_y=df.loc[folds[i]["pre_eval_dates"]]["target"],
                            use_cudf=use_cudf,
                        )
                    else:
                        model.fit(
                            train_data,
                            cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())]["target"],
                        )

                # Predict validation set
                print(f"Fold {i} eval_data shape before predict_proba: {eval_data.shape}")
                model.predict_proba(
                    eval_data,
                    y=cudf_df.loc[cudf_df.index.isin(folds[i]["eval_dates"].to_list())]["target"],
                    stacked_model_trained=False,
                )
            else:
                if early_stopping_rounds is not None:
                    print("early_stopping_rounds: ", early_stopping_rounds)
                    eval_set = [(df.loc[folds[i]["eval_dates"]][input_cols], df.loc[folds[i]["eval_dates"]]["target"])]
                    model.fit(
                        df.loc[folds[i]["pre_eval_dates"]][input_cols],
                        df.loc[folds[i]["pre_eval_dates"]]["target"],
                        eval_set=eval_set,
                        verbose=False,
                    )
                else:
                    model.fit(
                        df.loc[folds[i]["pre_eval_dates"]][input_cols],
                        df.loc[folds[i]["pre_eval_dates"]]["target"],
                    )

                print(f"Fold {i} eval_data shape before predict_proba: {df.loc[folds[i]['eval_dates']][input_cols].shape}")
                model.predict_proba(
                    df.loc[folds[i]["eval_dates"]][input_cols],
                    y=df.loc[folds[i]["eval_dates"]]["target"],
                    stacked_model_trained=False,
                )
        else:
            if use_cudf:
                if early_stopping_rounds is not None:
                    print("early_stopping_rounds: ", early_stopping_rounds)
                    eval_set = [(eval_data, cudf_df.loc[cudf_df.index.isin(folds[i]["eval_dates"].to_list())]["target"])]
                    if is_cf_model:
                        model.fit(
                            train_data,
                            cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())]["target"],
                            addi_X=df.loc[folds[i]["pre_eval_dates"]][input_cols],
                            addi_y=df.loc[folds[i]["pre_eval_dates"]]["target"],
                            use_cudf=use_cudf,
                        )
                    else:
                        model.fit(
                            train_data,
                            cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())]["target"],
                            eval_set=eval_set,
                            verbose=False,
                        )
                else:
                    if is_cf_model:
                        model.fit(
                            train_data,
                            cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())]["target"],
                            addi_X=df.loc[folds[i]["pre_eval_dates"]][input_cols],
                            addi_y=df.loc[folds[i]["pre_eval_dates"]]["target"],
                            use_cudf=use_cudf,
                        )
                    else:
                        model.fit(
                            train_data,
                            cudf_df.loc[cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())]["target"],
                        )
            else:
                if early_stopping_rounds is not None:
                    print("early_stopping_rounds: ", early_stopping_rounds)
                    eval_set = [(df.loc[folds[i]["eval_dates"]][input_cols], df.loc[folds[i]["eval_dates"]]["target"])]
                    model.fit(
                        df.loc[folds[i]["pre_eval_dates"]][input_cols],
                        df.loc[folds[i]["pre_eval_dates"]]["target"],
                        eval_set=eval_set,
                        verbose=False,
                    )
                else:
                    model.fit(
                        df.loc[folds[i]["pre_eval_dates"]][input_cols],
                        df.loc[folds[i]["pre_eval_dates"]]["target"],
                    )

        try:
            if is_cf_model:
                input_cols = model.model.feature_names_in_
            else:
                input_cols = model.feature_names_in_
        except:
            if is_cf_model:
                input_cols = model.model.feature_name_
            else:
                input_cols = model.feature_name_

        # Store feature importances for this fold
        if is_cf_model:
            for feature, importance in zip(input_cols, model.model.feature_importances_):
                feature_importances[feature].append(importance)
        else:
            for feature, importance in zip(input_cols, model.feature_importances_):
                feature_importances[feature].append(importance)

        toc = time.time()
        gc.collect()

        # Evaluate on all sets
        for set_name in ["pre_eval_dates", "eval_dates", "test_dates"]:
            ping = time.time()
            set_name_dict = {
                "pre_eval_dates": "train",
                "eval_dates": "valid",
                "test_dates": "test",
            }
            if use_cudf:
                data = cudf_df.loc[cudf_df.index.isin(folds[i][set_name].to_list())][input_cols]
                print(f"Fold {i} {set_name} shape before predict: {data.shape}")
                if is_cf_model:
                    preds, _ = model.predict(
                        data,
                        cudf_df.loc[cudf_df.index.isin(folds[i][set_name].to_list())]["target"],
                        set_name_dict[set_name],
                        addi_X=df.loc[folds[i][set_name]][input_cols],
                        addi_y=df.loc[folds[i][set_name]]["target"],
                    )
                    y_pred = preds.reshape(-1, 1)
                else:
                    y_pred = model.predict(data).reshape(-1, 1)
            else:
                data = df.loc[folds[i][set_name]][input_cols]
                print(f"Fold {i} {set_name} shape before predict: {data.shape}")
                if is_cf_model:
                    preds, _ = model.predict(
                        data,
                        df.loc[folds[i][set_name]]["target"],
                        set_name_dict[set_name],
                    )
                    y_pred = preds.reshape(-1, 1)
                else:
                    y_pred = model.predict(data).reshape(-1, 1)

            y_real = df.loc[folds[i][set_name]][["target"]]

            if set_name in ["eval_dates", "test_dates"]:
                pred_name = {"eval_dates": "val", "test_dates": "test"}
                df.loc[folds[i][set_name], "K"] = i
                df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] = y_pred

                if use_cudf:
                    if is_cf_model:
                        if model.use_valid_as_calib and pred_name[set_name] == "test":
                            _, confidence_levels = model.categorize_proba(
                                data,
                                cudf_df.loc[cudf_df.index.isin(folds[i][set_name].to_list())]["target"],
                                cnf_levels,
                                addi_X=df.loc[folds[i][set_name]][input_cols],
                                addi_y=df.loc[folds[i][set_name]]["target"],
                            )
                        else:
                            confidence_levels = (
                                np.ones((len(y_pred[y_pred == 1]),), dtype=np.float16)
                                if model.use_meta_labeling
                                else np.ones((len(y_pred),), dtype=np.float16)
                            )
                        if not model.use_meta_labeling:
                            df.loc[folds[i][set_name], "confidence_levels"] = confidence_levels
                    else:
                        confidence_levels = np.ones((len(y_pred[y_pred == 1]),), dtype=np.float16)
                else:
                    if is_cf_model:
                        if model.use_valid_as_calib and pred_name[set_name] == "test":
                            _, confidence_levels = model.categorize_proba(
                                data,
                                df.loc[folds[i][set_name]]["target"],
                                cnf_levels,
                            )
                        else:
                            confidence_levels = (
                                np.ones((len(y_pred[y_pred == 1]),), dtype=np.float16)
                                if model.use_meta_labeling
                                else np.ones((len(y_pred),), dtype=np.float16)
                            )
                        if not model.use_meta_labeling:
                            df.loc[folds[i][set_name], "confidence_levels"] = confidence_levels
                    else:
                        confidence_levels = np.ones((len(y_pred[y_pred == 1]),), dtype=np.float16)

                fold_unique_days = pd.Series(
                    df.loc[folds[i][set_name]].loc[
                        df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] == 1
                    ].index.date
                ).nunique()

                fold_max_daily_sig = (
                    df.loc[folds[i][set_name]]
                    .loc[df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] == 1]
                    .groupby(pd.Grouper(freq="D"))
                    .size()
                    .max()
                )

                if is_cf_model:
                    bt_report, bt_df = do_backtest(
                        df_model_signal=df.loc[folds[i][set_name]]
                        .loc[df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] == 1][
                            [f"pred_as_{pred_name[set_name]}", "confidence_levels"]
                        ]
                        .rename(columns={f"pred_as_{pred_name[set_name]}": "model_prediction"}),
                        target_symbol=target_symbol,
                        spread=default_spread,
                        volume=default_volume,
                        initial_balance=initial_balance,
                        accounts_leverage=accounts_leverage,
                        df_raw_backtest=df_raw_backtest,
                        bt_column_name=bt_column_name,
                        swap_rate=swap_rate,
                        stop_loss=stop_loss,
                        use_money_management=use_money_management,
                        n_max_OP=n_max_OP,
                        max_floating_dd=max_floating_dd,
                        max_daily_dd=max_daily_dd,
                        use_floating_risk=use_floating_risk,
                        use_dynamic_sl=use_dynamic_sl,
                        max_strg_sl_dynamic_perc=max_strg_sl_dynamic_perc,
                        confidence_levels=confidence_levels,
                        model=model,
                        is_final_bt=False,
                        is_cf_model=True,
                        trade_mode=trade_mode,
                        close_positions_at_midnight=close_positions_at_midnight,
                        use_perc_levels=use_perc_levels,
                    )
                else:
                    bt_report, bt_df = do_backtest(
                        df_model_signal=df.loc[folds[i][set_name]]
                        .loc[df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] == 1][
                            [f"pred_as_{pred_name[set_name]}"]
                        ]
                        .rename(columns={f"pred_as_{pred_name[set_name]}": "model_prediction"}),
                        target_symbol=target_symbol,
                        spread=default_spread,
                        volume=default_volume,
                        initial_balance=initial_balance,
                        accounts_leverage=accounts_leverage,
                        df_raw_backtest=df_raw_backtest,
                        bt_column_name=bt_column_name,
                        swap_rate=swap_rate,
                        stop_loss=stop_loss,
                        use_money_management=use_money_management,
                        n_max_OP=n_max_OP,
                        max_floating_dd=max_floating_dd,
                        max_daily_dd=max_daily_dd,
                        use_floating_risk=use_floating_risk,
                        use_dynamic_sl=use_dynamic_sl,
                        max_strg_sl_dynamic_perc=max_strg_sl_dynamic_perc,
                        confidence_levels=confidence_levels,
                        model=model,
                        is_final_bt=False,
                        is_cf_model=False,
                        trade_mode=trade_mode,
                        close_positions_at_midnight=close_positions_at_midnight,
                        use_perc_levels=use_perc_levels,
                    )

                fold_profit_percent = bt_report["profit_percent"]
                fold_max_dd = bt_report["max_draw_down"]
                fold_sortino = bt_report["sortino"]
                fold_win_rate = bt_report["win_rate(%)"]
                fold_max_exp_daily_dd = bt_report["max_exp_daily_dd"]
                fold_max_overall_dd = bt_report["max_overall_dd"]
                fold_max_n_open_position = bt_report["max_n_open_position"]
                fold_max_vol_open_positions = bt_report["max_vol_open_positions"]
                fold_no_iters_exceeding_dd = bt_report["no_iters_exceeding_dd"]

                general_backtest_df.update({f"bt_df_fold{i}_{set_name}": bt_df})
                del bt_df, bt_report
                gc.collect()
            else:
                fold_profit_percent = None
                fold_max_dd = None
                fold_sortino = None
                fold_win_rate = None
                fold_max_exp_daily_dd = None
                fold_max_overall_dd = None
                fold_unique_days = None
                fold_max_daily_sig = None
                fold_max_n_open_position = None
                fold_max_vol_open_positions = None
                fold_no_iters_exceeding_dd = None

            pong = time.time()
            time_taken = (
                f"{round(toc - tic, 1)} + {round(pong - ping, 1)}"
                if set_name == "pre_eval_dates"
                else str(round(pong - ping, 1))
            )

            eval_list = (
                [set_name_dict[set_name], i]
                + cal_eval(y_real=y_real, y_pred=y_pred)
                + min_max_dates[set_name]
                + [time_taken]
                + [fold_profit_percent, fold_max_dd]
                + [fold_sortino, fold_win_rate, fold_max_exp_daily_dd]
                + [fold_max_overall_dd, fold_unique_days, fold_max_daily_sig]
                + [fold_max_n_open_position]
                + [fold_max_vol_open_positions, fold_no_iters_exceeding_dd]
            )
            evals.loc[len(evals)] = eval_list

        with pd.option_context("display.max_columns", None):
            print(evals.iloc[-3:])

        input_cols_and_type = dict(df[input_cols].dtypes)

    # Backtest on the whole test & valid set
    general_backtest_report = {}
    for pred_name in ["val", "test"]:
        bt_report, bt_df = do_backtest(
            df_model_signal=df.loc[df[f"pred_as_{pred_name}"] == 1][[f"pred_as_{pred_name}"]].rename(
                columns={f"pred_as_{pred_name}": "model_prediction"}
            ),
            target_symbol=target_symbol,
            spread=default_spread,
            volume=default_volume,
            initial_balance=initial_balance,
            accounts_leverage=accounts_leverage,
            df_raw_backtest=df_raw_backtest,
            bt_column_name=bt_column_name,
            swap_rate=swap_rate,
            stop_loss=stop_loss,
            use_money_management=use_money_management,
            n_max_OP=n_max_OP,
            max_floating_dd=max_floating_dd,
            max_daily_dd=max_daily_dd,
            use_floating_risk=use_floating_risk,
            use_dynamic_sl=use_dynamic_sl,
            max_strg_sl_dynamic_perc=max_strg_sl_dynamic_perc,
            confidence_levels=confidence_levels,
            model=model,
            is_final_bt=True,
            is_cf_model=is_cf_model,
            trade_mode=trade_mode,
            close_positions_at_midnight=close_positions_at_midnight,
            use_perc_levels=use_perc_levels,
        )
        general_backtest_report[f"profit_percent_{pred_name}"] = bt_report["profit_percent"]
        general_backtest_report[f"max_dd_{pred_name}"] = bt_report["max_draw_down"]
        general_backtest_df[f"bt_df_{pred_name}"] = bt_df

    print("CV loop ends")
    print(general_backtest_report)

    # Create a DataFrame from the feature importances
    importance_df = pd.DataFrame(feature_importances)
    importance_df = importance_df.T.reset_index()
    importance_df.columns = ["feature_name"] + [f"importance_fold_{i}" for i in range(len(folds))]
    imp_cols = [f for f in importance_df if "importance_fold" in f]
    importance_df["mean_importance"] = importance_df[imp_cols].mean(axis=1)
    importance_df["median_importance"] = importance_df[imp_cols].median(axis=1)
    importance_df["std_importance"] = importance_df[imp_cols].std(axis=1)
    importance_df["cv"] = importance_df["std_importance"] / importance_df["mean_importance"]
    importance_df.sort_values("mean_importance", ascending=False, inplace=True)

    return (
        input_cols_and_type,
        input_cols,
        evals,
        df[df.pred_as_val != -1][["K", "pred_as_val", "pred_val_proba", "target"]],
        df[df.pred_as_test != -1][["K", "pred_as_test", "pred_test_proba", "target"]],
        general_backtest_report,
        importance_df,
        general_backtest_df,
    )