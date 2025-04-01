import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import time, gc
from utils.general_utils import cal_eval
import numpy as np
from backtest_funcs import do_backtest

def split_time_series(
    df_all: pd.DataFrame,
    max_train_size: int,
    n_splits: int,
    test_size: int,
    train_test_gap: int,
    eval_set_ratio: float = 0.4,
):
    """
    Return a nested dictionary key is k number and value is dicitonary of train, valid and test Dates
    :max_train_size: maximum size we for train
    :n_splits: K in cross-folds
    :test_size: test size
    train_test_gap is the gap between train and valid/test sets 
    """
    all_dates = df_all.index.get_level_values("_time").unique().sort_values(["_time"])
    tscv = TimeSeriesSplit(
        gap=train_test_gap,
        max_train_size=max_train_size,
        n_splits=n_splits,
        test_size=test_size*2,
    )
    folds = {}
    for i, (train_index, test_valid_index) in enumerate(tscv.split(all_dates[0])):
        train_dates = all_dates[0][train_index]

        # Calculate split point for pre_eval and eval (60%-40%)
        split_idx = int(len(train_dates) * (1.0-eval_set_ratio))

        folds[i] = {
            "train_dates": train_dates,
            "pre_eval_dates": train_dates[:split_idx],
            "eval_dates": train_dates[split_idx+(10*276):],
            "valid_dates": all_dates[0][test_valid_index[:test_size]],
            "test_dates": all_dates[0][test_valid_index[test_size:]],
        }

    return folds

def quant_CV(
    df: pd.DataFrame,
    folds: dict[int,pd.DatetimeIndex],
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
    This function runs Time Series CV with available embargo/purge 
    It also backtest model signals on each fold and the whole test and valid sets 
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
            # "meta_model_pos_label_perc",
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

    the_features = df.drop(columns=non_feature_columns).columns
    feature_importances = {feature: [] for feature in the_features}
    is_cf_model = model_name.startswith("CF-")
    is_ensemble_xgbf_model = "XGBF+" in model_name

    if "XGB" in model_name:
        if is_cf_model:
            if getattr(model.model, "device") != "cuda" and use_cudf:
                raise ValueError("CuDF dataframes are useful only if `device='cuda'`.")
        else:
            if getattr(model, "device") != "cuda" and use_cudf:
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

    general_backtest_df = {}

    for i in list(folds.keys()):
        print(f"Fold {i}:")
        tic = time.time()
        # sets,min_max_dates = data_split_loader(df,folds,i)

        train_min_max = [folds[i]["train_dates"].min(), folds[i]["train_dates"].max()]
        valid_min_max = [folds[i]["valid_dates"].min(), folds[i]["valid_dates"].max()]
        test_min_max = [folds[i]["test_dates"].min(), folds[i]["test_dates"].max()]
        min_max_dates = {
            "train_dates": train_min_max,
            "valid_dates": valid_min_max,
            "test_dates": test_min_max,
        }

        print(f"--> fold train size: {df.loc[folds[i]['train_dates']].shape}")
        print(f"--> fold valid size: {df.loc[folds[i]['valid_dates']].shape}")
        print(f"--> fold test size: {df.loc[folds[i]['test_dates']].shape}")

        if "confidence_levels" in cudf_df.columns or ("confidence_levels" in df.columns):
            df = df.drop(columns=["confidence_levels"], errors="ignore")
            if "confidence_levels" in cudf_df.columns:
                cudf_df = cudf_df.drop(columns=["confidence_levels"])

        if "confidence_levels" in cudf_df.columns or ("confidence_levels" in df.columns):
            raise ValueError(
                "The model's input dataframe contains the irrelevant column 'confidence_levels'."
            )

        if is_ensemble_xgbf_model:
            if use_cudf:
                if early_stopping_rounds is not None:
                    print("early_stopping_rounds: ", early_stopping_rounds)

                    eval_set = [
                        (
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["eval_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["eval_dates"].to_list())
                            ]["target"],
                        )
                    ]

                    if is_cf_model:
                        model.fit(
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())
                            ]["target"],
                            addi_X=df.loc[folds[i]["pre_eval_dates"]].drop(
                                columns=non_feature_columns
                            ),
                            addi_y=df.loc[folds[i]["pre_eval_dates"]]["target"],
                            use_cudf=use_cudf,
                        )
                    else:
                        model.fit(
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())
                            ]["target"],
                            eval_set=eval_set,
                            verbose = False,
                        )
                else:
                    if is_cf_model:
                        model.fit(
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())
                            ]["target"],
                            addi_X=df.loc[folds[i]["pre_eval_dates"]].drop(
                                columns=non_feature_columns
                            ),
                            addi_y=df.loc[folds[i]["pre_eval_dates"]]["target"],
                            use_cudf=use_cudf,
                        )
                    else:
                        model.fit(
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["pre_eval_dates"].to_list())
                            ]["target"],
                        )

                model.predict_proba(
                    cudf_df.loc[
                        cudf_df.index.isin(folds[i]["eval_dates"].to_list())
                    ].drop(
                        columns=non_feature_columns
                    ),
                    y=cudf_df.loc[
                        cudf_df.index.isin(folds[i]["eval_dates"].to_list())
                    ]["target"],
                    stacked_model_trained=False,
                )

            else:
                if early_stopping_rounds is not None:
                    print("early_stopping_rounds: ", early_stopping_rounds)

                    eval_set = [
                        (
                            df.loc[folds[i]["eval_dates"]].drop(
                                columns=non_feature_columns
                            ),
                            df.loc[folds[i]["eval_dates"]]["target"],
                        )
                    ]

                    model.fit(
                        df.loc[folds[i]["pre_eval_dates"]].drop(
                            columns=non_feature_columns
                        ),
                        df.loc[folds[i]["pre_eval_dates"]]["target"],
                        eval_set=eval_set,
                        verbose = False,
                    )
                else:
                    model.fit(
                        df.loc[folds[i]["pre_eval_dates"]].drop(
                            columns=non_feature_columns
                        ),
                        df.loc[folds[i]["pre_eval_dates"]]["target"],
                    )

                model.predict_proba(
                    df.loc[folds[i]["eval_dates"]].drop(
                        columns=non_feature_columns
                    ),
                    y=df.loc[folds[i]["eval_dates"]]["target"],
                    stacked_model_trained=False,
                )

        else:
            if use_cudf:
                if early_stopping_rounds is not None:
                    print("early_stopping_rounds: ", early_stopping_rounds)

                    eval_set = [
                        (
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["valid_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["valid_dates"].to_list())
                            ]["target"],
                        )
                    ]

                    if is_cf_model:
                        model.fit(
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["train_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["train_dates"].to_list())
                            ]["target"],
                            addi_X=df.loc[folds[i]["train_dates"]].drop(
                                columns=non_feature_columns
                            ),
                            addi_y=df.loc[folds[i]["train_dates"]]["target"],
                            use_cudf=use_cudf,
                        )
                    else:
                        model.fit(
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["train_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["train_dates"].to_list())
                            ]["target"],
                            eval_set=eval_set,
                            verbose = False,
                        )
                else:
                    if is_cf_model:
                        model.fit(
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["train_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["train_dates"].to_list())
                            ]["target"],
                            addi_X=df.loc[folds[i]["train_dates"]].drop(
                                columns=non_feature_columns
                            ),
                            addi_y=df.loc[folds[i]["train_dates"]]["target"],
                            use_cudf=use_cudf,
                        )
                    else:
                        model.fit(
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["train_dates"].to_list())
                            ].drop(
                                columns=non_feature_columns
                            ),
                            cudf_df.loc[
                                cudf_df.index.isin(folds[i]["train_dates"].to_list())
                            ]["target"],
                        )

            else:
                if early_stopping_rounds is not None:
                    print("early_stopping_rounds: ", early_stopping_rounds)

                    eval_set = [
                        (
                            df.loc[folds[i]["valid_dates"]].drop(
                                columns=non_feature_columns
                            ),
                            df.loc[folds[i]["valid_dates"]]["target"],
                        )
                    ]

                    model.fit(
                        df.loc[folds[i]["train_dates"]].drop(
                            columns=non_feature_columns
                        ),
                        df.loc[folds[i]["train_dates"]]["target"],
                        eval_set=eval_set,
                        verbose = False,
                    )
                else:
                    model.fit(
                        df.loc[folds[i]["train_dates"]].drop(
                            columns=non_feature_columns
                        ),
                        df.loc[folds[i]["train_dates"]]["target"],
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
        # repetetive part I can improve by a function
        for set_name in ["train_dates", "valid_dates", "test_dates"]:
            ping = time.time()
            set_name_dict = {
                "train_dates": "train",
                "valid_dates": "valid",
                "test_dates": "test",
            }
            if use_cudf:
                if is_cf_model:
                    preds, _ = model.predict(
                        cudf_df.loc[
                            cudf_df.index.isin(folds[i][set_name].to_list())
                        ][input_cols],
                        cudf_df.loc[cudf_df.index.isin(folds[i][set_name].to_list())]["target"],
                        set_name_dict[set_name],
                        addi_X=df.loc[folds[i][set_name]][input_cols],
                        addi_y=df.loc[folds[i][set_name]]["target"]
                    )
                    y_pred = preds.reshape(-1, 1)
                else:
                    y_pred = model.predict(cudf_df.loc[
                        cudf_df.index.isin(folds[i][set_name].to_list())
                    ][input_cols]).reshape(
                        -1, 1
                    )
            else:
                if is_cf_model:
                    preds, _ = model.predict(
                        df.loc[folds[i][set_name]][input_cols],
                        df.loc[folds[i][set_name]]["target"],
                        set_name_dict[set_name]
                    )
                    y_pred = preds.reshape(-1, 1)
                else:
                    y_pred = model.predict(df.loc[folds[i][set_name]][input_cols]).reshape(
                        -1, 1
                    )

            y_real = df.loc[folds[i][set_name]][["target"]]

            if set_name in ["valid_dates", "test_dates"]:
                pred_name = {
                "valid_dates": "val",
                "test_dates": "test"}
                df.loc[folds[i][set_name], "K"] = i
                df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] = y_pred
                # if use_cudf:
                #     proba_pred = model.predict_proba(
                #         cudf_df.loc[cudf_df.index.isin(folds[i][set_name].to_list())][input_cols]
                #     )
                # else:
                #     proba_pred = model.predict_proba(df.loc[folds[i][set_name]][input_cols])

                cf_df = df.copy()

                if use_cudf:
                    if is_cf_model:
                        if model.use_valid_as_calib:
                            if pred_name[set_name] == "test":
                                _, confidence_levels = model.categorize_proba(
                                    cudf_df.loc[
                                        cudf_df.index.isin(folds[i][set_name].to_list())
                                    ][input_cols],
                                    cudf_df.loc[
                                        cudf_df.index.isin(folds[i][set_name].to_list())
                                    ]["target"],
                                    cnf_levels,
                                    addi_X=df.loc[folds[i][set_name]][input_cols],
                                    addi_y=df.loc[folds[i][set_name]]["target"]
                                )
                            else:
                                if model.use_meta_labeling:
                                    confidence_levels = np.ones((len(y_pred[y_pred == 1]),), dtype=np.float16)
                                else:
                                    confidence_levels = np.ones((len(y_pred),), dtype=np.float16)

                            if not model.use_meta_labeling:
                                cf_df.loc[
                                    folds[i][set_name],
                                    "confidence_levels"
                                ] = confidence_levels
                        else:
                            _, confidence_levels = model.categorize_proba(
                                cudf_df.loc[
                                    cudf_df.index.isin(folds[i][set_name].to_list())
                                ][input_cols],
                                cnf_levels,
                                addi_X=df.loc[folds[i][set_name]][input_cols]
                            )

                            if not model.use_meta_labeling:
                                cf_df.loc[
                                    folds[i][set_name],
                                    "confidence_levels"
                                ] = confidence_levels
                    else:
                        confidence_levels = np.ones((len(y_pred[y_pred == 1]),), dtype=np.float16)
                else:
                    if is_cf_model:
                        if model.use_valid_as_calib:
                            if pred_name[set_name] == "test":
                                _, confidence_levels = model.categorize_proba(
                                    df.loc[folds[i][set_name]][input_cols],
                                    df.loc[folds[i][set_name]]["target"],
                                    cnf_levels
                                )
                            else:
                                if model.use_meta_labeling:
                                    confidence_levels = np.ones((len(y_pred[y_pred == 1]),), dtype=np.float16)
                                else:
                                    confidence_levels = np.ones((len(y_pred),), dtype=np.float16)

                            if not model.use_meta_labeling:
                                cf_df.loc[
                                    folds[i][set_name],
                                    "confidence_levels"
                                ] = confidence_levels
                        else:
                            _, confidence_levels = model.categorize_proba(
                                df.loc[folds[i][set_name]][input_cols], cnf_levels
                            )

                            if not model.use_meta_labeling:
                                cf_df.loc[
                                    folds[i][set_name],
                                    "confidence_levels"
                                ] = confidence_levels
                    else:
                        confidence_levels = np.ones((len(y_pred[y_pred == 1]),), dtype=np.float16)

                # if np.shape(proba_pred)[1] > 1:
                #     df.loc[
                #         folds[i][set_name], f"pred_{pred_name[set_name]}_proba"
                #     ] = proba_pred[:, 1]
                # else:
                #     print("Proba doesn't have class1")
                #     df.loc[folds[i][set_name], f"pred_{pred_name[set_name]}_proba"] = 0

                # Calculate n_unique days and max daily n_signals in each fold
                fold_unique_days = pd.Series(df.loc[folds[i][set_name]].loc[
                    df.loc[folds[i][set_name],
                           f"pred_as_{pred_name[set_name]}"] == 1
                ].index.date).nunique()

                fold_max_daily_sig = df.loc[folds[i][set_name]].loc[
                    df.loc[folds[i][set_name],
                            f"pred_as_{pred_name[set_name]}"] == 1
                ].groupby(pd.Grouper(freq='D')).size().max()

                if is_cf_model:
                    #? Backtest
                    bt_report, bt_df = do_backtest(
                        df_model_signal = cf_df.loc[folds[i][set_name]].loc[
                                cf_df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] == 1
                        ][[f"pred_as_{pred_name[set_name]}", "confidence_levels"]].rename(
                                columns={f"pred_as_{pred_name[set_name]}":"model_prediction"}
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
                        is_final_bt=False,
                        is_cf_model=True,
                        trade_mode=trade_mode,
                        close_positions_at_midnight=close_positions_at_midnight,
                        use_perc_levels=use_perc_levels,
                    )
                else:
                    #? Backtest
                    bt_report, bt_df = do_backtest(
                        df_model_signal = cf_df.loc[folds[i][set_name]].loc[
                                cf_df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] == 1
                        ][[f"pred_as_{pred_name[set_name]}"]].rename(
                                columns={f"pred_as_{pred_name[set_name]}":"model_prediction"}
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
                        is_final_bt=False,
                        is_cf_model=False,
                        trade_mode=trade_mode,
                        close_positions_at_midnight=close_positions_at_midnight,
                        use_perc_levels=use_perc_levels,
                    )

                df = df.drop(columns=["confidence_levels"], errors="ignore")
                if "confidence_levels" in cudf_df.columns:
                    cudf_df = cudf_df.drop(columns=["confidence_levels"])

                fold_profit_percent = bt_report['profit_percent']
                fold_max_dd = bt_report['max_draw_down']
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

            if set_name == "train_dates":
                time_taken = f"{round(toc - tic, 1)} + {round(pong - ping, 1)}"
                # meta_model_pos_label_perc = None
            else:
                time_taken = str(round(pong - ping, 1))
                # if is_cf_model:
                #     if model.prob_estimator == "meta":
                #         meta_model_pos_label_perc = model.meta_pos_label_perc

            eval_list = (
                [set_name_dict[set_name], i]
                + cal_eval(y_real=y_real, y_pred=y_pred)
                + min_max_dates[set_name]
                + [time_taken]
                + [fold_profit_percent, fold_max_dd]
                + [fold_sortino, fold_win_rate, fold_max_exp_daily_dd]
                + [fold_max_overall_dd, fold_unique_days, fold_max_daily_sig]
                # + [meta_model_pos_label_perc]
                + [fold_max_n_open_position]
                + [fold_max_vol_open_positions, fold_no_iters_exceeding_dd]
            )

            evals.loc[len(evals)] = eval_list

        with pd.option_context('display.max_columns', None):
            print(evals.iloc[-3:])

        input_cols_and_type = dict(df[input_cols].dtypes)

    if "confidence_levels" in cudf_df.columns or ("confidence_levels" in df.columns):
        df = df.drop(columns=["confidence_levels"], errors="ignore")
        if "confidence_levels" in cudf_df.columns:
            cudf_df = cudf_df.drop(columns=["confidence_levels"])

    if "confidence_levels" in cudf_df.columns or ("confidence_levels" in df.columns):
        raise ValueError(
            "The model's input dataframe contains the irrelevant column 'confidence_levels'."
        )

    # Backtest on the whole test & valid set
    general_backtest_report = {}
    for pred_name in ["val", "test"]:
        bt_report, bt_df = do_backtest(
            df_model_signal = df.loc[df[f"pred_as_{pred_name}"] == 1][[f"pred_as_{pred_name}"]].rename(
                    columns={f"pred_as_{pred_name}":"model_prediction"}),
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
        general_backtest_report[f"profit_percent_{pred_name}"] = bt_report['profit_percent']
        general_backtest_report[f"max_dd_{pred_name}"] = bt_report['max_draw_down']

    print('CV loop ends')
    print(general_backtest_report)

    # # Create a DataFrame from the feature importances
    # importance_df = pd.DataFrame(feature_importances)
    # importance_df = importance_df.T.reset_index()
    # importance_df.columns = ['feature_name'] + [f'importance_fold_{i}' for i in range(len(folds))]

    # imp_cols = [f for f in importance_df if 'importance_fold' in f]
    # importance_df['mean_importance'] = importance_df[imp_cols].mean(axis=1)
    # importance_df['median_importance'] = importance_df[imp_cols].median(axis=1)
    # importance_df['std_importance'] = importance_df[imp_cols].std(axis=1)

    # # Calculate coefficient of variation (CV)
    # importance_df['cv'] = importance_df['std_importance'] / importance_df['mean_importance']
    # importance_df.sort_values('mean_importance', ascending=False, inplace=True)

    importance_df = 0

    return (
        input_cols_and_type,
        input_cols,
        evals,
        df[df.pred_as_val != -1][["K", "pred_as_val", "pred_val_proba", "target"]],
        df[df.pred_as_test != -1][["K", "pred_as_test", "pred_test_proba", "target"]],
        general_backtest_report,
        importance_df,
        general_backtest_df
    )
