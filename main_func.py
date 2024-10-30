import os
import wandb
from models import model_func
from quant_cross_validation import split_time_series, quant_CV
from ETL import ETL
from save_model import train_model_to_save
import matplotlib.pyplot as plt
from utils.general_utils import eval_summerize_dict
from utils.wandb_utils import evals_logger
from utils.evaluation_utils import cal_aggregated_evals
from backtest_funcs import cal_backtest_on_raw_cndl
from experiment_tracker import QuantExpTracker
from datetime import datetime
import traceback
import gc
from configss.symbols_info import symbols_dict


def main(
    manual=False,
    man_params=None,
    dataset_path="/kaggle/working/ML-Algotrading-Project/dataset/data/dataset/dataset.parquet",
    C5M_data_path = "/kaggle/working/ML-Algotrading-Project/dataset/data/stage_one_data/",
):
    try:
        # _______________________________Get Inputs from Manual or W&B Config _______________________________
        if manual:
            the_config = man_params
            model_name = man_params["model_name"]
            # Target:
            target_symbol = man_params["target_symbol"]
            trade_mode = man_params["trade_mode"]
            trg_look_ahead = man_params["trg_look_ahead"]
            trg_take_profit = man_params["trg_take_profit"]
            trg_stop_loss = man_params["trg_stop_loss"]
            # Strategy:
            strg_look_ahead = man_params["strg_look_ahead"]
            strg_take_profit = man_params["strg_take_profit"]
            strg_stop_loss = man_params["strg_stop_loss"]
            # Output Model:
            save_model_mode = man_params["save_model_mode"]

            n_rand_features = man_params['n_rand_features']
            # Split Data & Time Series Cross Validation:
            n_splits = man_params["n_splits"]
            max_train_size = man_params["max_train_size"]
            test_size = man_params["test_size"]
            train_test_gap = man_params["train_test_gap"]
            try:
                early_stopping_rounds = man_params["paremeters"][
                    "early_stopping_rounds"
                ]
            except:
                early_stopping_rounds = None

        else:
            wandb.init()
            not_model_parameters = [
                "model_name",
                "target_symbol",
                "trade_mode",
                "trg_look_ahead",
                "trg_take_profit",
                "trg_stop_loss",
                "strg_look_ahead",
                "strg_take_profit",
                "strg_stop_loss",
                "save_model_mode",
                "feature_set",
                "imp_features",
                "drop_features_conf",
                "pca_n_components",
                "standardizing_features",
                "n_rand_features",
                "n_splits",
                "max_train_size",
                "test_size",
                "train_test_gap",
                "early_stopping_rounds",
            ]

            the_config = {param:value for param,value in dict(wandb.config).items() if param in not_model_parameters}
            the_config["parameters"] = {param:value for param,value in dict(wandb.config).items() if param not in not_model_parameters}
            model_name = wandb.config.model_name
            # Target:
            target_symbol = wandb.config.target_symbol
            trade_mode = wandb.config.trade_mode
            trg_look_ahead = wandb.config.trg_look_ahead
            trg_take_profit = wandb.config.trg_take_profit
            trg_stop_loss = wandb.config.trg_stop_loss
            # Strategy:
            strg_look_ahead = wandb.config.strg_look_ahead
            strg_take_profit = wandb.config.strg_take_profit
            strg_stop_loss = wandb.config.strg_stop_loss
            # Output Model:
            save_model_mode = wandb.config.save_model_mode
            # Feature Selection & Transform:
            n_rand_features = wandb.config.n_rand_features
            # Split Data & Time Series Cross Validation:
            n_splits = wandb.config.n_splits
            max_train_size = wandb.config.max_train_size
            test_size = wandb.config.test_size
            train_test_gap = wandb.config.train_test_gap
            early_stopping_rounds = wandb.config.early_stopping_rounds


        # _______________________________Get the Classification Model________________________________________
        clf = model_func(
            man_params=man_params, manual=manual, model=model_name
        )

        # _______________________________Read Data & ETL_____________________________________________________
        target_col = f"trg_clf_{trade_mode}_{target_symbol}_M{trg_look_ahead}_TP{trg_take_profit}_SL{trg_stop_loss}"

        # Check for dataset file existence
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found at: {dataset_path}")
            dataset_path = "/content/ML-Algotrading-Project/dataset/data/dataset/dataset.parquet"
            C5M_data_path = "/content/ML-Algotrading-Project/dataset/data/stage_one_data/"

        df_all = ETL(
            path =dataset_path,
            C5M_data_path=C5M_data_path,
            trade_mode = trade_mode,
            target_symbol =target_symbol,
            trg_look_ahead =trg_look_ahead,
            trg_take_profit =trg_take_profit,
            trg_stop_loss =trg_stop_loss,
            n_rand_features = n_rand_features,
            target_col=target_col,
            base_time_frame=5,
        )

        if manual:
            print(f"Percentage of the True Class: {df_all[df_all.target == 1].shape[0]/ df_all.shape[0]*100:.1f}")
            df_all["target"].value_counts().sort_index().plot(
                kind="bar", rot=0, ylabel="count"
            )
            plt.show()

        # ______________________________Create Time-Series Cross Validation Folds____________________________
        folds = split_time_series(
            df_all,
            max_train_size=max_train_size,
            n_splits=n_splits,
            test_size=test_size,
            train_test_gap=train_test_gap,
        )
        # ______________________________Pre-Backtest: Backtest on all raw data_______________________________
        df_raw_backtest, bt_column_name = cal_backtest_on_raw_cndl(
            df_raw_path = C5M_data_path,
            target_symbol = target_symbol,
            look_ahead= strg_look_ahead,
            take_profit= strg_take_profit,
            stop_loss= strg_stop_loss,
            trade_mode= trade_mode)
        # ______________________________RUN Quant Cross-Validation and Backtest on Folds_____________________
        non_feature_columns = ["target", "pred_as_val", "pred_val_proba", "pred_as_test", "pred_test_proba", "K","symbol"]
        swap_rate = symbols_dict[target_symbol]["swap_rate"][trade_mode]
        the_config["swap_rate"] = symbols_dict[target_symbol]["swap_rate"]
        input_cols_and_type, input_cols, evals, val_predictions, test_predictions, general_backtest_report, importance_df = quant_CV(
            df_all,
            folds,
            model=clf,
            early_stopping_rounds=early_stopping_rounds,
            df_raw_backtest = df_raw_backtest,
            bt_column_name = bt_column_name,
            non_feature_columns = non_feature_columns,
            swap_rate = swap_rate,
            )
        # ______________________________Retrain Last Model to Save___________________________________________
        if save_model_mode is not None:
            final_clf = model_func(
                man_params=man_params, manual=manual, model=model_name
            )
            final_clf = train_model_to_save(
                df_all, final_clf, max_train_size, save_model_mode, non_feature_columns
            )
        else:
            final_clf = None

        # ______________________________Delete df and Clear Memory___________________________________________
        del df_all
        gc.collect()

        # ______________________________Calculate and Aggregate Evaluations__________________________________
        selected_evals = {}
        eval_train = cal_aggregated_evals(evals[evals.dataset == "train"], set_name="train")
        selected_evals.update(eval_train)
        eval_valid = cal_aggregated_evals(evals[evals.dataset == "valid"], set_name="valid")
        selected_evals.update(eval_valid)
        eval_test = cal_aggregated_evals(evals[evals.dataset == "test"], set_name="test")
        selected_evals.update(eval_test)
        eval_valid_test = cal_aggregated_evals(evals[evals.dataset != "train"], set_name="valid&test")
        selected_evals.update(eval_valid_test)

        selected_evals.update(general_backtest_report)

        raw_aggregated_evals = eval_summerize_dict(evals, cal_mode="")

        # ______________________________Create Quant Experiment Tracker Object_______________________________

        exp_date = str(datetime.today().strftime("%Y-%m-%d_%H:%M"))
        # Experiment Name
        name = f"{model_name}_{target_col.replace('trg_clf_','')}_prof{selected_evals['profit_percent_test']:.2f}_max_dd{selected_evals['max_dd_test']:.2f}_median_sig{selected_evals['signal_count_median_test']:.2f}_date{exp_date}"

        QuantExpTracker_arguments = {
            "model": final_clf,

            "folds": folds,

            "val_predictions": val_predictions,
            "test_predictions": test_predictions,
            
            "evals": evals,
            "raw_agg_evals": raw_aggregated_evals,

            "input_cols": input_cols_and_type,
            "feature_importance_df": importance_df.sort_values("mean_importance", ascending=False),

            "train_duration_mean_fold": evals["train_duration"].astype(float).mean(),

        }
        exp_metadata = {
            "name": name,
            "config": the_config,
            "save_model_mode": save_model_mode,
            "selected_evals": selected_evals,
            "features_count": len(input_cols),
            "exp_date": exp_date,
            "max_CV_train_date": evals[evals.dataset == "train"]["Max_date"].max(),
        }

        QuantExpTracker_arguments.update(exp_metadata)
        exp_obj = QuantExpTracker(**QuantExpTracker_arguments)
        # Store Experiment Object in Pickle & Zip
        exp_obj.store_obj()
        if not manual:
        # ______________________________WandB Sweep Mode: Log & Return Artifact______________________________
            evals_logger(evals[evals.dataset == "train"],eval_train, name="train")
            evals_logger(evals[evals.dataset != "train"],eval_valid_test, name="valid&test")
            evals_logger(evals[evals.dataset == "valid"],eval_valid, name="valid")
            evals_logger(evals[evals.dataset == "test"],eval_test, name="test")
            wandb.log(general_backtest_report)

            artifact_name = target_col
            obj_artifact = wandb.Artifact(
                artifact_name, "experiment", metadata=exp_metadata
            )
            obj_artifact.add_file(
                local_path=exp_obj.store_name + ".zip", name=exp_obj.store_name + ".zip"
            )
            wandb.log_artifact(obj_artifact, QuantExpTracker_arguments)
            gc.collect()

        if manual:
        # ______________________________Manual Mode: Return Experiment Tracker Object_________________________

            artifact_name = "manual" + target_col
            gc.collect()
            return exp_obj, exp_metadata, artifact_name
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise ValueError("!!!")
