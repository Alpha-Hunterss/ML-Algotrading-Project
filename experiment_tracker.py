import zipfile
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd

class QuantExpTracker:
    from typing_extensions import runtime_checkable
    from typing import Protocol, Any, Type
    def __init__(
        self,
        name,
        model,
        folds,

        val_predictions,
        test_predictions,

        evals,
        raw_agg_evals,
        input_cols,
        feature_importance_df,
        train_duration_mean_fold,
        features_count,

        config,
        save_model_mode,
        selected_evals,
        exp_date,
        max_CV_train_date,

    ):
        self.name = name
        self.model = model
        self.folds = folds

        self.val_predictions = val_predictions
        self.test_predictions = test_predictions

        self.evals = evals
        self.raw_agg_evals = raw_agg_evals
        self.input_cols = input_cols
        self.feature_importance = feature_importance_df
        self.train_duration_mean_fold = train_duration_mean_fold
        self.features_count = features_count

        self.config = config
        self.save_model_mode = save_model_mode
        self.selected_evals = selected_evals
        self.exp_date = exp_date
        self.max_CV_train_date = max_CV_train_date

        self.store_name = f"{self.name}.pkl"

    def store_obj(self, store_zip: bool = True):
        file = open(self.store_name, "wb")
        pkl.dump(self, file)
        file.close()
        print(f"object stored as pickle: {self.store_name}")

        if store_zip:
            with zipfile.ZipFile(self.store_name + ".zip", "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(self.store_name)
            print(f"object pickle stored zipped: {self.store_name}")

    #__________________________________________ETL____________________________________________    
    def ETL(
            self,
            path,
            feature_set = None,
            imp_features = None,
            drop_features_conf = None,
            C5M_data_path = "/kaggle/input/tradeset-002/5M_candels_all_symbols_with_symbol_column.parquet",
            imp_features_path = "/kaggle/working/imp_features/",
            n_rand_features = None,
            ):
        import time
        import datetime
        import pandas as pd
        import numpy as np

        trade_mode = self.config['trade_mode']
        target_symbol = self.config['target_symbol']
        trg_look_ahead = self.config['trg_look_ahead']
        trg_take_profit = self.config['trg_take_profit']
        trg_stop_loss = self.config['trg_stop_loss']
        symbols_dict = self.__class__.get_symbols_info()

        target_col = f"trg_clf_{trade_mode}_{target_symbol}_M{trg_look_ahead}_TP{trg_take_profit}_SL{trg_stop_loss}"
        if feature_set is not None:
            if imp_features is not None:
                selected_fes = imp_features
            else:
                selected_cols = self.__class__.select_columns(path, target_col, feature_set)
                selected_fes = selected_cols + ["_time"]
            df = pd.read_parquet(path, columns=selected_fes)
        elif imp_features is not None:
            imp_features += ["_time"]
            df = pd.read_parquet(path, columns=imp_features)
        else: 
            df = pd.read_parquet(path)

        print(f'Len read columns is {df.shape[1]}')
        print("Calculating target --->")
        
        base_time_frame = 5
        window_size = int(trg_look_ahead // base_time_frame)
        df_raw = self.__class__.get_columns_from_5M_candle_parquet_for_targeting(
            parquet_path=C5M_data_path,
            symbol=target_symbol
            )
        array = df.merge(df_raw, on = '_time', how = 'left')[
            [f"{target_symbol}_M5_CLOSE", f"{target_symbol}_M5_HIGH", f"{target_symbol}_M5_LOW"]
        ].to_numpy()
        tic = time.time()
        df[target_col] = self.calculate_classification_target_numpy_ver(
                array,
                window_size,
                symbol_decimal_multiply = symbols_dict[target_symbol]["pip_size"],
            )

        toc = time.time()
        df.dropna(inplace = True)
        print(f"---> Target {target_col} has been generated in {toc-tic:.2f} seconds")
        print("df shape: ", df.shape)
        df.set_index(["_time"], inplace=True, drop=True)

        df.rename(columns={f"{target_col}": "target"}, inplace=True)
        df["target"] = df["target"].astype(int)

        ##? set targets to 0 in bad hours 
        df.loc[(df.index.get_level_values('_time').time>=datetime.time(0, 0))&(df.index.get_level_values('_time').time<=datetime.time(1, 0)),'target'] = 0

        other_target_cols = [col for col in df.columns if "trg_" in col]
        if len(other_target_cols) > 0:
            print("columns_removed: ", other_target_cols)
        df = df.drop(columns=other_target_cols, errors="ignore")

        # _____________________________ADD RANDOM FEATURES_______________________________________________
        random_features = []
        if n_rand_features is not None:
            for i in range(n_rand_features):
                df[f'RANDOM_{i}'] = np.random.random(df.shape[0])
                random_features.append(f'RANDOM_{i}')

        if drop_features_conf is not None:
            import glob
            imp_features_path = glob.glob(f'{imp_features_path}df_feature_imp_{trade_mode}_{target_symbol}*.pkl')[0]
            df_imp = pd.read_pickle(imp_features_path)
            df_imp['fe_type'] = df_imp['feature_names'].apply(self.__class__.get_fe_type)
            selected_features = self.__class__.drop_features_and_get_final_fe(
                df_imp=df_imp,
                keep_ratio=drop_features_conf['keep_ratio'],
                rand_drop_ratio=drop_features_conf['rand_drop_ratio'],
                rand_drop_rows_threshold=drop_features_conf['rand_drop_rows_threshold'],
                random_seed=drop_features_conf['random_seed']
                )
            selected_features = list(set(selected_features).intersection(set(list(df.columns))))
            df = df[selected_features + ["target"] + random_features]
            print("------------ APPLY DROP FEATURES CONF --------------")
            print("DataShape after drop_features_conf: ", df.shape)

        print("=" * 30)
        print("--> df final shape:", df.shape)
        print(
            f"--> df min_time: {df.index.get_level_values('_time').min()} | df max_time: {df.index.get_level_values('_time').max()}"
        )
        print(
            f"--> number of unique days: {df.index.get_level_values('_time').unique().shape[0]}"
        )
        print("=" * 30)

        return df        

    @staticmethod
    def get_columns_from_5M_candle_parquet_for_targeting(parquet_path:str,symbol:str):
        """
        the symbol must be capital
        
        """
        import pyarrow.parquet as pq
        fp = pq.read_table(
        source=parquet_path,
        use_threads=True,
        columns=['_time',"open","high","low","close"],
        filters=[('symbol', '=', symbol)]
        )
        df = fp.to_pandas() 
        df.columns = [
            "_time",
            f"{symbol}_M5_OPEN",
            f"{symbol}_M5_HIGH",
            f"{symbol}_M5_LOW",
            f"{symbol}_M5_CLOSE",
        
        ]
        
        assert (df[f"{symbol}_M5_OPEN"] <= df[f"{symbol}_M5_HIGH"]).all(), "Open higher than high"
        assert (df[f"{symbol}_M5_OPEN"] >= df[f"{symbol}_M5_LOW"]).all(), "Open lower than low"
        assert (df[f"{symbol}_M5_HIGH"] >= df[f"{symbol}_M5_LOW"]).all(), "High lower than low"
        return df
    #__________________________________________MODEL__________________________________________
    @runtime_checkable
    class TrainableModel(Protocol):
        """
        This is the base protocol on which each ML model in this repo should based.
        """
        
        def fit(self, X, y, *args, **kwargs):
            ...

        def predict(self, X, *args, **kwargs):
            ...

        def predict_proba(self, X, *args, **kwargs):
            ...

    def _import_model(self):
        try:
            if self.config['model_name'] == "RF":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier
            elif self.config['model_name'] == "XGB":
                from xgboost import XGBClassifier
                return XGBClassifier
            elif self.config['model_name'] == "LGBM":
                from lightgbm import LGBMClassifier
                return LGBMClassifier
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except ImportError as e:
            raise ImportError(f"Failed to import {self.model_type}. Please ensure it's installed: {str(e)}")
        
    def set_model_parameters(self, model_class:TrainableModel):
        import inspect
        # Check if the model class has the `get_params` method
        if hasattr(model_class(), 'get_params'):
            model_instance = model_class()
            valid_params = model_instance.get_params()
        else:
            # Fall back to inspecting the constructor's parameters
            model_signature = inspect.signature(model_class.__init__)
            model_params = model_signature.parameters
            valid_params = {param: None for param in model_params if param != 'self'}
        
        all_params = self.config["parameters"]
        parameters = {param: value for param, value in all_params.items() if param in valid_params}
        
        return parameters

    def model_func(self):

        model_class = self._import_model()
        if not model_class or not issubclass(model_class, QuantExpTracker.TrainableModel):
            raise TypeError("Make sure the model has the following methods: `fit`, `predict` and `predict_proba`")
        
        parameters = self.set_model_parameters(model_class)
        assert len(parameters) > 0,"!!! NO parameters"
        print(f'Final Parameters of the Model: {parameters}')
        clf = model_class(**parameters)

        return clf
    
    #__________________________________________TRAIN__________________________________________
    def retrain_model(
            self,
            dataset_path:str ="/kaggle/input/tradeset-002/database.parquet",
            C5M_data_path:str = "/kaggle/input/tradeset-002/5M_candels_all_symbols_with_symbol_column.parquet",
            save_model_mode:str = 'production_valid',
            non_feature_columns:list[str] = ["target"],
            save_model_in_object: bool = False,
            ):
        the_features = list(self.input_cols.keys())
        df = self.ETL(
            path = dataset_path,
            imp_features = the_features,
            C5M_data_path= C5M_data_path
            )
        final_clf = self.model_func()
        max_train_size = self.config['max_train_size']
        test_size = self.config['test_size']
        train_test_gap = self.config['train_test_gap']

        trained_model = self.train_model(
            df = df,
            final_clf = final_clf,
            max_train_size = max_train_size,
            test_size = test_size,
            train_test_gap = train_test_gap,
            save_model_mode = save_model_mode,
            non_feature_columns = non_feature_columns)
        
        if save_model_in_object:
            self.model = trained_model
            print('Model saved to self.model')

        print('Retraining is done')

        return trained_model

    @staticmethod
    def train_model(
        df: pd.DataFrame,
        final_clf: TrainableModel,
        max_train_size: int,
        test_size: int,
        train_test_gap: int,
        save_model_mode: str|None = 'last_train_size',
        non_feature_columns: list[str] = ["target", "pred_as_val", "pred_val_proba", "pred_as_test", "pred_test_proba", "K"],
        ):
        """
        This function is used for retrainig the model on specific set of data specifically the most recent historical data
            df: the historical dataset,
            final_clf: classification model,
            max_train_size: maximum number of training samples,
            save_model_mode: types of trainig set:
                'production_valid': Cutting valid & test size from the last historical day of df
                'production_test': Cutting test size from the last historical day of df
                'sample_train_size': Use all historical data and with max_train_size as sample size 
                'last_train_size': Use the most recent max_train_size 
                'all_data': Use all historical data
                None: The function will return none
        # 
        """
        print(f'Training based on {save_model_mode} mode')

        if save_model_mode == "last_train_size":
            final_clf.fit(
                df.iloc[-max_train_size:].drop(
                    columns=non_feature_columns
                ),
                df.iloc[-max_train_size:]["target"],
            )
        
        elif save_model_mode == "production_valid":
            final_clf.fit(
                df.iloc[-(max_train_size+train_test_gap):-train_test_gap].drop(
                    columns=non_feature_columns
                ),
                df.iloc[-(max_train_size+train_test_gap):-train_test_gap]["target"],
            )

        elif save_model_mode == "production_test":

            final_clf.fit(
                df.iloc[-(max_train_size+train_test_gap+test_size):-(train_test_gap+test_size)].drop(
                    columns=non_feature_columns
                ),
                df.iloc[-(max_train_size+train_test_gap+test_size):-(train_test_gap+test_size)]["target"],
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
            final_clf.fit(
                df.drop(columns=non_feature_columns),
                df["target"],
            )
        else:
            final_clf = None

        return final_clf

    #__________________________________________TARGET________________________________________
    def calculate_classification_target_numpy_ver(
            self,
            array,
            window_size,
            symbol_decimal_multiply: float,
            ):
        
        take_profit= self.config['trg_take_profit']
        stop_loss = self.config['trg_stop_loss']
        mode = self.config['trade_mode']
        target_list = []
        if mode == "long":
            for i in range(array.shape[0] - window_size):
                selected_chunk = array[i : i + window_size]
                pip_diff_close = (
                    selected_chunk[1:, 0] - selected_chunk[0, 0]
                ) / symbol_decimal_multiply
                pip_diff_low = (
                    selected_chunk[1:, 2] - selected_chunk[0, 0]
                ) / symbol_decimal_multiply
                # BUY CLASS
                target = 0

                buy_tp_cond = pip_diff_close >= take_profit
                buy_sl_cond = pip_diff_low <= -stop_loss

                if buy_tp_cond.any() == True:
                    arg_buy_tp_cond = np.where((pip_diff_close >= take_profit))[0][0]
                    if buy_sl_cond[0 : arg_buy_tp_cond + 1].any() == False:
                        target = 1

                target_list.append(target)

        elif mode == "short":
            for i in range(array.shape[0] - window_size):
                selected_chunk = array[i : i + window_size]

                pip_diff_high = (
                    selected_chunk[1:, 1] - selected_chunk[0, 0]
                ) / symbol_decimal_multiply
                pip_diff_close = (
                    selected_chunk[1:, 0] - selected_chunk[0, 0]
                ) / symbol_decimal_multiply

                # BUY CLASS
                target = 0
                sell_tp_cond = pip_diff_close <= -take_profit
                sell_sl_cond = pip_diff_high >= stop_loss

                if sell_tp_cond.any() == True:
                    arg_sell_tp_cond = np.where((pip_diff_close <= -take_profit))[0][0]
                    if sell_sl_cond[0 : arg_sell_tp_cond + 1].any() == False:
                        target = 1

                target_list.append(target)
        
        
        for _ in range(window_size):
            target_list.append(None)

        return target_list
    
    #__________________________________________RAW BACKTEST__________________________________
    @staticmethod
    def cal_backtest_on_raw_cndl(
        df_raw_path: str,
        target_symbol: str,
        look_ahead: int,
        take_profit: int,
        stop_loss: int,
        trade_mode: str
        )-> pd.DataFrame:
        """
        This function is basicaly a pre-backtest fucntion that calculates Backtest on all raw data (all times) based on strategy. 
        This function assumes we trade on each and every time step and calculates the backtest result for each time.
        The result can be merged with actual model signals to reach final backtest 
        """

        base_time_frame = 5
        window_size = int(look_ahead // base_time_frame)
        bt_column_name = (
            f"trg_clf_{trade_mode}_{target_symbol}_M{look_ahead}_TP{take_profit}_SL{stop_loss}"
        )

        df_raw_backtest = QuantExpTracker.get_columns_from_5M_candle_parquet_for_targeting(
            parquet_path=df_raw_path,
            symbol=target_symbol
            )
        df_raw_backtest.sort_values("_time", inplace=True)
        df_raw_backtest['days_diff'] = (df_raw_backtest['_time'].dt.date - df_raw_backtest['_time'].dt.date.shift()).bfill().dt.days
        array = df_raw_backtest[
            [f"{target_symbol}_M5_CLOSE", f"{target_symbol}_M5_HIGH", f"{target_symbol}_M5_LOW", "days_diff"]
        ].to_numpy()

        df_raw_backtest[bt_column_name], df_raw_backtest["pip_diff"], df_raw_backtest["swap_days"] = QuantExpTracker.calculate_classification_target_backtest(
            array,
            window_size,
            symbol_decimal_multiply= QuantExpTracker.get_symbols_info()[target_symbol]["pip_size"],
            take_profit=take_profit,
            stop_loss=stop_loss,
            mode=trade_mode,
        )
        df_raw_backtest.dropna(inplace=True)
        return df_raw_backtest
    
    @staticmethod
    def calculate_classification_target_backtest(
        array,
        window_size,
        symbol_decimal_multiply: float = 0.0001,
        take_profit: int = 70,
        stop_loss: int = 30,
        mode: str = "long",
    ):
        """
        This function returns two elements:
        Target: which has 3 different values. 1 means the position reaches the take profit price.
            -1 means the position ended in stoploss. 0 is in between.
        exit_price_diff is in pips.
        """
        swap_days_list = []
        target_list = []
        exit_price_diff_list = []

        if mode == "long":
            for i in range(array.shape[0] - window_size):
                selected_chunk = array[i : i + window_size]

                pip_diff_high = (
                    selected_chunk[1:, 1] - selected_chunk[0, 0]
                ) / symbol_decimal_multiply
                pip_diff_low = (
                    selected_chunk[1:, 2] - selected_chunk[0, 0]
                ) / symbol_decimal_multiply

                # BUY CLASS

                buy_tp_cond = pip_diff_high >= take_profit
                buy_sl_cond = pip_diff_low <= -stop_loss

                if buy_tp_cond.any():
                    arg_buy_tp_cond = np.where((pip_diff_high >= take_profit))[0][0]
                    if buy_sl_cond[0 : arg_buy_tp_cond + 1].any() == False:
                        swap_days = selected_chunk[1 : arg_buy_tp_cond + 1,3].sum()
                        target = 1
                        exit_price_diff = take_profit
                    else:
                        arg_buy_sl_cond = np.where((pip_diff_low <= -stop_loss))[0][0]
                        swap_days = selected_chunk[1 : arg_buy_sl_cond + 1,3].sum()
                        target = -1
                        exit_price_diff = -stop_loss

                elif buy_sl_cond.any():
                    arg_buy_sl_cond = np.where((pip_diff_low <= -stop_loss))[0][0]
                    swap_days = selected_chunk[1 : arg_buy_sl_cond + 1,3].sum()
                    target = -1
                    exit_price_diff = -stop_loss

                else:
                    target = 0
                    swap_days = selected_chunk[1 : ,3].sum()
                    exit_price_diff = (
                        selected_chunk[-1, 0] - selected_chunk[0, 0]
                    ) / symbol_decimal_multiply 

                target_list.append(target)
                swap_days_list.append(swap_days)
                exit_price_diff_list.append(exit_price_diff)

        elif mode == "short":
            for i in range(array.shape[0] - window_size):
                selected_chunk = array[i : i + window_size]

                pip_diff_high = (
                    selected_chunk[1:, 1] - selected_chunk[0, 0]
                ) / symbol_decimal_multiply
                pip_diff_low = (
                    selected_chunk[1:, 2] - selected_chunk[0, 0]
                ) / symbol_decimal_multiply

                # BUY CLASS
                target = 0
                sell_tp_cond = pip_diff_low <= -take_profit
                sell_sl_cond = pip_diff_high >= stop_loss

                if sell_tp_cond.any():
                    arg_sell_tp_cond = np.where((pip_diff_low <= -take_profit))[0][0]
                    if sell_sl_cond[0 : arg_sell_tp_cond + 1].any() == False:
                        swap_days = selected_chunk[1 : arg_buy_tp_cond + 1,3].sum()
                        target = 1
                        exit_price_diff = take_profit
                    else:
                        arg_sell_sl_cond = np.where((pip_diff_high >= stop_loss))[0][0]
                        swap_days = selected_chunk[1 : arg_sell_sl_cond + 1,3].sum()
                        target = -1
                        exit_price_diff = -stop_loss

                elif sell_sl_cond.any():
                    arg_sell_sl_cond = np.where((pip_diff_high >= stop_loss))[0][0]
                    swap_days = selected_chunk[1 : arg_sell_sl_cond + 1,3].sum()
                    target = -1
                    exit_price_diff = -stop_loss

                else:
                    target = 0
                    swap_days = selected_chunk[1:,3].sum()
                    exit_price_diff = (
                        selected_chunk[-1, 0] - selected_chunk[0, 0]
                    ) / symbol_decimal_multiply

                target_list.append(target)
                swap_days_list.append(swap_days)
                exit_price_diff_list.append(exit_price_diff)

        for _ in range(window_size):
            swap_days_list.append(None)
            target_list.append(None)
            exit_price_diff_list.append(None)

        return target_list, exit_price_diff_list, swap_days_list

    #__________________________________________BACKTEST_______________________________________
    @staticmethod
    def do_backtest(
        df_model_signal: pd.DataFrame,
        spread: float,
        volume: float,
        initial_balance: int,
        df_raw_backtest : pd.DataFrame,
        swap_rate: float,
    ):  # TODO: add max_open_positions_volume.

        new_trg_df = df_model_signal.merge(df_raw_backtest, on="_time", how="inner")
        new_trg_df["net_profit"] = new_trg_df.pip_diff - spread


        ##? calculate balance
        new_trg_df["balance"] = new_trg_df["net_profit"] * volume * 10 + new_trg_df["swap_days"] * volume * swap_rate
        new_trg_df["balance"] = new_trg_df["balance"].cumsum()
        new_trg_df["balance"] += initial_balance

        ##? calculate max_drawdown
        max_drawdown = QuantExpTracker.calculate_max_drawdown(new_trg_df["balance"])

        ##? calculate duration:
        if new_trg_df.shape[0] == 0:
            bactesk_report = {
                "balance_cash": initial_balance,
                "profit_pips": 0,
                "max_draw_down": 0,
                "profit_percent":0
                }
        else:
            bactesk_report = {
                "balance_cash": int(new_trg_df.iloc[-1]["balance"]),
                "profit_pips": int(new_trg_df["net_profit"].sum()),
                "max_draw_down": round(max_drawdown, 2),
                "profit_percent": round(
                    ((new_trg_df.iloc[-1]["balance"] - initial_balance) / initial_balance)
                    * 100,
                    2,
                ),
            }

        return (
            bactesk_report,
            new_trg_df)

    @staticmethod
    def calculate_max_drawdown(balance_series):
        """
        Calculate the maximum drawdown from a balance column in a pandas DataFrame.

        Args:
            df (pandas.DataFrame): Input DataFrame containing the balance column.
            balance_col (str): Name of the column containing the balance values.

        Returns:
            float: Maximum drawdown value.
        """
        # Get the cumulative maximum balance up to each point in time
        cum_max = balance_series.cummax()

        # Calculate the drawdown at each point in time
        drawdowns = (balance_series - cum_max) / cum_max

        # Return the maximum drawdown
        return drawdowns.min() * 100

    #__________________________________________FEATURE SELECTION______________________________
    @staticmethod
    def select_columns(
        path:str,
        target_column:str,
        feature_set:dict[str, bool| list[str]]
        ):

        import itertools
        import ParquetFile
        raw_columns = [f.name for f in ParquetFile(path).schema]

        target_symbol = target_column.split("_")[3]
        print(f"TARGET SYMBOL: {target_symbol}")
        second_alternative = [
            "AUDUSD",
            "NZDUSD",
            "USDCHF",
            "USA500IDXUSD",
            "CADJPY",
            "EURGBP",
        ]
        all_cols = {}
        all_cols["main"] = [f for f in raw_columns if target_symbol in f and "fe_" in f]
        all_cols["alter"] = [
            f for f in raw_columns if any(i in f for i in second_alternative) and "fe_" in f
        ]
        all_cols["main_alter"] = [
            f
            for f in raw_columns
            if target_symbol not in f and f not in all_cols["alter"] and "fe_" in f
        ]

        fe_cndl_init = list(
            itertools.chain(*[all_cols[k] for k in feature_set["fe_cndl"]])
        )
        fe_cndl = [
            f
            for f in fe_cndl_init
            if "_cndl_" in f
            and all(
                i not in f
                for i in ["shift", "EMA", "SMA", "RSI", "ptrn", "ratio", "WIN", "RSTD", "ATR"]
            )
        ]
        if feature_set["fe_time"]:
            fe_time = [f for f in raw_columns if "fe_time" in f]
        else:
            fe_time = []

        selected_cols = []
        selected_cols.extend(fe_cndl)
        selected_cols.extend(fe_time)

        for fe_type in ["fe_RSTD", "fe_ATR", "fe_RSI","fe_ratio","fe_EMA","fe_SMA","fe_cndl_shift","fe_WIN","fe_cndl_ptrn","fe_market_close"]:
            fe_type_list = QuantExpTracker.select_feature_type(all_cols = all_cols, feature_set= feature_set, feature_type= fe_type)
            selected_cols.extend(fe_type_list)

        selected_cols = list(set(selected_cols))

        print(f"--> number of selected cols: {len(selected_cols)}")
        return selected_cols
    
    @staticmethod
    def select_feature_type(
        all_cols: dict[str, list[str]], 
        feature_set: dict[str, bool| list[str]],
        feature_type: str
        ):
        import itertools
        fe_init = list(
            itertools.chain(*[all_cols[k] for k in feature_set[feature_type]])
        )
        if feature_type == 'fe_ratio':
            fe_list = [f for f in fe_init if f"{feature_type}" in f]
        else:
            fe_list = [f for f in fe_init if f"{feature_type}" in f and 'ratio' not in f]

        return fe_list
    
    @staticmethod
    def get_fe_type(f: str):
        if "_cndl_" in f and all(i not in f for i in ["shift", "EMA", "SMA", "RSI", "ptrn", "ratio", "WIN", "RSTD", "ATR"]):
            return 'fe_cndl'
        elif 'ratio' in f:
            return 'ratio'
        else: 
            for feature_type in ["fe_time","fe_RSTD", "fe_ATR", "fe_RSI","fe_ratio","fe_EMA","fe_SMA",
                                "fe_cndl_shift","fe_WIN","fe_cndl_ptrn","fe_market_close", "RANDOM"]:
                if feature_type in f:
                    return feature_type
            return 'other'
        
    @staticmethod
    def drop_features_and_get_final_fe(
        df_imp: pd.DataFrame,
        keep_ratio:float,\
        rand_drop_ratio: float,
        rand_drop_rows_threshold:float,
        random_seed: int)->list[str]:
        import pandas as pd
        """
        this function keep important features based on the given config and drop some  features randomly
        Inputs:
            keep_ratio : only keep the top keep_ratio features from each type
            rand_drop_ratio : drop rand_drop_ratio of features from types that have at least rand_drop_rows_threshold features
            rand_drop_rows_threshold 
            random_seed : random seed for the randomly droping process
        Output:
            selected features list
        """
        max_rand_di = df_imp[df_imp.feature_names.str.startswith('RANDOM')].mean_importance.max()
        df_imp = df_imp[df_imp.mean_importance>max_rand_di]

        df_imp_list = []
        for g in df_imp.groupby('fe_type'):
            
            df_temp = g[1]
            df_temp.sort_values('mean_importance',inplace = True, ascending = False)
            df_temp = df_temp.iloc[:int(keep_ratio*g[1].shape[0])]
            
            if df_temp.shape[0]>= rand_drop_rows_threshold:
                df_temp = df_temp.drop(df_temp.sample(int(rand_drop_ratio*df_temp.shape[0]), random_state=random_seed).index)
                print(int(rand_drop_ratio*df_temp.shape[0]), 'rows randomly droped from ', g[0])
            df_imp_list.append(df_temp)
            print(g[0],': ',g[1].shape[0],'down to -->' ,df_temp.shape[0])
            print('- '*20)
        df_imp_selected = pd.concat(df_imp_list)
        print('N.o. final selected features', df_imp_selected.shape[0])

        return list(df_imp_selected.feature_names)
    
    #__________________________________________SYMBOL CONFIG____________________________________
    @staticmethod
    def get_symbols_info():
        symbols_dict = {
            # ? Majors
            "EURUSD": {
                "decimal_divide": 1e5,
                "pip_size": 0.0001,
                "metatrader_id": "EURUSD",
                "dukascopy_id": "EURUSD",
            },
            "AUDUSD": {
                "decimal_divide": 1e5,
                "pip_size": 0.0001,
                "metatrader_id": "AUDUSD",
                "dukascopy_id": "AUDUSD",
            },
            "GBPUSD": {
                "decimal_divide": 1e5,
                "pip_size": 0.0001,
                "metatrader_id": "GBPUSD",
                "dukascopy_id": "GBPUSD",
            },
            "NZDUSD": {
                "decimal_divide": 1e5,
                "pip_size": 0.0001,
                "metatrader_id": "NZDUSD",
                "dukascopy_id": "NZDUSD",
            },
            "USDCAD": {
                "decimal_divide": 1e5,
                "pip_size": 0.0001,
                "metatrader_id": "USDCAD",
                "dukascopy_id": "USDCAD",
            },
            "USDCHF": {
                "decimal_divide": 1e5,
                "pip_size": 0.0001,
                "metatrader_id": "USDCHF",
                "dukascopy_id": "USDCHF",
            },
            "USDJPY": {
                "decimal_divide": 1e3,
                "pip_size": 0.01,
                "swap_rate": {"long": 2, "short": -5},
                "metatrader_id": "USDJPY",
                "dukascopy_id": "USDJPY",
            },
            # ? Metals
            # "XAGUSD":{"decimal_divide":1e+3,"pip_size":0.1,"metatrader_id":"XAGUSD","dukascopy_id":"XAGUSD"}, # Spot silver
            "XAUUSD": {
                "decimal_divide": 1e3,
                "pip_size": 0.1,
                "yahoo_finance": ["GC=F"],
                "metatrader_id": "XAUUSD",
                "dukascopy_id": "XAUUSD",
            },  # Spot gold
            # ? Crosses
            "EURJPY": {
                "decimal_divide": 1e3,
                "pip_size": 0.01,
                "metatrader_id": "EURJPY",
                "dukascopy_id": "EURJPY",
            },
            # "AUDCHF":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"AUDCHF","dukascopy_id":"AUDCHF"},
            # "AUDJPY":{"decimal_divide":1e+3,"pip_size":0.01,"metatrader_id":"AUDJPY","dukascopy_id":"AUDJPY"},
            "CADJPY": {
                "decimal_divide": 1e3,
                "pip_size": 0.01,
                "metatrader_id": "CADJPY",
                "dukascopy_id": "CADJPY",
            },
            # "CADCHF":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"CADCHF","dukascopy_id":"CADCHF"},
            # "CHFJPY":{"decimal_divide":1e+3,"pip_size":0.01,"metatrader_id":"CHFJPY","dukascopy_id":"CHFJPY"},
            # "EURAUD":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"EURAUD","dukascopy_id":"EURAUD"},
            # "EURCAD":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"EURCAD","dukascopy_id":"EURCAD"},
            # "EURCHF":{"decimal_divide":1e+5,"pip_size":0.0001,"metatrader_id":"EURCHF","dukascopy_id":"EURCHF"},
            "EURGBP": {
                "decimal_divide": 1e5,
                "pip_size": 0.0001,
                "metatrader_id": "EURGBP",
                "dukascopy_id": "EURGBP",
            },
            # #? Indies:
            # "DOLLARIDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"DOLLARIDXUSD"},
            # "USA30IDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"USA30IDXUSD"},
            # "USATECHIDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"USATECHIDXUSD"},
            # "USA500IDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"SP500.r","dukascopy_id":"USA500IDXUSD"},
            # "USSC2000IDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"USSC2000IDXUSD"},
            # "VOLIDXUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"VOLIDXUSD"},
            # #? Energy:
            # "DIESELCMDUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"DIESELCMDUSD"},
            # "BRENTCMDUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"BRENTCMDUSD"},
            # "LIGHTCMDUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"","dukascopy_id":"LIGHTCMDUSD"},
            # "GASCMDUSD":{"decimal_divide":1e+3,"pip_size":0.0001,"metatrader_id":"NG-Cr","dukascopy_id":"GASCMDUSD"},
        }

        return symbols_dict
    
    #__________________________________________PLOT_____________________________________________
    def plot_evals(self, dataset = "train"):
        
        self.evals[self.evals.dataset == dataset][
            ["profit_percent", "max_dd", "precision"]
        ].boxplot(showmeans=True)
        plt.yticks([i for i in np.arange(0, 1, 0.1)])
        plt.title(self.name)

    def plot_feature_importance(self, top_count=20):

        importances_df = self.feature_importance.sort_values(
            "mean_importance", ascending=False
        )
        plt.barh(
            importances_df.feature_names.iloc[:top_count],
            importances_df.mean_importance.iloc[:top_count],
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Feature Importance")