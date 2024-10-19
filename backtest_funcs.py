import numpy as np
import pandas as pd
from configss.symbols_info import symbols_dict

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

    df_raw_backtest = pd.read_parquet(df_raw_path,columns=["_time","open","high","low","close"])
    df_raw_backtest.columns = [
        "_time",
        f"{target_symbol}_M5_OPEN",
        f"{target_symbol}_M5_HIGH",
        f"{target_symbol}_M5_LOW",
        f"{target_symbol}_M5_CLOSE",  
    ]

    df_raw_backtest.sort_values("_time", inplace=True)
    df_raw_backtest['days_diff'] = (df_raw_backtest['_time'].dt.date - df_raw_backtest['_time'].dt.date.shift()).bfill().dt.days
    array = df_raw_backtest[
        [f"{target_symbol}_M5_CLOSE", f"{target_symbol}_M5_HIGH", f"{target_symbol}_M5_LOW", "days_diff"]
    ].to_numpy()

    df_raw_backtest[bt_column_name], df_raw_backtest["pip_diff"], df_raw_backtest["swap_days"] = calculate_classification_target_backtest(
        array,
        window_size,
        symbol_decimal_multiply=symbols_dict[target_symbol]["pip_size"],
        take_profit=take_profit,
        stop_loss=stop_loss,
        mode=trade_mode,
    )
    df_raw_backtest.dropna(inplace=True)
    return df_raw_backtest, bt_column_name

def do_backtest(
    df_model_signal: pd.DataFrame,
    spread: float,
    volume: float,
    initial_balance: int,
    df_raw_backtest : pd.DataFrame,
    bt_column_name:   str,
    swap_rate: float,
): 

    new_trg_df = df_model_signal.merge(df_raw_backtest, on="_time", how="inner")
    new_trg_df["net_profit"] = new_trg_df.pip_diff - spread


    ##? calculate balance
    new_trg_df["balance"] = new_trg_df["net_profit"] * volume * 10 + new_trg_df["swap_days"] * volume * swap_rate
    new_trg_df["balance"] = new_trg_df["balance"].cumsum()
    new_trg_df["balance"] += initial_balance

    ##? calculate max_drawdown
    max_drawdown = calculate_max_drawdown(new_trg_df["balance"])

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
        new_trg_df[
            [
                "_time",
                "model_prediction",
                f"{bt_column_name}",
                "pip_diff",
                "net_profit",
                "balance",
            ]
        ],
    )

