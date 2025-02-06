import numpy as np
import pandas as pd
from configss.symbols_info import symbols_dict
import matplotlib.pyplot as plt
import math


# Function to calculate the rolling standard deviation (RSTD)
def calculate_rstd(selected_chunk, symbol_decimal_multiply):
    """
    Calculate the rolling standard deviation (RSTD) based on the selected chunk of data.
    """
    chunks_len = len(selected_chunk)
    if chunks_len == 0:
        return 0

    if chunks_len == 1:
        raise ValueError("'rstd_window_size' cannot be 1.")

    returns = [
        (selected_chunk[i+1, 0] - selected_chunk[i, 0]) / symbol_decimal_multiply
        for i in range(chunks_len-1)
    ]

    return float(np.std(returns))


def calculate_classification_target_backtest(
    array,
    window_size,
    use_dynamic_sl: bool = False,
    max_strg_sl_dynamic_perc: int = 20,
    dynamic_sl_scale_type: str = "third_quartile",
    rstd_window_size: int = 12,
    close_positions_at_midnight: bool = False,
    symbol_decimal_multiply: float = 0.0001,
    take_profit: int = 70,
    stop_loss: int = 30,
    take_profit_perc: float = 0.1,
    stop_loss_perc: float = 0.033,
    use_perc_levels: bool = False,
    dynamic_sl_type: str = None,
    atr_level_multiplication: float = 1.5,
    spread_pip: int = 5,
    mode: str = "long",
):
    """
    This function returns three elements:
    Target: which has 3 different values. 1 means the position reaches the take profit price.
        -1 means the position ended in stoploss. 0 is in between.
    exit_price_diff is in pips, and swap_days tracks the overnight holding costs.
    """
    swap_days_list = []
    target_list = []
    exit_price_diff_list = []
    time_open_position_list = []
    stop_losses_list = []
    take_profits_list = []
    date_column = np.array(
        [np.datetime64(datetime, 'D') for datetime in array[:, 4]]
    )
    max_strg_sl_dynamic_ratio = max_strg_sl_dynamic_perc / 100
    take_profit_ratio = take_profit_perc / 100
    stop_loss_ratio = stop_loss_perc / 100
    spread = spread_pip * symbol_decimal_multiply

    if use_dynamic_sl:
        if dynamic_sl_type=="rstd":
            rstds = [
                calculate_rstd(array[i - rstd_window_size: i], symbol_decimal_multiply)
                for i in range(rstd_window_size, array.shape[0] - window_size)
            ]

            if dynamic_sl_scale_type == 'third_quartile':
                third_quartile = np.percentile(rstds, 75)
                rstd_exponent = stop_loss/third_quartile
            elif dynamic_sl_scale_type == 'second_tercile':
                second_tercile = np.percentile(rstds, 66)
                rstd_exponent = stop_loss/second_tercile
            elif dynamic_sl_scale_type == 'median':
                median = np.percentile(rstds, 50)
                rstd_exponent = stop_loss/median
            else:
                raise ValueError(
                    "The scale type should be either `third_quartile`, `second_tercile` or `median`"
                )

            rstds_norm = np.array(rstds) * rstd_exponent

        if use_perc_levels:
            reward = take_profit_perc/stop_loss_perc
        else:
            reward = take_profit/stop_loss

        if mode == "long":
            for i in range(array.shape[0] - window_size):
                selected_chunk = array[i : i + window_size]

                if close_positions_at_midnight:
                    dates = date_column[i: i + window_size]
                    curr_date = dates[0]
                    selected_chunk = selected_chunk[dates == curr_date]

                if dynamic_sl_type in ["atr", "etr"]:
                    curr_close = selected_chunk[0, 0]

                    if selected_chunk[0, 5] is not None:
                        calc_sl = -selected_chunk[0, 5]*atr_level_multiplication
                        max_strg_sl = (curr_close / symbol_decimal_multiply) * max_strg_sl_dynamic_ratio
                        calc_sl = max(calc_sl, -max_strg_sl)
                        calc_tp = -reward*calc_sl
                    else:
                        calc_sl = -(curr_close / symbol_decimal_multiply) * stop_loss_ratio
                        max_strg_sl = (curr_close / symbol_decimal_multiply) * max_strg_sl_dynamic_ratio
                        calc_sl = max(calc_sl, -max_strg_sl)
                        calc_tp = (curr_close / symbol_decimal_multiply) * take_profit_ratio

                elif dynamic_sl_type=="rstd":
                    if i >= rstd_window_size:  # Ensure that there's enough data for RSTD calculation
                        rstd_sl = rstds_norm[i-rstd_window_size]
                        calc_sl = -max(stop_loss/4, min(stop_loss, rstd_sl))
                        calc_tp = -reward * calc_sl
                    else:
                        calc_sl = -stop_loss
                        calc_tp = take_profit

                # Calculate pip differences
                pip_diff_high = (selected_chunk[1:, 1] - selected_chunk[0, 0]) / symbol_decimal_multiply
                pip_diff_low = (selected_chunk[1:, 2] - selected_chunk[0, 0]) / symbol_decimal_multiply

                buy_tp_cond = pip_diff_high >= calc_tp+spread_pip
                buy_sl_cond = pip_diff_low <= calc_sl+spread_pip

                if buy_tp_cond.any():
                    arg_buy_tp_cond = np.where(pip_diff_high >= calc_tp+spread_pip)[0][0]
                    if not buy_sl_cond[: arg_buy_tp_cond + 1].any():
                        swap_days = selected_chunk[1 : arg_buy_tp_cond + 1, 3].sum()
                        target = 1
                        exit_price_diff = calc_tp
                        index_open_position = arg_buy_tp_cond + 1
                    else:
                        arg_buy_sl_cond = np.where(pip_diff_low <= calc_sl+spread_pip)[0][0]
                        swap_days = selected_chunk[1 : arg_buy_sl_cond + 1, 3].sum()
                        target = -1
                        exit_price_diff = calc_sl
                        index_open_position = arg_buy_sl_cond + 1

                elif buy_sl_cond.any():
                    arg_buy_sl_cond = np.where(pip_diff_low <= calc_sl+spread_pip)[0][0]
                    swap_days = selected_chunk[1 : arg_buy_sl_cond + 1, 3].sum()
                    target = -1
                    exit_price_diff = calc_sl
                    index_open_position = arg_buy_sl_cond + 1

                else:
                    target = 0
                    swap_days = selected_chunk[1:, 3].sum()
                    exit_price_diff = (selected_chunk[-1, 0] - selected_chunk[0, 0]) / symbol_decimal_multiply

                    if close_positions_at_midnight:
                        index_open_position = selected_chunk.shape[0]
                    else:
                        index_open_position = window_size

                target_list.append(target)
                swap_days_list.append(swap_days)
                exit_price_diff_list.append(exit_price_diff)
                time_open_position_list.append(index_open_position)
                stop_losses_list.append(-calc_sl)
                take_profits_list.append(calc_tp)

        elif mode == "short":
            for i in range(array.shape[0] - window_size):
                selected_chunk = array[i : i + window_size]

                if close_positions_at_midnight:
                    dates = date_column[i: i + window_size]
                    curr_date = dates[0]
                    selected_chunk = selected_chunk[dates == curr_date]

                if dynamic_sl_type in ["atr", "etr"]:
                    curr_close = selected_chunk[0, 0]

                    if selected_chunk[0, 5] is not None:
                        calc_sl = selected_chunk[0, 5]*atr_level_multiplication
                        max_strg_sl = (curr_close / symbol_decimal_multiply) * max_strg_sl_dynamic_ratio
                        calc_sl = min(calc_sl, max_strg_sl)
                        calc_tp = -reward*calc_sl
                    else:
                        calc_sl = (curr_close / symbol_decimal_multiply) * stop_loss_ratio
                        max_strg_sl = (curr_close / symbol_decimal_multiply) * max_strg_sl_dynamic_ratio
                        calc_sl = min(calc_sl, max_strg_sl)
                        calc_tp = -(curr_close / symbol_decimal_multiply) * take_profit_ratio

                elif dynamic_sl_type=="rstd":
                    if i >= rstd_window_size:  # Ensure that there's enough data for RSTD calculation
                        rstd_sl = rstds_norm[i-rstd_window_size]
                        calc_sl = max(stop_loss/4, min(stop_loss, rstd_sl))
                        calc_tp = -reward * calc_sl
                    else:
                        calc_sl = stop_loss
                        calc_tp = -take_profit

                pip_diff_high = (selected_chunk[1:, 1] - (selected_chunk[0, 0]+spread)) / symbol_decimal_multiply
                pip_diff_low = (selected_chunk[1:, 2] - (selected_chunk[0, 0]+spread)) / symbol_decimal_multiply

                sell_tp_cond = pip_diff_low <= calc_tp-spread_pip
                sell_sl_cond = pip_diff_high >= calc_sl-spread_pip

                if sell_tp_cond.any():
                    arg_sell_tp_cond = np.where(pip_diff_low <= calc_tp-spread_pip)[0][0]
                    if not sell_sl_cond[: arg_sell_tp_cond + 1].any():
                        swap_days = selected_chunk[1 : arg_sell_tp_cond + 1, 3].sum()
                        target = 1
                        exit_price_diff = -calc_tp
                        index_open_position = arg_sell_tp_cond + 1
                    else:
                        arg_sell_sl_cond = np.where(pip_diff_high >= calc_sl-spread_pip)[0][0]
                        swap_days = selected_chunk[1 : arg_sell_sl_cond + 1, 3].sum()
                        target = -1
                        exit_price_diff = -calc_sl
                        index_open_position = arg_sell_sl_cond + 1

                elif sell_sl_cond.any():
                    arg_sell_sl_cond = np.where(pip_diff_high >= calc_sl-spread_pip)[0][0]
                    swap_days = selected_chunk[1 : arg_sell_sl_cond + 1, 3].sum()
                    target = -1
                    exit_price_diff = -calc_sl
                    index_open_position = arg_sell_sl_cond + 1

                else:
                    target = 0
                    swap_days = selected_chunk[1:, 3].sum()
                    exit_price_diff = (selected_chunk[0, 0] - selected_chunk[-1, 0]) / symbol_decimal_multiply

                    if close_positions_at_midnight:
                        index_open_position = selected_chunk.shape[0]
                    else:
                        index_open_position = window_size

                target_list.append(target)
                swap_days_list.append(swap_days)
                exit_price_diff_list.append(exit_price_diff)
                time_open_position_list.append(index_open_position)
                stop_losses_list.append(calc_sl)
                take_profits_list.append(-calc_tp)

    else:
        if mode == "long":
            for i in range(array.shape[0] - window_size):
                selected_chunk = array[i : i + window_size]

                if close_positions_at_midnight:
                    dates = date_column[i: i + window_size]
                    curr_date = dates[0]
                    selected_chunk = selected_chunk[dates == curr_date]

                # Calculate pip differences
                pip_diff_high = (selected_chunk[1:, 1] - selected_chunk[0, 0]) / symbol_decimal_multiply
                pip_diff_low = (selected_chunk[1:, 2] - selected_chunk[0, 0]) / symbol_decimal_multiply

                if use_perc_levels:
                    curr_close = selected_chunk[0, 0]
                    take_profit = (curr_close / symbol_decimal_multiply) * take_profit_ratio
                    stop_loss = (curr_close / symbol_decimal_multiply) * stop_loss_ratio

                buy_tp_cond = pip_diff_high >= take_profit+spread_pip
                buy_sl_cond = pip_diff_low <= -stop_loss+spread_pip

                if buy_tp_cond.any():
                    arg_buy_tp_cond = np.where(pip_diff_high >= take_profit+spread_pip)[0][0]
                    if not buy_sl_cond[: arg_buy_tp_cond + 1].any():
                        swap_days = selected_chunk[1 : arg_buy_tp_cond + 1, 3].sum()
                        target = 1
                        exit_price_diff = take_profit
                        index_open_position = arg_buy_tp_cond + 1
                    else:
                        arg_buy_sl_cond = np.where(pip_diff_low <= -stop_loss+spread_pip)[0][0]
                        swap_days = selected_chunk[1 : arg_buy_sl_cond + 1, 3].sum()
                        target = -1
                        exit_price_diff = -stop_loss
                        index_open_position = arg_buy_sl_cond + 1

                elif buy_sl_cond.any():
                    arg_buy_sl_cond = np.where(pip_diff_low <= -stop_loss+spread_pip)[0][0]
                    swap_days = selected_chunk[1 : arg_buy_sl_cond + 1, 3].sum()
                    target = -1
                    exit_price_diff = -stop_loss
                    index_open_position = arg_buy_sl_cond + 1

                else:
                    target = 0
                    swap_days = selected_chunk[1:, 3].sum()
                    exit_price_diff = (selected_chunk[-1, 0] - selected_chunk[0, 0]) / symbol_decimal_multiply

                    if close_positions_at_midnight:
                        index_open_position = selected_chunk.shape[0]
                    else:
                        index_open_position = window_size

                target_list.append(target)
                swap_days_list.append(swap_days)
                exit_price_diff_list.append(exit_price_diff)
                time_open_position_list.append(index_open_position)
                stop_losses_list.append(stop_loss)
                take_profits_list.append(take_profit)

        elif mode == "short":
            for i in range(array.shape[0] - window_size):
                selected_chunk = array[i : i + window_size]

                if close_positions_at_midnight:
                    dates = date_column[i: i + window_size]
                    curr_date = dates[0]
                    selected_chunk = selected_chunk[dates == curr_date]

                pip_diff_high = (selected_chunk[1:, 1] - (selected_chunk[0, 0]+spread)) / symbol_decimal_multiply
                pip_diff_low = (selected_chunk[1:, 2] - (selected_chunk[0, 0]+spread)) / symbol_decimal_multiply

                if use_perc_levels:
                    curr_close = selected_chunk[0, 0]
                    take_profit = (curr_close / symbol_decimal_multiply) * take_profit_ratio
                    stop_loss = (curr_close / symbol_decimal_multiply) * stop_loss_ratio

                sell_tp_cond = pip_diff_low <= -take_profit-spread_pip
                sell_sl_cond = pip_diff_high >= stop_loss-spread_pip

                if sell_tp_cond.any():
                    arg_sell_tp_cond = np.where(pip_diff_low <= -take_profit-spread_pip)[0][0]
                    if not sell_sl_cond[: arg_sell_tp_cond + 1].any():
                        swap_days = selected_chunk[1 : arg_sell_tp_cond + 1, 3].sum()
                        target = 1
                        exit_price_diff = take_profit
                        index_open_position = arg_sell_tp_cond + 1
                    else:
                        arg_sell_sl_cond = np.where(pip_diff_high >= stop_loss-spread_pip)[0][0]
                        swap_days = selected_chunk[1 : arg_sell_sl_cond + 1, 3].sum()
                        target = -1
                        exit_price_diff = -stop_loss
                        index_open_position = arg_sell_sl_cond + 1

                elif sell_sl_cond.any():
                    arg_sell_sl_cond = np.where(pip_diff_high >= stop_loss-spread_pip)[0][0]
                    swap_days = selected_chunk[1 : arg_sell_sl_cond + 1, 3].sum()
                    target = -1
                    exit_price_diff = -stop_loss
                    index_open_position = arg_sell_sl_cond + 1

                else:
                    target = 0
                    swap_days = selected_chunk[1:, 3].sum()
                    exit_price_diff = (selected_chunk[0, 0] - selected_chunk[-1, 0]) / symbol_decimal_multiply

                    if close_positions_at_midnight:
                        index_open_position = selected_chunk.shape[0]
                    else:
                        index_open_position = window_size

                target_list.append(target)
                swap_days_list.append(swap_days)
                exit_price_diff_list.append(exit_price_diff)
                time_open_position_list.append(index_open_position)
                stop_losses_list.append(stop_loss)
                take_profits_list.append(take_profit)

    for _ in range(window_size):
        swap_days_list.append(None)
        target_list.append(None)
        exit_price_diff_list.append(None)
        time_open_position_list.append(None)
        stop_losses_list.append(None)
        take_profits_list.append(None)

    return (
        target_list,
        exit_price_diff_list,
        swap_days_list,
        time_open_position_list,
        stop_losses_list,
        take_profits_list
    )


def calculate_max_drawdown(balance_series, init_balance):
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

    # Calculate the drawdown ratio at each point in time
    drawdowns = (balance_series - cum_max) / cum_max

    # Return the maximum drawdown
    return drawdowns.min() * 100, (balance_series.min() - init_balance) * 100 / init_balance


def cal_backtest_on_raw_cndl(
    path: str,
    df_raw_path: str,
    target_symbol: str,
    look_ahead: int,
    take_profit: int,
    stop_loss: int,
    take_profit_perc: float,
    stop_loss_perc: float,
    use_perc_levels: bool,
    dynamic_sl_type: str,
    atr_window_size: int,
    atr_level_multiplication: float,
    spread: int,
    trade_mode: str,
    use_dynamic_sl: bool,
    max_strg_sl_dynamic_perc: int,
    dynamic_sl_scale_type: str,
    rstd_window_size: int,
    close_positions_at_midnight: bool,
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
    df = pd.read_parquet(path)

    df_raw_backtest = pd.read_parquet(
        f"{df_raw_path}/{target_symbol}_stage_one.parquet",
        columns=["_time", "open", "high", "low", "close"]
    ).rename(columns={
        "open": f"{target_symbol}_M5_OPEN",
        "high": f"{target_symbol}_M5_HIGH",
        "low": f"{target_symbol}_M5_LOW",
        "close": f"{target_symbol}_M5_CLOSE",  
    })

    df_raw_backtest.sort_values("_time", inplace=True)
    df_raw_backtest['days_diff'] = (
        df_raw_backtest['_time'].dt.date - df_raw_backtest['_time'].dt.date.shift()
    ).bfill().dt.days

    if use_dynamic_sl and dynamic_sl_type=="atr":
        col_name = f"fe_ATR_{target_symbol}_W{atr_window_size}_M5"

        if col_name not in df.columns:
            raise ValueError(f"{col_name} col not in the dataset.")

        array = df.merge(df_raw_backtest, on='_time', how='right')[
            [
                f"{target_symbol}_M5_CLOSE",
                f"{target_symbol}_M5_HIGH",
                f"{target_symbol}_M5_LOW",
                "days_diff",
                "_time",
                col_name
            ]
        ].to_numpy()

    elif use_dynamic_sl and dynamic_sl_type=="etr":
        col_name = f"fe_ETR_{target_symbol}_W{atr_window_size}_M5"

        array = df.merge(df_raw_backtest, on='_time', how='right')[
            [
                f"{target_symbol}_M5_CLOSE",
                f"{target_symbol}_M5_HIGH",
                f"{target_symbol}_M5_LOW",
                "days_diff",
                "_time",
                col_name
            ]
        ].to_numpy()

    else:
        array = df_raw_backtest[
            [
                f"{target_symbol}_M5_CLOSE",
                f"{target_symbol}_M5_HIGH",
                f"{target_symbol}_M5_LOW",
                "days_diff",
                "_time"
            ]
        ].to_numpy()

    (
        df_raw_backtest[bt_column_name],
        df_raw_backtest["pip_diff"],
        df_raw_backtest["swap_days"],
        df_raw_backtest["time_open_position"],
        df_raw_backtest["stop_losses"],
        df_raw_backtest["take_profits"]
    ) = calculate_classification_target_backtest(
        array,
        window_size,
        use_dynamic_sl=use_dynamic_sl,
        max_strg_sl_dynamic_perc=max_strg_sl_dynamic_perc,
        dynamic_sl_scale_type=dynamic_sl_scale_type,
        rstd_window_size=rstd_window_size,
        close_positions_at_midnight=close_positions_at_midnight,
        symbol_decimal_multiply=symbols_dict[target_symbol]["pip_size"],
        take_profit=take_profit,
        stop_loss=stop_loss,
        take_profit_perc=take_profit_perc,
        stop_loss_perc=stop_loss_perc,
        use_perc_levels=use_perc_levels,
        dynamic_sl_type=dynamic_sl_type,
        atr_level_multiplication=atr_level_multiplication,
        spread_pip=spread,
        mode=trade_mode,
    )
    df_raw_backtest.dropna(inplace=True)

    return df_raw_backtest, bt_column_name


def cal_sortino_ratio(balance: pd.Series):
    """
    This function calculates the sortino ratio of a given balance"""
    return_balance = balance.pct_change()
    mean_portfolio_return = return_balance.mean()
    risk_free_rate = 0
    downside_returns = np.minimum(return_balance - risk_free_rate, 0)
    downside_deviation = np.maximum(np.sqrt((downside_returns**2).mean()), 1e-5)
    sortino_ratio = (mean_portfolio_return - risk_free_rate) / downside_deviation

    return sortino_ratio


def plot_profit_distribution(df, bins=100, figsize=(9, 7)):
    """
    Plot the distribution of net profits from trading data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the trading data with 'net_profit' column
    bins : int, optional
        Number of bins for the histogram (default: 20)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (12, 6))
    """

    # Create figure and axis
    plt.figure(figsize=figsize)

    # Plot histogram
    _, bins, _ = plt.hist(df['net_profit'], bins=bins, 
                               edgecolor='black', alpha=0.7)

    # Add mean and median lines
    mean_profit = df['net_profit'].mean()
    median_profit = df['net_profit'].median()

    plt.axvline(mean_profit, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_profit:.1f} pips')
    plt.axvline(median_profit, color='green', linestyle='dashed', linewidth=2, 
                label=f'Median: {median_profit:.1f} pips')

    # Customize plot
    plt.title('Distribution of Trading Profits/Losses', fontsize=12, pad=15)
    plt.xlabel('Profit/Loss (pips)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add statistics as text
    stats_text = (
        f'Statistics:\n'
        f'Count: {len(df):,}\n'
        f'Std Dev: {df["net_profit"].std():.1f} pips\n'
        f'Min: {df["net_profit"].min():.1f} pips\n'
        f'Max: {df["net_profit"].max():.1f} pips'
    )

    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    return plt.gcf()


def money_management(
    df: pd.DataFrame,
    stop_loss: int,
    spread: int,
    initial_balance: int,
    accounts_leverage: int,
    target_symbol: str,
    pip_value: dict[str, float],
    n_max_OP: int,
    max_floating_dd: float,
    max_daily_dd: float,
    use_floating_risk: bool,
    use_dynamic_sl: bool,
    max_strg_sl_dynamic_perc: int,
    symbol_decimal_multiply: float,
    close_positions_at_midnight: bool,
    use_perc_levels: bool,
):
    symbols_base_lot = {
        'EURUSD': 0.01,
        'GBPUSD': 0.01,
        'USDJPY': 0.01,
        'XAUUSD': 0.01,
        'US30': 0.01,
        'US100': 0.01,
        'SPX500': 0.01,
        'BTCUSD': 0.01,
    }
    symbols_max_lot = {
        'EURUSD': (0.99*accounts_leverage) / 100000,
        'GBPUSD': (0.99*accounts_leverage) / 100000,
        'USDJPY': (0.99*accounts_leverage) / 100000,
        'XAUUSD': (0.99*accounts_leverage) / 100,
        'US30': (0.99*accounts_leverage) / 1,
        'US100': (0.99*accounts_leverage) / 1,
        'SPX500': (0.99*accounts_leverage) / 1,
        'BTCUSD': (0.99*accounts_leverage) / 1,
    }
    weights = {
        0: 0.0,
        0.5: 1.0,
        0.6: 1.0,
        0.7: 1.0,
        0.8: 1.25,
        0.9: 1.25,
        1: 1.0,
    }

    df['index'] = df.index
    df = df.sort_values(by="_time")
    df['close_position'] = df['index'] + df['time_open_position']
    df['position_closed'] = False
    array = df[
        [
            'index', '_time', 'close_position', 'volume', 'net_profit',
            'position_closed', 'confidence_levels', 'stop_losses', f"{target_symbol}_M5_CLOSE"
        ]
    ].to_numpy()

    max_strg_sl_dynamic_ratio = max_strg_sl_dynamic_perc / 100
    date_column = np.array(
        [np.datetime64(datetime, 'D') for datetime in array[:, 1]]
    )
    symbols_exp = 1/symbols_base_lot.get(target_symbol)
    max_open_volume_possible = symbols_max_lot.get(target_symbol)
    pip_risk = stop_loss + spread
    # max_pip_risk = max_strg_sl_dynamic_perc + spread
    start_day_balance = initial_balance
    floating_balance = initial_balance
    variable_balance = initial_balance
    n_open_position = []
    total_open_volume = []
    prev_date = np.datetime64(array[0, 1], 'D')
    day_before_prev_date = np.datetime64(array[0, 1], 'D')
    historic_closed_pos_cond = None
    aug_closed_pos_cond = None
    volumes = []
    start_day_balances = []
    floating_balances = []
    daily_dds = []
    daily_dd_exp = []
    used_balances = []
    remaining_positions = []
    used_dd_budgets = []
    max_exp_daily_dd = 0
    no_exceeding_dds = 0
    todays_exp_daily_dd = 0

    for i in range(array.shape[0]):
        chunk = array[:i+1]
        dates = date_column[:i+1]
        cond = chunk[:-1, 2] > chunk[-1, 0]
        open_cond = cond & (chunk[:-1, 3] != 0.0)
        cond_len = len(np.where(open_cond)[0])
        n_open_position.append(cond_len)
        chunk[:-1, 5][~cond] = True

        open_volumes = chunk[:-1, 3][open_cond]
        total_open_volume.append(open_volumes.sum())

        closed_pos_cond = (chunk[:-1, 5] == True) & (dates[:-1] == prev_date)

        if not close_positions_at_midnight:
            if day_before_prev_date != prev_date:
                historic_closed_pos_cond = (chunk[:-1, 5] == True) & (dates[:-1] == day_before_prev_date)
                historic_closed_pos_cond = historic_closed_pos_cond ^ aug_closed_pos_cond
                aug_closed_pos_cond = (chunk[:, 5] == True) & (dates[:] == day_before_prev_date)

                if historic_closed_pos_cond.any():
                    profits_n_losses = chunk[:-1, 3][historic_closed_pos_cond] * (
                        pip_value[target_symbol] * chunk[:-1, 4][historic_closed_pos_cond]
                    )
                    added_balance = profits_n_losses.sum()
                    variable_balance += added_balance

        profits_n_losses = chunk[:-1, 3][closed_pos_cond] * (
            pip_value[target_symbol] * chunk[:-1, 4][closed_pos_cond]
        )
        added_balance = profits_n_losses.sum()

        used_balance = (variable_balance-start_day_balance) + added_balance
        if used_balance < 0:
            dd = used_balance / start_day_balance

            if dd < todays_exp_daily_dd:
                todays_exp_daily_dd = dd

            if dd < max_exp_daily_dd:
                max_exp_daily_dd = dd

        floating_balance = start_day_balance + used_balance

        curr_date = np.datetime64(chunk[-1, 1], 'D')
        if curr_date != prev_date:
            start_day_balance += used_balance
            variable_balance = start_day_balance
            day_before_prev_date = prev_date
            prev_date = curr_date
            added_balance = 0
            todays_exp_daily_dd = 0

            if aug_closed_pos_cond is None:
                aug_closed_pos_cond = (chunk[:, 5] == True) & (dates[:] == day_before_prev_date)

        remaining_pos = n_max_OP - cond_len
        daily_dd_budget = (start_day_balance * max_daily_dd) + used_balance

        start_day_balances.append(start_day_balance)
        floating_balances.append(floating_balance)
        daily_dds.append(start_day_balance * max_daily_dd)
        daily_dd_exp.append(todays_exp_daily_dd)
        used_balances.append(used_balance)
        remaining_positions.append(remaining_pos)

        if target_symbol == 'USDJPY':
            max_vol = max_open_volume_possible * floating_balance
        else:
            max_vol = (max_open_volume_possible * floating_balance) / chunk[-1, 8]

        if use_dynamic_sl or use_perc_levels:
            used_dd_budget = chunk[:-1, 3][open_cond] * (pip_value[target_symbol] * chunk[:-1, 7][open_cond])
            used_dd_budget = used_dd_budget.sum()

            max_pip_risk = (chunk[-1, 8] / symbol_decimal_multiply) * max_strg_sl_dynamic_ratio
            max_pip_risk += spread

            if remaining_pos > 0:
                base_lot = (
                    (daily_dd_budget - used_dd_budget) / remaining_pos
                ) / (pip_value[target_symbol] * max_pip_risk)
        else:
            used_dd_budget = total_open_volume[i] * (pip_value[target_symbol] * pip_risk)

            if remaining_pos > 0:
                base_lot = (
                    (daily_dd_budget - used_dd_budget) / remaining_pos
                ) / (pip_value[target_symbol] * pip_risk)

        used_dd_budgets.append(used_dd_budget)
        potential_dd = (used_balance - used_dd_budget) / start_day_balance

        if (remaining_pos <= 0) or (potential_dd <= -max_daily_dd):
            volumes.append(0.0)
            array[i, 3] = volumes[i]

            if potential_dd <= -max_daily_dd:
                no_exceeding_dds += 1

            continue

        if use_floating_risk:
            floating_dd_budget = floating_balance * max_floating_dd

            if use_dynamic_sl or use_perc_levels:
                floating_base_lot = (
                    (floating_dd_budget - used_dd_budget) / remaining_pos
                ) / (pip_value[target_symbol] * max_pip_risk)
            else:
                floating_base_lot = (
                    (floating_dd_budget - used_dd_budget) / remaining_pos
                ) / (pip_value[target_symbol] * pip_risk)

            base_lot = min(base_lot, floating_base_lot)

        cnf_level_exp = weights.get(chunk[-1, 6])
        floored_base_lot = math.floor((cnf_level_exp*base_lot)*symbols_exp)/symbols_exp

        if (floored_base_lot <= 0) or (total_open_volume[i]+floored_base_lot >= max_vol):
            volumes.append(0.0)
            array[i, 3] = volumes[i]
            continue

        volumes.append(floored_base_lot)
        array[i, 3] = volumes[i]

    df["volume"] = np.array(volumes)
    df["n_open_position"] = np.array(n_open_position)
    df["volume_open_position"] = np.array(total_open_volume)
    df["start_day_balances"] = np.array(start_day_balances)
    df["floating_balances"] = np.array(floating_balances)
    df["daily_dds"] = np.array(daily_dds)
    df["daily_dd_exp"] = np.array(daily_dd_exp)
    df["used_balances"] = np.array(used_balances)
    df["remaining_positions"] = np.array(remaining_positions)
    df["used_dd_budgets"] = np.array(used_dd_budgets)

    return max_exp_daily_dd, no_exceeding_dds, df


def do_backtest(
    df_model_signal: pd.DataFrame,
    target_symbol: str,
    spread: float,
    volume: float,
    initial_balance: int,
    accounts_leverage: int,
    df_raw_backtest: pd.DataFrame,
    bt_column_name:   str,
    swap_rate: float,
    stop_loss: int,
    use_money_management: bool,
    n_max_OP: int,
    max_floating_dd: float,
    max_daily_dd: float,
    use_floating_risk: bool,
    use_dynamic_sl: bool,
    max_strg_sl_dynamic_perc: int,
    confidence_levels: np.ndarray,
    model,
    is_final_bt: bool,
    is_cf_model: bool,
    trade_mode: str,
    close_positions_at_midnight: bool,
    use_perc_levels: bool,
):
    pip_value = {
        'EURUSD': 10,
        'GBPUSD': 10,
        'USDJPY': 6.68,
        'XAUUSD': 1,
        'US30': 0.01,
        'US100': 0.01,
        'SPX500': 0.01,
        'BTCUSD': 0.01
    }

    symbol_decimal_multiply = symbols_dict[target_symbol]["pip_size"]
    df_model_signal = df_model_signal.reset_index().rename(columns={'index': '_time'})
    new_trg_df = df_model_signal.merge(df_raw_backtest, on="_time", how="inner")
    new_trg_df["net_profit"] = new_trg_df.pip_diff - spread
    new_trg_df["volume"] = volume
    new_trg_df["n_open_position"] = 0
    new_trg_df["volume_open_position"] = 0.0
    new_trg_df["start_day_balances"] = 0.0
    new_trg_df["floating_balances"] = 0.0
    new_trg_df["daily_dds"] = 0.0
    new_trg_df["daily_dd_exp"] = 0.0
    new_trg_df["used_balances"] = 0.0
    new_trg_df["remaining_positions"] = 0
    new_trg_df["used_dd_budgets"] = 0.0
    max_exp_daily_dd = 0.0
    no_exceeding_dds = 0

    plot_profit_distribution(new_trg_df)
    plt.show()

    if is_cf_model:
        if model.use_meta_labeling and not is_final_bt:
            new_trg_df["confidence_levels"] = confidence_levels

    if 'confidence_levels' in new_trg_df.columns:
        positive_profits = new_trg_df[new_trg_df['net_profit'] > 0]
        negative_profits = new_trg_df[new_trg_df['net_profit'] < 0]

        result_positive = positive_profits['confidence_levels'].value_counts(normalize=True) * 100
        result_negative = negative_profits['confidence_levels'].value_counts(normalize=True) * 100

        print(f"win trade:\n {result_positive}")
        print('**')
        print(f"loss trade:\n {result_negative}")
        print('==========')

        if use_money_management:
            max_exp_daily_dd, no_exceeding_dds, new_trg_df = money_management(
                new_trg_df, stop_loss, spread, initial_balance, accounts_leverage,
                target_symbol, pip_value, n_max_OP, max_floating_dd,
                max_daily_dd, use_floating_risk, use_dynamic_sl,
                max_strg_sl_dynamic_perc, symbol_decimal_multiply,
                close_positions_at_midnight, use_perc_levels
            )

            ##? calculate balance
            new_trg_df["profits_n_losses"] = new_trg_df["net_profit"] * new_trg_df["volume"] * pip_value[
                target_symbol
            ] + (new_trg_df["swap_days"] * new_trg_df["volume"] * swap_rate)
        else:
            ##? calculate balance
            new_trg_df["profits_n_losses"] = new_trg_df["net_profit"] * new_trg_df["volume"] * new_trg_df[
                "confidence_levels"
            ] * pip_value[target_symbol] + (new_trg_df["swap_days"] * new_trg_df["volume"] * swap_rate)

    else:
        if not is_final_bt:
            new_trg_df["confidence_levels"] = confidence_levels
        else:
            new_trg_df["confidence_levels"] = 1.0

        if use_money_management:
            max_exp_daily_dd, no_exceeding_dds, new_trg_df = money_management(
                new_trg_df, stop_loss, spread, initial_balance, accounts_leverage,
                target_symbol, pip_value, n_max_OP, max_floating_dd,
                max_daily_dd, use_floating_risk, use_dynamic_sl,
                max_strg_sl_dynamic_perc, symbol_decimal_multiply,
                close_positions_at_midnight, use_perc_levels
            )

        ##? calculate balance
        new_trg_df["profits_n_losses"] = new_trg_df["net_profit"] * new_trg_df["volume"] * pip_value[
            target_symbol
        ] + (new_trg_df["swap_days"] * new_trg_df["volume"] * swap_rate)

    new_trg_df["balance"] = new_trg_df["profits_n_losses"].cumsum()
    new_trg_df["balance"] += initial_balance

    new_trg_df["stop_losses"] *= symbol_decimal_multiply
    new_trg_df["take_profits"] *= symbol_decimal_multiply

    if trade_mode == 'long':
        price_spread = spread * symbol_decimal_multiply

        new_trg_df["stop_losses"] = (new_trg_df[f"{target_symbol}_M5_CLOSE"] + price_spread) - new_trg_df["stop_losses"]
        new_trg_df["take_profits"] = (new_trg_df[f"{target_symbol}_M5_CLOSE"] + price_spread) + new_trg_df["take_profits"]
    elif trade_mode == 'short':
        new_trg_df["stop_losses"] = new_trg_df[f"{target_symbol}_M5_CLOSE"] + new_trg_df["stop_losses"]
        new_trg_df["take_profits"] = new_trg_df[f"{target_symbol}_M5_CLOSE"] - new_trg_df["take_profits"]

    ##? calculate max_drawdown
    max_drawdown, max_overall_dd = calculate_max_drawdown(new_trg_df["balance"], initial_balance)

    ##? calculate duration:
    if new_trg_df.shape[0] == 0:
        backtest_report = {
            "balance_cash": initial_balance,
            "profit_pips": 0,
            "max_draw_down": 0,
            "sortino": 0.0,
            "win_rate(%)": 0.0,
            "profit_percent": 0,
            "max_exp_daily_dd": 0.0,
            "max_overall_dd": 0.0,
            "max_n_open_position": 0,
            "max_vol_open_positions": 0.0,
            "no_iters_exceeding_dd": 0,
        }
    else:
        backtest_report = {
            "balance_cash": int(new_trg_df.iloc[-1]["balance"]),
            "profit_pips": int(new_trg_df["net_profit"].sum()),
            "max_draw_down": round(max_drawdown, 2),
            "sortino": round(cal_sortino_ratio(new_trg_df["balance"]), 2),
            "win_rate(%)": round(
                (
                    len(
                        new_trg_df[(new_trg_df['volume'] > 0) & (new_trg_df['net_profit'] > 0)]
                        ) / len(new_trg_df[new_trg_df['volume'] > 0])
                    if len(new_trg_df[new_trg_df['volume'] > 0]) > 0 else 0.0
                ) * 100, 2
            ),
            "profit_percent": round(
                ((new_trg_df.iloc[-1]["balance"] - initial_balance) / initial_balance)
                * 100,
                2,
            ),
            "max_exp_daily_dd": round(max_exp_daily_dd*100, 2),
            "max_overall_dd": round(max_overall_dd, 2),
            "max_n_open_position": new_trg_df["n_open_position"].max(),
            "max_vol_open_positions": new_trg_df["volume_open_position"].max(),
            "no_iters_exceeding_dd": no_exceeding_dds,
        }

    return (
        backtest_report,
        new_trg_df[
            [
                "_time",
                "model_prediction",
                f"{bt_column_name}",
                "pip_diff",
                "net_profit",
                "profits_n_losses",
                "balance",
                "volume",
                "volume_open_position",
                "confidence_levels",
                "stop_losses",
                "take_profits",
                "start_day_balances",
                "floating_balances",
                "daily_dds",
                "daily_dd_exp",
                "used_balances",
                "remaining_positions",
                "used_dd_budgets",
            ]
        ],
    )
