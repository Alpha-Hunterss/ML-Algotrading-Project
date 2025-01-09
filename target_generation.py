import numpy as np


def calculate_classification_target_numpy_ver(
    array,
    window_size,
    symbol_decimal_multiply: float = 0.0001,
    take_profit: int = 70,
    stop_loss: int = 30,
    take_profit_perc: float = 0.1,
    stop_loss_perc: float = 0.033,
    use_perc_levels: bool = False,
    mode: str = "long",
):
    target_list = []
    take_profit_ratio = take_profit_perc / 100
    stop_loss_ratio = stop_loss_perc / 100

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

            if use_perc_levels:
                curr_close = selected_chunk[0, 0]
                take_profit = (curr_close / symbol_decimal_multiply) * take_profit_ratio
                stop_loss = (curr_close / symbol_decimal_multiply) * stop_loss_ratio

            buy_tp_cond = pip_diff_close >= take_profit
            buy_sl_cond = pip_diff_low <= -stop_loss

            if buy_tp_cond.any():
                arg_buy_tp_cond = np.where((pip_diff_close >= take_profit))[0][0]
                if not buy_sl_cond[0 : arg_buy_tp_cond + 1].any():
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

            if use_perc_levels:
                curr_close = selected_chunk[0, 0]
                take_profit = (curr_close / symbol_decimal_multiply) * take_profit_ratio
                stop_loss = (curr_close / symbol_decimal_multiply) * stop_loss_ratio

            sell_tp_cond = pip_diff_close <= -take_profit
            sell_sl_cond = pip_diff_high >= stop_loss

            if sell_tp_cond.any():
                arg_sell_tp_cond = np.where((pip_diff_close <= -take_profit))[0][0]
                if not sell_sl_cond[0 : arg_sell_tp_cond + 1].any():
                    target = 1

            target_list.append(target)

    for _ in range(window_size):
        target_list.append(None)

    return target_list
