import pandas as pd
from typing import List

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

def drop_features_and_get_final_fe(df_imp: pd.DataFrame,
                                   keep_ratio:float,\
                                     rand_drop_ratio: float,
                                     rand_drop_rows_threshold:float,
                                     random_seed: int)->List[str]:
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