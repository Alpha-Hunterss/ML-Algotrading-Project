import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Union
from sklearn.decomposition import PCA



def standardize_features(df: pd.DataFrame, save_path: str):

    """
    Standardize (normalize) the numerical features in a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the features to be standardized.
        
    Returns:
        pandas.DataFrame: A new DataFrame with standardized numerical features.
    """
    # Identify numerical columns
    # numerical_cols = df.select_dtypes(include=['float32', 'int8']).columns
    numerical_cols = [f for f in df.columns if f not in ['_time','target']]
    
    # Standardize numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    if save_path:
        joblib.dump(scaler, save_path)
    
    return df

def apply_pca(dataset: pd.DataFrame, not_feature_cols: List[str] , n_components: Union[float,int], save_path :str):
    """
    Apply PCA to the input dataset and return the transformed dataset.

    Args:
        dataset (pandas.DataFrame): Input dataset with features and the target variable column.
        not_feature_cols (List of str): columns that are not features including target and random features
        n_components (float or int, optional): Number of principal components to retain or the variance threshold to preserve. Defaults to 0.95.

    Returns:
        pandas.DataFrame: Transformed dataset with the target variable column.
    """
    pca = PCA(n_components=n_components)

    # Separate the features and non-features variables
    X = dataset.drop(columns = not_feature_cols)
    not_X = dataset[not_feature_cols]

    # Instantiate the PCA object
    pca = PCA(n_components=n_components)

    # Fit and transform the input features and save the PCA model
    X_transformed = pca.fit_transform(X)
    if save_path:
        joblib.dump(pca, save_path)

    # Create a DataFrame with the transformed features and the non-features variables
    X_transformed_df = pd.DataFrame(X_transformed, index=X.index)
    transformed_dataset = pd.concat([not_X, X_transformed_df], axis=1)
    transformed_dataset.rename(columns={f: f'feature_{f}' for f in transformed_dataset.columns if f not in not_feature_cols}, inplace=True)

    return transformed_dataset