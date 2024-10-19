from typing import Dict
from utils.feature_config_extractor.extract_config_from_features import extract_config_from_selected_feature
from utils.logging_tools import default_logger


def read_feature_config(logger=default_logger) -> Dict:
    """
    Reads a feature configuration from a specified JSON file, processes it, removes an unwanted key, and logs the process.

    Args:
        logger: Optional; a logging object used to log information. Defaults to `default_logger`.

    Returns:
        A dictionary containing the processed feature configuration with the "NOT_SYMBOL" key removed.
    """
    logger.info(f"= "*15)
    
    feature_map_path = "data/models/jamesv01/tradeset_usdjpy_feature_map.json"
    
    feature_config = extract_config_from_selected_feature(feature_info=feature_map_path)
    feature_config.pop("NOT_SYMBOL")
    
    logger.info(f"= "*15)
    
    return feature_config
