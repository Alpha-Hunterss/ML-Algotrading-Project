from pathlib import Path
import os

# ? save path
root_path = str(os.path.dirname(os.path.abspath(__file__))).replace("configs", "")
stage_one_data_path = f"{root_path}/data/stage_one_data/"
Path(stage_one_data_path).mkdir(parents=True, exist_ok=True)