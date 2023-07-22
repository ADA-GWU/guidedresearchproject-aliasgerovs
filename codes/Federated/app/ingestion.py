import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import gzip
import math
pd.options.display.float_format = "{:,.4f}".format

def data_ingestion(DATA_PATH, FILENAME):
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")
    return x_train, y_train, x_valid, y_valid, x_test, y_test