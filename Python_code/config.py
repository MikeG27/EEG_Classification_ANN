import os

MAIN_DIR = os.getcwd()
PLOTS_DIR = os.path.join(MAIN_DIR, "plots")
DATA_DIR = os.path.join(MAIN_DIR, "dataset")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
DATA_REPORTS = os.path.join(DATA_DIR, "reports")