"""ML pipeline configuration — all settings read from environment variables."""
import os

ML_DEFAULT_TEST_SIZE = float(os.getenv("ML_DEFAULT_TEST_SIZE", "0.2"))
ML_DEFAULT_CV_FOLDS = int(os.getenv("ML_DEFAULT_CV_FOLDS", "5"))
ML_MAX_ONEHOT_CATEGORIES = int(os.getenv("ML_MAX_ONEHOT_CATEGORIES", "20"))
ML_DEFAULT_RANDOM_STATE = int(os.getenv("ML_DEFAULT_RANDOM_STATE", "42"))
ML_MAX_POLYNOMIAL_DEGREE = int(os.getenv("ML_MAX_POLYNOMIAL_DEGREE", "4"))
ML_MAX_PCA_COMPONENTS = int(os.getenv("ML_MAX_PCA_COMPONENTS", "50"))
