import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Our two CSV files
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# Model v2 paths for safe, uncorrupted retraining
MODEL_V2_PATH = os.path.join(MODEL_DIR, "hybrid_model_v2.h5")
TOKENIZER_V2_PATH = os.path.join(MODEL_DIR, "tokenizer_v2.pkl")

# Text preprocessing / tokenization
MAX_WORDS = 20000       # vocabulary size
MAX_LEN = 300           # max tokens per news article

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training hyperparameters
TEST_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 5
EMBEDDING_DIM = 128

# Prediction uncertainty bounds (calibrated on validation data)
UNCERTAINTY_LOW = 0.40
UNCERTAINTY_HIGH = 0.60

