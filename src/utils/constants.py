"""Constants for the ViAG project."""

# Special tokens
CONTEXT_PREFIX = "context: "
QUESTION_PREFIX = "question: "
ANSWER_PREFIX = "answer: "

# Default file paths
DEFAULT_CONFIG_PATH = "configs/config.json"
DEFAULT_TRAIN_PATH = "datasets/train.csv"
DEFAULT_VAL_PATH = "datasets/val.csv"
DEFAULT_TEST_PATH = "datasets/test.csv"
DEFAULT_OUTPUT_DIR = "outputs"

# Model defaults
DEFAULT_MODEL_NAME = "VietAI/vit5-base"
DEFAULT_MAX_INPUT_LENGTH = 1024
DEFAULT_MAX_TARGET_LENGTH = 256

# Training defaults
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM_STEPS = 16
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_WEIGHT_DECAY = 0.01