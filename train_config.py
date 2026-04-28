"""Configurações globais do treino e avaliação."""

OUTPUT_DIR = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Processados"
RESULTS_DIR = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Resultados_CNN_BaseSplit_Aux"

TRAIN_BASES = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia", "Chapman-Shaoxing-Ningbo"]
VAL_BASES = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia", "Chapman-Shaoxing-Ningbo"]
TEST_BASES = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia", "Chapman-Shaoxing-Ningbo"]

STRICT_BASE_SPLIT = False
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
SPLIT_SEED = 42
STRATIFY_INTERNAL_SPLIT = True

MAIN_SUPERCLASS_LIST = ["CD", "HYP", "MI", "NORM", "STTC"]
NUM_MAIN_CLASSES = len(MAIN_SUPERCLASS_LIST)
TARGET_LEADS = 12
TARGET_SAMPLES = 5000

USE_AUX_HEAD = True
AUX_LOSS_WEIGHT = 0.30
AUX_CLASS_LIST = ["RHYTHM", "FORM", "PACE"]
NUM_AUX_CLASSES = len(AUX_CLASS_LIST)
AUX_CLASS_ALIASES = {
    "RHYTHM": ["RHYTHM", "RHYTHM_ARRHYTHMIA", "Rhythm/Arrhythmia", "Rhythm", "Arrhythmia"],
    "FORM": ["FORM", "AXIS_FORM_VOLTAGE", "Axis/Form/Voltage Abnormality", "Axis/Form", "Form", "Axis", "Voltage"],
    "PACE": ["PACE", "PACED", "Paced Rhythm/Device Pattern", "Paced Rhythm", "Device Pattern", "Pacing"],
}
KEEP_SAMPLES_WITHOUT_AUX_LABEL = True

USE_META_FEATURES = False
META_FEATURE_DIM = 3

DATA_LOADING_MODE = "ram"
RAM_DTYPE = "float32"
NPZ_CACHE_SIZE = 4

EXCLUDE_WARNINGS = ["nan_inf", "flatline"]
MIN_POSITIVE_MAIN_LABELS = 1

BATCH_SIZE_TRAIN = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER = True
LR_FACTOR = 0.5
LR_PATIENCE = 5

USE_MAIN_CLASS_WEIGHTS = True
USE_AUX_CLASS_WEIGHTS = True

CNN_CHANNELS = [32, 64, 128]
KERNEL_SIZE = 7
POOL_SIZE = 2
DROPOUT_RATE = 0.3
FC_HIDDEN_DIM = 256

THRESHOLD_MAIN = 0.5
THRESHOLD_AUX = 0.5
SAVE_PREDICTIONS = True

GRADCAM_NUM_EXAMPLES_PER_CLASS = 2
GRADCAM_SMOOTHING_WINDOW = 25

SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = True
AGE_NORMALIZATION_DIVISOR = 100.0
