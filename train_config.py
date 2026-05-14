"""Configurações globais do treino e avaliação do modelo ECG.

Arquivo dedicado para centralizar todos os knobs do experimento.
"""

# ==============================
# Paths de entrada e saída
# ==============================
# Pasta gerada no preprocessamento (metadata.csv + batches/*.npz).
OUTPUT_DIR = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Processados"
# Pasta onde serão salvos checkpoints, métricas, plots e reports.
RESULTS_DIR = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Resultados_CNN_head"

# ==============================
# Split por base
# ==============================
# Bases usadas no treino.
TRAIN_BASES = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia","Chapman-Shaoxing-Ningbo"]
# Bases usadas na validação.
VAL_BASES = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia","Chapman-Shaoxing-Ningbo"]
# Bases usadas no teste.
TEST_BASES = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia","Chapman-Shaoxing-Ningbo"]

# Se True, proíbe a mesma base em mais de um split.
STRICT_BASE_SPLIT = False
# Razões internas quando uma mesma base aparece em múltiplos splits.
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
# Semente para split interno por base.
SPLIT_SEED = 42
# Se True, estratifica split interno pelo primeiro rótulo principal positivo.
STRATIFY_INTERNAL_SPLIT = True

# ==============================
# Classes alvo
# ==============================
# Superclasses principais da tarefa diagnóstica (head principal).
MAIN_SUPERCLASS_LIST = ["CD", "HYP", "MI", "NORM", "STTC"]
NUM_MAIN_CLASSES = len(MAIN_SUPERCLASS_LIST)

# Dimensões do sinal ECG de entrada.
TARGET_LEADS = 12
TARGET_SAMPLES = 5000

# ==============================
# Cabeça auxiliar (multitask)
# ==============================
# Ativa/desativa cabeça auxiliar.
USE_AUX_HEAD = True
# Peso da loss auxiliar na loss total: total = main + AUX_LOSS_WEIGHT * aux.
AUX_LOSS_WEIGHT = 0.20
# Classes da cabeça auxiliar.
AUX_CLASS_LIST = ["RHYTHM", "FORM", "PACE"]
NUM_AUX_CLASSES = len(AUX_CLASS_LIST)
# Aliases aceitos em metadata para mapear labels auxiliares.
AUX_CLASS_ALIASES = {
    "RHYTHM": ["RHYTHM", "RHYTHM_ARRHYTHMIA", "Rhythm/Arrhythmia", "Rhythm", "Arrhythmia"],
    "FORM": ["FORM", "AXIS_FORM_VOLTAGE", "Axis/Form/Voltage Abnormality", "Axis/Form", "Form", "Axis", "Voltage"],
    "PACE": ["PACE", "PACED", "Paced Rhythm/Device Pattern", "Paced Rhythm", "Device Pattern", "Pacing"],
}
# Se True, mantém exames sem label auxiliar (vetor auxiliar zerado).
KEEP_SAMPLES_WITHOUT_AUX_LABEL = False

# ==============================
# Features de metadados
# ==============================
# Se True, concatena metadados clínicos ao embedding da CNN.
USE_META_FEATURES = False
# Dimensão das features de metadados (age + sex one-hot).
META_FEATURE_DIM = 3

# ==============================
# Carregamento dos dados
# ==============================
# Modo: "ram" para pré-carregar split na RAM, "lazy" para carregar sob demanda.
DATA_LOADING_MODE = "ram"
# Tipo numérico usado no modo RAM.
RAM_DTYPE = "float32"
# Tamanho do cache LRU de npz no modo lazy.
NPZ_CACHE_SIZE = 4

# ==============================
# Data augmentation (treino)
# ==============================
ENABLE_ECG_AUGMENTATION = True
AUG_NOISE_PROB = 0.5
AUG_NOISE_STD = 0.01
AUG_SCALE_PROB = 0.5
AUG_SCALE_MIN = 0.8
AUG_SCALE_MAX = 1.2
AUG_SHIFT_PROB = 0.3
AUG_SHIFT_MAX = 50
AUG_LEAD_INVERT_PROB = 0.2

# ==============================
# Filtros de qualidade
# ==============================
# Warnings que removem exame durante preparação.
EXCLUDE_WARNINGS = ["nan_inf", "flatline"]
# Mínimo de classes principais positivas para manter amostra.
MIN_POSITIVE_MAIN_LABELS = 1

# ==============================
# Treinamento
# ==============================
BATCH_SIZE_TRAIN = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER = True
LR_FACTOR = 0.5
LR_PATIENCE = 5

# ==============================
# Balanceamento de classes
# ==============================
USE_MAIN_CLASS_WEIGHTS = True
USE_AUX_CLASS_WEIGHTS = True

# ==============================
# Arquitetura
# ==============================
CNN_CHANNELS = [32, 64, 128]
KERNEL_SIZE = 7
POOL_SIZE = 2
DROPOUT_RATE = 0.3
FC_HIDDEN_DIM = 256

# ==============================
# Avaliação
# ==============================
THRESHOLD_MAIN = 0.5
THRESHOLD_AUX = 0.5
USE_CLASSWISE_THRESHOLDS = False
OPTIMIZE_CLASSWISE_THRESHOLDS_ON_VAL = True
CLASSWISE_THRESHOLDS = {
    "CD": 0.5,
    "HYP": 0.5,
    "MI": 0.5,
    "NORM": 0.5,
    "STTC": 0.5,
}
SAVE_PREDICTIONS = True

# Flags legadas de avaliação avançada.
GENERATE_ADVANCED_EVAL = False
GENERATE_ROC_CURVES = False
GENERATE_PR_CURVES = False
GENERATE_CALIBRATION_PLOT = False
GENERATE_MULTILABEL_CONFUSION = False
GENERATE_GRADCAM = True
GENERATE_METADATA_INFLUENCE = False

# ==============================
# Grad-CAM
# ==============================
# Número de exemplos por superclasse (solicitado: 2 positivos por superclasse).
GRADCAM_NUM_EXAMPLES_PER_CLASS = 1
# Nome da camada alvo (mantido por compatibilidade).
GRADCAM_TARGET_LAYER_NAME = "feature_extractor"
# Janela de suavização do mapa 1D.
GRADCAM_SMOOTHING_WINDOW = 25

# ==============================
# Influência de metadados (ablação)
# ==============================
# Máximo de amostras para análise (None = usar todo teste).
METADATA_INFLUENCE_MAX_SAMPLES = None
# Valor baseline para zerar metadados na análise de influência.
METADATA_BASELINE_VALUE = 0.0

# ==============================
# Reprodutibilidade / runtime
# ==============================
SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = True
AGE_NORMALIZATION_DIVISOR = 100.0
