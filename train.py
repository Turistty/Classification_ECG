# -*- coding: utf-8 -*-
"""
CNN 1D simples para classificação multi-label de ECG 12 derivações, com:

1) Split por base:
   você define explicitamente quais bases entram em treino, validação e teste.

2) Cabeça principal:
   diagnósticos-alvo:
     ["CD", "HYP", "MI", "NORM", "STTC"]

3) Cabeça auxiliar:
   achados não diagnósticos usados apenas como tarefa auxiliar:
     Rhythm/Arrhythmia, Axis/Form/Voltage Abnormality, Paced Rhythm/Device Pattern
   A cabeça auxiliar NÃO é usada como saída final. Ela só contribui para a loss.

4) Dois modos de leitura:
   DATA_LOADING_MODE = "lazy"
      lê os .npz sob demanda durante o treino.
      usa menos RAM, mais lento.

   DATA_LOADING_MODE = "ram"
      carrega os sinais filtrados na RAM antes do treino.
      usa muita RAM, mas treina mais rápido.

Entrada esperada do preprocessamento:
  OUTPUT_DIR/
    batches/
      batch_0000.npz com x: (N, 12, 5000), exam_ids
      ...
    metadata.csv

Saídas:
  RESULTS_DIR/
    best_model.pt
    last_model.pt
    config.json
    split_metadata.csv
    filtered_distribution_by_base.csv
    train_history.csv
    test_metrics_summary.csv
    test_per_class_metrics.csv
    test_predictions.csv
    plots/
      loss_curve.png
      f1_curve.png
      multilabel_confusion_matrices.png

Dependências:
  pip install numpy pandas torch scikit-learn matplotlib tqdm

Uso:
  python train_simple_cnn_ecg_base_split_aux.py
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# Variáveis globais
# =============================================================================

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR   = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Processados"
RESULTS_DIR  = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Resultados_CNN_BaseSplit_Aux"

# ── Split por base ─────────────────────────────────────────────────────────
# Use exatamente os nomes que aparecem em metadata.csv/source_base.
# Pelo seu relatório, os nomes são:
#   "Chapman-Shaoxing-Ningbo"
#   "cpsc_2018"
#   "cpsc_2018_extra"
#   "georgia"
#   "ptb"
#   "ptb-xl"
#
# Exemplo conservador de generalização cross-base:
#   treino: Chapman + CPSC + Georgia
#   validação: ptb-xl
#   teste: ptb
#
# Ajuste conforme o experimento desejado.
TRAIN_BASES = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia","Chapman-Shaoxing-Ningbo"]
VAL_BASES   = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia","Chapman-Shaoxing-Ningbo"]
TEST_BASES  = ["ptb-xl", "ptb", "cpsc_2018", "cpsc_2018_extra", "georgia","Chapman-Shaoxing-Ningbo"]


# Se True, impede que a mesma base apareça em mais de um split.
# Se False, permite usar a mesma base em train/val/test.
# Quando a mesma base aparece em mais de um split, o script faz split interno aleatório
# dentro dessa base, respeitando TRAIN_RATIO/VAL_RATIO/TEST_RATIO.
STRICT_BASE_SPLIT = False

# Razões usadas apenas para bases que aparecem em mais de um split.
# Exemplo:
#   TRAIN_BASES = ["ptb-xl"]
#   VAL_BASES   = ["ptb-xl"]
#   TEST_BASES  = ["ptb-xl"]
# então ptb-xl será dividida internamente em 70/15/15.
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20
SPLIT_SEED  = 42

# Se True, para bases compartilhadas entre splits, estratifica pelo primeiro label principal positivo.
STRATIFY_INTERNAL_SPLIT = True
PATIENT_LEVEL_SPLIT = True
PATIENT_ID_CSV_MAP = {
    "ptb-xl": "ptbxl_database.csv",
}

# ── Dados ──────────────────────────────────────────────────────────────────
MAIN_SUPERCLASS_LIST = ["CD", "HYP", "MI", "NORM", "STTC"]
NUM_MAIN_CLASSES     = len(MAIN_SUPERCLASS_LIST)

TARGET_LEADS    = 12
TARGET_SAMPLES  = 5000          # 500 Hz × 10 s

# ── Cabeça auxiliar: achados do ECG, não saída diagnóstica final ───────────
USE_AUX_HEAD = True
AUX_LOSS_WEIGHT = 0.15
AUX_LOSS_DECAY = True
AUX_LOSS_WEIGHT_FINAL = 0.05

# IDs canônicos da cabeça auxiliar. São nomes internos do treino.
AUX_CLASS_LIST = ["RHYTHM", "FORM", "PACE"]
NUM_AUX_CLASSES = len(AUX_CLASS_LIST)

# Mapeamento flexível: procura tanto em metadata["superclass_id"] quanto em
# metadata["superclass"]. Isso evita depender do nome exato usado no Map.csv.
AUX_CLASS_ALIASES = {
    "RHYTHM": [
        "RHYTHM", "RHYTHM_ARRHYTHMIA", "Rhythm/Arrhythmia",
        "Rhythm", "Arrhythmia"
    ],
    "FORM": [
        "FORM", "AXIS_FORM_VOLTAGE", "Axis/Form/Voltage Abnormality",
        "Axis/Form", "Form", "Axis", "Voltage"
    ],
    "PACE": [
        "PACE", "PACED", "Paced Rhythm/Device Pattern",
        "Paced Rhythm", "Device Pattern", "Pacing"
    ],
}

# Se True, exames sem nenhum label auxiliar continuam no treino com vetor auxiliar 0.
# Se False, esses exames seriam removidos. Para multitask, geralmente manter True.
KEEP_SAMPLES_WITHOUT_AUX_LABEL = True

# ── Metadados clínicos como features extras ────────────────────────────────
USE_META_FEATURES = False        # Se True, concatena age + sex ao vetor após CNN
META_FEATURE_DIM  = 3           # age (1) + sex one-hot (2: Male/Female, Unknown=00)

# ── Modo de carregamento ───────────────────────────────────────────────────
# "lazy": carrega cada batch .npz sob demanda com cache LRU.
# "ram" : carrega todos os exames filtrados do split na RAM antes do treino.
DATA_LOADING_MODE = "ram"      # "lazy" ou "ram"
RAM_DTYPE = "float32"           # "float32" recomendado. "float16" reduz RAM, pode perder precisão.
NPZ_CACHE_SIZE = 4              # usado apenas no modo lazy

# ── Exclusão de dados de baixa qualidade ──────────────────────────────────
EXCLUDE_WARNINGS = ["nan_inf", "flatline"]   # [] para usar todos
MIN_POSITIVE_MAIN_LABELS = 1                 # remove exames sem nenhuma classe principal

# ── Treinamento ────────────────────────────────────────────────────────────
BATCH_SIZE_TRAIN = 64
EPOCHS           = 50
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER     = True
LR_FACTOR        = 0.5
LR_PATIENCE      = 5
LR_SCHEDULER_TYPE = "cosine"   # "plateau" ou "cosine"
COSINE_T0 = 10
COSINE_T_MULT = 2
COSINE_ETA_MIN = 1e-6
WARMUP_EPOCHS = 3

# ── Balanceamento de classes ───────────────────────────────────────────────
USE_MAIN_CLASS_WEIGHTS = True
USE_AUX_CLASS_WEIGHTS  = True

# ── Arquitetura CNN ────────────────────────────────────────────────────────
CNN_CHANNELS   = [32, 64, 128]
KERNEL_SIZE    = 7
POOL_SIZE      = 2
DROPOUT_RATE   = 0.3
FC_HIDDEN_DIM  = 256
USE_RESIDUAL_BLOCKS = True
N_RESIDUAL_PER_STAGE = 1
USE_ATTENTION_POOL = True

# ── Avaliação ──────────────────────────────────────────────────────────────
THRESHOLD_MAIN = 0.5
THRESHOLD_AUX  = 0.5
USE_OPTIMAL_THRESHOLDS = True
USE_TEMPERATURE_SCALING = True
SAVE_PREDICTIONS = True
USE_AUGMENTATION = True
AUG_PROBABILITY = 0.5
AUG_LEAD_DROPOUT_PROB = 0.1

# Plots/artefatos avançados solicitados
GENERATE_ADVANCED_EVAL = True
GENERATE_CALIBRATION_PLOT = True
GENERATE_ROC_CURVES = True
GENERATE_PR_CURVES = True
GENERATE_MULTILABEL_CONFUSION = True
GENERATE_GRADCAM = True
GENERATE_METADATA_INFLUENCE = True

# Grad-CAM: escolhe automaticamente exemplo true positive de maior confiança por classe.
# Se não houver true positive para uma classe, usa o maior score previsto daquela classe.
GRADCAM_NUM_EXAMPLES_PER_CLASS = 1
GRADCAM_TARGET_LAYER_NAME = "feature_extractor"  # última Conv1d dentro do feature_extractor
GRADCAM_SMOOTHING_WINDOW = 25

# Influência dos metadados:
# mede diferença nas probabilidades quando metadados reais são substituídos por zero.
# Isso não é causalidade clínica; é uma análise de sensibilidade do modelo.
METADATA_INFLUENCE_MAX_SAMPLES = None  # None = test inteiro
METADATA_BASELINE_VALUE = 0.0

# ── Reprodutibilidade ──────────────────────────────────────────────────────
SEED = 42

# ── Outros ─────────────────────────────────────────────────────────────────
NUM_WORKERS = 0                 # Windows: 0 evita problemas com npz + multiprocessing.
PIN_MEMORY = True
AGE_NORMALIZATION_DIVISOR = 100.0


# =============================================================================
# Setup
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_pipe(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def contains_excluded_warning(warning_text: Any, excluded: Sequence[str]) -> bool:
    if not excluded:
        return False
    warnings = split_pipe(warning_text)
    for w in warnings:
        for bad in excluded:
            if w == bad or w.startswith(f"{bad}:") or bad in w:
                return True
    return False


def normalize_token(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")


def save_config(results_dir: Path) -> None:
    config = {
        "OUTPUT_DIR": OUTPUT_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "TRAIN_BASES": TRAIN_BASES,
        "VAL_BASES": VAL_BASES,
        "TEST_BASES": TEST_BASES,
        "STRICT_BASE_SPLIT": STRICT_BASE_SPLIT,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "SPLIT_SEED": SPLIT_SEED,
        "STRATIFY_INTERNAL_SPLIT": STRATIFY_INTERNAL_SPLIT,
        "PATIENT_LEVEL_SPLIT": PATIENT_LEVEL_SPLIT,
        "PATIENT_ID_CSV_MAP": PATIENT_ID_CSV_MAP,
        "MAIN_SUPERCLASS_LIST": MAIN_SUPERCLASS_LIST,
        "AUX_CLASS_LIST": AUX_CLASS_LIST,
        "AUX_CLASS_ALIASES": AUX_CLASS_ALIASES,
        "USE_AUX_HEAD": USE_AUX_HEAD,
        "AUX_LOSS_WEIGHT": AUX_LOSS_WEIGHT,
        "AUX_LOSS_DECAY": AUX_LOSS_DECAY,
        "AUX_LOSS_WEIGHT_FINAL": AUX_LOSS_WEIGHT_FINAL,
        "TARGET_LEADS": TARGET_LEADS,
        "TARGET_SAMPLES": TARGET_SAMPLES,
        "USE_META_FEATURES": USE_META_FEATURES,
        "META_FEATURE_DIM": META_FEATURE_DIM,
        "DATA_LOADING_MODE": DATA_LOADING_MODE,
        "RAM_DTYPE": RAM_DTYPE,
        "EXCLUDE_WARNINGS": EXCLUDE_WARNINGS,
        "MIN_POSITIVE_MAIN_LABELS": MIN_POSITIVE_MAIN_LABELS,
        "BATCH_SIZE_TRAIN": BATCH_SIZE_TRAIN,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "LR_SCHEDULER": LR_SCHEDULER,
        "LR_FACTOR": LR_FACTOR,
        "LR_PATIENCE": LR_PATIENCE,
        "LR_SCHEDULER_TYPE": LR_SCHEDULER_TYPE,
        "COSINE_T0": COSINE_T0,
        "COSINE_T_MULT": COSINE_T_MULT,
        "COSINE_ETA_MIN": COSINE_ETA_MIN,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "USE_MAIN_CLASS_WEIGHTS": USE_MAIN_CLASS_WEIGHTS,
        "USE_AUX_CLASS_WEIGHTS": USE_AUX_CLASS_WEIGHTS,
        "CNN_CHANNELS": CNN_CHANNELS,
        "KERNEL_SIZE": KERNEL_SIZE,
        "POOL_SIZE": POOL_SIZE,
        "DROPOUT_RATE": DROPOUT_RATE,
        "FC_HIDDEN_DIM": FC_HIDDEN_DIM,
        "USE_RESIDUAL_BLOCKS": USE_RESIDUAL_BLOCKS,
        "N_RESIDUAL_PER_STAGE": N_RESIDUAL_PER_STAGE,
        "USE_ATTENTION_POOL": USE_ATTENTION_POOL,
        "THRESHOLD_MAIN": THRESHOLD_MAIN,
        "THRESHOLD_AUX": THRESHOLD_AUX,
        "USE_OPTIMAL_THRESHOLDS": USE_OPTIMAL_THRESHOLDS,
        "USE_TEMPERATURE_SCALING": USE_TEMPERATURE_SCALING,
        "USE_AUGMENTATION": USE_AUGMENTATION,
        "AUG_PROBABILITY": AUG_PROBABILITY,
        "AUG_LEAD_DROPOUT_PROB": AUG_LEAD_DROPOUT_PROB,
        "SEED": SEED,
        "GENERATE_ADVANCED_EVAL": GENERATE_ADVANCED_EVAL,
        "GENERATE_CALIBRATION_PLOT": GENERATE_CALIBRATION_PLOT,
        "GENERATE_ROC_CURVES": GENERATE_ROC_CURVES,
        "GENERATE_PR_CURVES": GENERATE_PR_CURVES,
        "GENERATE_MULTILABEL_CONFUSION": GENERATE_MULTILABEL_CONFUSION,
        "GENERATE_GRADCAM": GENERATE_GRADCAM,
        "GENERATE_METADATA_INFLUENCE": GENERATE_METADATA_INFLUENCE,
        "GRADCAM_NUM_EXAMPLES_PER_CLASS": GRADCAM_NUM_EXAMPLES_PER_CLASS,
        "GRADCAM_TARGET_LAYER_NAME": GRADCAM_TARGET_LAYER_NAME,
        "GRADCAM_SMOOTHING_WINDOW": GRADCAM_SMOOTHING_WINDOW,
        "METADATA_INFLUENCE_MAX_SAMPLES": METADATA_INFLUENCE_MAX_SAMPLES,
        "METADATA_BASELINE_VALUE": METADATA_BASELINE_VALUE,
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Labels e metadados
# =============================================================================

def build_main_multilabel_vector(superclass_id_value: Any) -> np.ndarray:
    labels_present = set(split_pipe(superclass_id_value))
    y = np.zeros(NUM_MAIN_CLASSES, dtype=np.float32)
    for i, sc in enumerate(MAIN_SUPERCLASS_LIST):
        if sc in labels_present:
            y[i] = 1.0
    return y


def build_aux_multilabel_vector(superclass_id_value: Any, superclass_text_value: Any) -> np.ndarray:
    """
    Procura labels auxiliares em duas colunas:
      - superclass_id
      - superclass

    Isso permite funcionar mesmo que seu Map.csv use:
      superclass_id = RHYTHM / FORM / PACE
    ou apenas nomes longos:
      superclass = Rhythm/Arrhythmia etc.
    """
    raw_tokens = split_pipe(superclass_id_value) + split_pipe(superclass_text_value)
    norm_tokens = {normalize_token(t) for t in raw_tokens}

    y = np.zeros(NUM_AUX_CLASSES, dtype=np.float32)

    for i, aux_class in enumerate(AUX_CLASS_LIST):
        aliases = AUX_CLASS_ALIASES.get(aux_class, [aux_class])
        norm_aliases = {normalize_token(a) for a in aliases}

        matched = False
        for token in norm_tokens:
            if token in norm_aliases:
                matched = True
                break
            # Match parcial útil para nomes longos.
            for alias in norm_aliases:
                if alias and (alias in token or token in alias):
                    matched = True
                    break
            if matched:
                break

        if matched:
            y[i] = 1.0

    return y


def build_meta_features(age_value: Any, sex_value: Any) -> np.ndarray:
    try:
        age = float(age_value)
        if not np.isfinite(age):
            age = 0.0
    except Exception:
        age = 0.0

    age_norm = age / AGE_NORMALIZATION_DIVISOR

    sex = str(sex_value).strip().lower()
    male = 1.0 if sex == "male" else 0.0
    female = 1.0 if sex == "female" else 0.0

    return np.array([age_norm, male, female], dtype=np.float32)


def validate_base_split(metadata: pd.DataFrame) -> None:
    all_bases = set(metadata["source_base"].dropna().astype(str).unique())

    split_bases = {
        "train": set(TRAIN_BASES),
        "val": set(VAL_BASES),
        "test": set(TEST_BASES),
    }

    if STRICT_BASE_SPLIT:
        overlap_train_val = split_bases["train"].intersection(split_bases["val"])
        overlap_train_test = split_bases["train"].intersection(split_bases["test"])
        overlap_val_test = split_bases["val"].intersection(split_bases["test"])

        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                "Base presente em mais de um split e STRICT_BASE_SPLIT=True. "
                "Use STRICT_BASE_SPLIT=False para dividir internamente a mesma base. "
                f"train∩val={overlap_train_val}, train∩test={overlap_train_test}, val∩test={overlap_val_test}"
            )

    selected = split_bases["train"].union(split_bases["val"]).union(split_bases["test"])
    missing = selected.difference(all_bases)
    if missing:
        raise ValueError(f"As seguintes bases foram definidas mas não existem em metadata.csv/source_base: {sorted(missing)}")

    if not TRAIN_BASES:
        raise ValueError("TRAIN_BASES não pode ser vazio.")
    if not VAL_BASES:
        raise ValueError("VAL_BASES não pode ser vazio.")
    if not TEST_BASES:
        raise ValueError("TEST_BASES não pode ser vazio.")

    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO deve ser 1.0. Recebido: {ratio_sum}")


def _first_positive_main_label(row: pd.Series) -> str:
    for sc in MAIN_SUPERCLASS_LIST:
        if int(row.get(f"main_{sc}", 0)) == 1:
            return sc
    return "NO_LABEL"


def _split_indices_for_group(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide uma base internamente em train/val/test.
    Usado quando a mesma base aparece em mais de um split.
    """
    rng = np.random.default_rng(SPLIT_SEED)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    if STRATIFY_INTERNAL_SPLIT and "_internal_stratify_label" in group.columns:
        iterator = group.groupby("_internal_stratify_label")
    else:
        iterator = [("all", group)]

    for _, sub in iterator:
        idx = sub.index.to_numpy().copy()
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))

        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        train_indices.extend(idx[:n_train].tolist())
        val_indices.extend(idx[n_train:n_train + n_val].tolist())
        test_indices.extend(idx[n_train + n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def _split_patient_groups(group: pd.DataFrame, patient_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED)
    patients = group[patient_col].dropna().astype(str).unique().tolist()
    if not patients:
        return _split_indices_for_group(group)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    p_train = set(patients[:n_train])
    p_val = set(patients[n_train:n_train + n_val])
    p_test = set(patients[n_train + n_val:])

    train_idx = group.index[group[patient_col].astype(str).isin(p_train)].to_numpy()
    val_idx = group.index[group[patient_col].astype(str).isin(p_val)].to_numpy()
    test_idx = group.index[group[patient_col].astype(str).isin(p_test)].to_numpy()
    return train_idx, val_idx, test_idx


def assign_split_by_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split por base com dois comportamentos:

    1. Base exclusiva de um split:
       TRAIN_BASES=["chapman"], VAL_BASES=["ptb-xl"], TEST_BASES=["ptb"]
       -> cada base vai inteira para o split definido.

    2. Base repetida em mais de um split:
       TRAIN_BASES=["ptb-xl"], VAL_BASES=["ptb-xl"], TEST_BASES=["ptb-xl"]
       -> a base é dividida internamente em TRAIN_RATIO/VAL_RATIO/TEST_RATIO.

    Isso resolve o caso em que você quer testar o pipeline com uma única base.
    """
    df = df.copy()
    df["split"] = ""

    train_set = set(TRAIN_BASES)
    val_set = set(VAL_BASES)
    test_set = set(TEST_BASES)
    selected = train_set.union(val_set).union(test_set)

    # Remove bases não selecionadas.
    df = df[df["source_base"].isin(selected)].copy()

    if df.empty:
        raise ValueError("Nenhum exame sobrou após seleção por base. Verifique TRAIN_BASES/VAL_BASES/TEST_BASES.")

    base_to_splits: Dict[str, List[str]] = {}
    for base in selected:
        splits = []
        if base in train_set:
            splits.append("train")
        if base in val_set:
            splits.append("val")
        if base in test_set:
            splits.append("test")
        base_to_splits[base] = splits

    # Bases exclusivas entram inteiras no split correspondente.
    for base, splits in base_to_splits.items():
        if len(splits) == 1:
            df.loc[df["source_base"] == base, "split"] = splits[0]

    # Bases compartilhadas são divididas internamente.
    shared_bases = [base for base, splits in base_to_splits.items() if len(splits) > 1]
    for base in shared_bases:
        base_mask = df["source_base"] == base
        group = df.loc[base_mask].copy()
        patient_col = "_patient_id"
        if PATIENT_LEVEL_SPLIT and patient_col in group.columns and group[patient_col].notna().any():
            train_idx, val_idx, test_idx = _split_patient_groups(group, patient_col=patient_col)
            print(
                f"[MELHORIA] Split por patient_id ativo para base {base} "
                f"(N pacientes únicos: {group[patient_col].nunique():,})"
            )
        else:
            train_idx, val_idx, test_idx = _split_indices_for_group(group)

        # Só atribui aos splits em que a base foi declarada.
        # Se a base estiver em train/test mas não em val, a parte val é redistribuída para train.
        declared = set(base_to_splits[base])

        if "train" in declared:
            df.loc[train_idx, "split"] = "train"
        if "val" in declared:
            df.loc[val_idx, "split"] = "val"
        else:
            df.loc[val_idx, "split"] = "train" if "train" in declared else "test"
        if "test" in declared:
            df.loc[test_idx, "split"] = "test"
        else:
            df.loc[test_idx, "split"] = "train" if "train" in declared else "val"

    # Segurança: remove o que eventualmente ficou sem split.
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    if df.empty:
        raise ValueError("Nenhum exame sobrou após atribuição de split.")

    return df

def load_and_prepare_metadata(output_dir: Path, results_dir: Path) -> pd.DataFrame:
    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv não encontrado: {metadata_path}")

    df = pd.read_csv(metadata_path, dtype=str)

    required_cols = [
        "exam_id", "source_base", "batch_file", "batch_index",
        "age", "sex", "superclass_id", "superclass", "warning_exame"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv sem colunas obrigatórias: {missing}")

    initial_n = len(df)

    validate_base_split(df)

    # Primeiro seleciona somente as bases configuradas, mas ainda sem split final.
    selected_bases = set(TRAIN_BASES).union(set(VAL_BASES)).union(set(TEST_BASES))
    df = df[df["source_base"].isin(selected_bases)].copy()
    selected_n = len(df)

    if PATIENT_LEVEL_SPLIT:
        df["_patient_id"] = np.nan
        for base, csv_name in PATIENT_ID_CSV_MAP.items():
            sub_mask = df["source_base"].astype(str) == str(base)
            if not sub_mask.any():
                continue
            candidates = [
                output_dir / base / csv_name,
                output_dir / csv_name,
            ]
            patient_csv = next((p for p in candidates if p.exists()), None)
            if patient_csv is None:
                continue
            try:
                map_df = pd.read_csv(patient_csv, dtype=str)
                if "ecg_id" not in map_df.columns or "patient_id" not in map_df.columns:
                    continue
                map_df = map_df[["ecg_id", "patient_id"]].dropna()
                map_dict = dict(zip(map_df["ecg_id"].astype(str), map_df["patient_id"].astype(str)))
                df.loc[sub_mask, "_patient_id"] = df.loc[sub_mask, "exam_id"].astype(str).map(map_dict)
            except Exception as exc:
                print(f"[WARN] Falha ao carregar patient_id para {base}: {exc}")

    # Remove warnings definidos.
    if EXCLUDE_WARNINGS:
        mask_bad = df["warning_exame"].apply(lambda x: contains_excluded_warning(x, EXCLUDE_WARNINGS))
        df = df.loc[~mask_bad].copy()

    # Labels principais.
    main_labels = np.stack([build_main_multilabel_vector(v) for v in df["superclass_id"].values])
    df["_main_label_sum"] = main_labels.sum(axis=1)

    if MIN_POSITIVE_MAIN_LABELS > 0:
        df = df.loc[df["_main_label_sum"] >= MIN_POSITIVE_MAIN_LABELS].copy()
        main_labels = np.stack([build_main_multilabel_vector(v) for v in df["superclass_id"].values])

    # Labels auxiliares.
    aux_labels = np.stack([
        build_aux_multilabel_vector(row["superclass_id"], row["superclass"])
        for _, row in df.iterrows()
    ])
    df["_aux_label_sum"] = aux_labels.sum(axis=1)

    if USE_AUX_HEAD and not KEEP_SAMPLES_WITHOUT_AUX_LABEL:
        df = df.loc[df["_aux_label_sum"] > 0].copy()
        main_labels = np.stack([build_main_multilabel_vector(v) for v in df["superclass_id"].values])
        aux_labels = np.stack([
            build_aux_multilabel_vector(row["superclass_id"], row["superclass"])
            for _, row in df.iterrows()
        ])

    # Valida arquivos de batch.
    batches_dir = output_dir / "batches"
    df["_batch_path"] = df["batch_file"].apply(lambda x: str(batches_dir / str(x)))
    exists_mask = df["_batch_path"].apply(lambda p: Path(p).exists())
    if not exists_mask.all():
        missing_count = int((~exists_mask).sum())
        print(f"[WARN] {missing_count} linhas apontam para batch inexistente e serão removidas.")
        df = df.loc[exists_mask].copy()
        main_labels = np.stack([build_main_multilabel_vector(v) for v in df["superclass_id"].values])
        aux_labels = np.stack([
            build_aux_multilabel_vector(row["superclass_id"], row["superclass"])
            for _, row in df.iterrows()
        ])

    df["batch_index"] = df["batch_index"].astype(float).astype(int)
    df["_row_index"] = np.arange(len(df))

    # Salva colunas de labels.
    for i, sc in enumerate(MAIN_SUPERCLASS_LIST):
        df[f"main_{sc}"] = main_labels[:, i].astype(int)

    for i, ac in enumerate(AUX_CLASS_LIST):
        df[f"aux_{ac}"] = aux_labels[:, i].astype(int)

    # Label simples para estratificação interna quando a mesma base aparece em train/val/test.
    df["_internal_stratify_label"] = df.apply(_first_positive_main_label, axis=1)

    # Agora atribui split por base. Se a base estiver repetida em mais de um split,
    # o script divide internamente conforme TRAIN_RATIO/VAL_RATIO/TEST_RATIO.
    df = assign_split_by_base(df)

    print(f"Metadata inicial: {initial_n}")
    print(f"Após seleção por base: {selected_n}")
    print(f"Após filtros de qualidade/labels: {len(df)}")

    print("\nSplit por base:")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print(f"  {split}: n={len(sub)} | bases={sorted(sub['source_base'].unique().tolist())}")

    repeated_bases = set(TRAIN_BASES).intersection(VAL_BASES).union(
        set(TRAIN_BASES).intersection(TEST_BASES)
    ).union(set(VAL_BASES).intersection(TEST_BASES))
    if repeated_bases:
        print(f"\nBases repetidas entre splits detectadas: {sorted(repeated_bases)}")
        print(f"Foi aplicado split interno com TRAIN_RATIO={TRAIN_RATIO}, VAL_RATIO={VAL_RATIO}, TEST_RATIO={TEST_RATIO}.")

    print("\nLabels principais:")
    for sc in MAIN_SUPERCLASS_LIST:
        print(f"  {sc}: {int(df[f'main_{sc}'].sum())}")

    print("\nLabels auxiliares:")
    for ac in AUX_CLASS_LIST:
        print(f"  {ac}: {int(df[f'aux_{ac}'].sum())}")

    # Distribuição filtrada por base/split.
    dist_rows = []
    for (split, base), sub in df.groupby(["split", "source_base"]):
        row = {"split": split, "source_base": base, "n": len(sub)}
        for sc in MAIN_SUPERCLASS_LIST:
            row[f"main_{sc}"] = int(sub[f"main_{sc}"].sum())
        for ac in AUX_CLASS_LIST:
            row[f"aux_{ac}"] = int(sub[f"aux_{ac}"].sum())
        dist_rows.append(row)

    pd.DataFrame(dist_rows).to_csv(results_dir / "filtered_distribution_by_base.csv", index=False, encoding="utf-8-sig")
    df.to_csv(results_dir / "split_metadata.csv", index=False, encoding="utf-8-sig")

    # Checagem mínima.
    for split in ["train", "val", "test"]:
        if (df["split"] == split).sum() == 0:
            raise ValueError(f"Split {split} ficou vazio.")

    return df.reset_index(drop=True)


# =============================================================================
# Dataset: lazy e RAM
# =============================================================================

class NPZLRUCache:
    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self.cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        data = np.load(path, allow_pickle=True)
        x = data["x"]
        exam_ids = data["exam_ids"].astype(str) if "exam_ids" in data.files else np.array([])
        data.close()

        self.cache[path] = (x, exam_ids)
        self.cache.move_to_end(path)

        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return x, exam_ids


class ECGAugmentation:
    def __init__(self, p: float = 0.5, lead_dropout_prob: float = 0.1):
        self.p = p
        self.lead_dropout_prob = lead_dropout_prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32).copy()
        amp = float(np.max(np.abs(out)) + 1e-8)

        if np.random.rand() < self.p:
            out *= np.random.uniform(0.8, 1.2)
        if np.random.rand() < self.p:
            std = np.random.uniform(0.001, 0.01) * amp
            out += np.random.normal(0.0, std, size=out.shape).astype(np.float32)
        if np.random.rand() < self.p:
            t = np.arange(out.shape[1], dtype=np.float32) / 500.0
            freq = np.random.uniform(0.05, 0.5)
            bamp = np.random.uniform(0.0, 0.05) * amp
            out += (bamp * np.sin(2.0 * np.pi * freq * t))[None, :].astype(np.float32)
        if np.random.rand() < self.p:
            shift = int(np.random.randint(-250, 251))
            out = np.roll(out, shift=shift, axis=1)

        if np.random.rand() < self.lead_dropout_prob:
            valid = [i for i in range(out.shape[0]) if i != 1]  # nunca zerar derivação II
            if valid:
                lead_idx = int(np.random.choice(valid))
                out[lead_idx] = 0.0

        return out


class ECGLazyDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.cache = NPZLRUCache(max_size=NPZ_CACHE_SIZE)
        self.augmentation = ECGAugmentation(AUG_PROBABILITY, AUG_LEAD_DROPOUT_PROB)
        self.main_cols = [f"main_{sc}" for sc in MAIN_SUPERCLASS_LIST]
        self.aux_cols = [f"aux_{ac}" for ac in AUX_CLASS_LIST]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]

        x_batch, _ = self.cache.get(str(row["_batch_path"]))
        x = x_batch[int(row["batch_index"])]
        if USE_AUGMENTATION and str(row.get("split", "")) == "train":
            x = self.augmentation(x)

        if x.shape != (TARGET_LEADS, TARGET_SAMPLES):
            raise ValueError(f"Shape inválido para exam_id={row['exam_id']}: {x.shape}")

        y_main = row[self.main_cols].astype(float).to_numpy(dtype=np.float32)
        y_aux = row[self.aux_cols].astype(float).to_numpy(dtype=np.float32)
        meta = build_meta_features(row.get("age", ""), row.get("sex", ""))

        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "meta": torch.from_numpy(meta),
            "y_main": torch.from_numpy(y_main),
            "y_aux": torch.from_numpy(y_aux),
            "exam_id": str(row["exam_id"]),
        }


class ECGRamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split_name: str):
        self.df = df.reset_index(drop=True).copy()
        self.main_cols = [f"main_{sc}" for sc in MAIN_SUPERCLASS_LIST]
        self.aux_cols = [f"aux_{ac}" for ac in AUX_CLASS_LIST]
        self.split_name = split_name
        self.augmentation = ECGAugmentation(AUG_PROBABILITY, AUG_LEAD_DROPOUT_PROB)

        dtype = np.float16 if RAM_DTYPE.lower() == "float16" else np.float32

        x_list: List[np.ndarray] = []
        print(f"\nCarregando split '{split_name}' na RAM: {len(self.df)} exames | dtype={dtype}")

        # Agrupa por batch para ler cada .npz uma vez.
        for batch_path, sub in tqdm(self.df.groupby("_batch_path"), desc=f"RAM load {split_name}"):
            data = np.load(batch_path, allow_pickle=True)
            xb = data["x"]
            indices = sub["batch_index"].astype(int).to_numpy()
            x_list.append(np.asarray(xb[indices], dtype=dtype))
            data.close()

        self.x = np.concatenate(x_list, axis=0)
        if self.x.shape[1:] != (TARGET_LEADS, TARGET_SAMPLES):
            raise ValueError(f"RAM dataset com shape inválido: {self.x.shape}")

        self.y_main = self.df[self.main_cols].astype(float).to_numpy(dtype=np.float32)
        self.y_aux = self.df[self.aux_cols].astype(float).to_numpy(dtype=np.float32)
        self.meta = np.stack([build_meta_features(r.get("age", ""), r.get("sex", "")) for _, r in self.df.iterrows()])
        self.exam_ids = self.df["exam_id"].astype(str).tolist()

        gb = self.x.nbytes / (1024 ** 3)
        print(f"Split '{split_name}' carregado: x.shape={self.x.shape} | RAM aproximada={gb:.2f} GB")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        x = np.asarray(self.x[index], dtype=np.float32)
        if USE_AUGMENTATION and self.split_name == "train":
            x = self.augmentation(x)
        return {
            "x": torch.from_numpy(x),
            "meta": torch.from_numpy(self.meta[index]),
            "y_main": torch.from_numpy(self.y_main[index]),
            "y_aux": torch.from_numpy(self.y_aux[index]),
            "exam_id": self.exam_ids[index],
        }


def make_dataset(df: pd.DataFrame, split_name: str) -> Dataset:
    mode = DATA_LOADING_MODE.lower().strip()
    if mode == "lazy":
        return ECGLazyDataset(df)
    if mode == "ram":
        return ECGRamDataset(df, split_name=split_name)
    raise ValueError(f"DATA_LOADING_MODE inválido: {DATA_LOADING_MODE}. Use 'lazy' ou 'ram'.")


# =============================================================================
# Modelo
# =============================================================================

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        p = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=p, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=p, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(1, channels // 4)
        self.attn = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x), dim=-1)
        return (x * w).sum(dim=-1)

class SimpleECGCNNMultiTask(nn.Module):
    def __init__(
        self,
        num_main_classes: int,
        num_aux_classes: int,
        in_leads: int,
        cnn_channels: Sequence[int],
        kernel_size: int,
        pool_size: int,
        dropout_rate: float,
        fc_hidden_dim: int,
        use_meta_features: bool,
        meta_feature_dim: int,
        use_aux_head: bool,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        in_ch = in_leads
        padding = kernel_size // 2

        for out_ch in cnn_channels:
            if USE_RESIDUAL_BLOCKS:
                layers.extend([
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                ])
                for _ in range(N_RESIDUAL_PER_STAGE):
                    layers.append(ResBlock1D(out_ch, kernel_size=kernel_size, dropout=dropout_rate))
                layers.extend([
                    nn.MaxPool1d(kernel_size=pool_size),
                    nn.Dropout(dropout_rate),
                ])
            else:
                layers.extend([
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=pool_size),
                    nn.Dropout(dropout_rate),
                ])
            in_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = TemporalAttentionPool(cnn_channels[-1]) if USE_ATTENTION_POOL else nn.AdaptiveAvgPool1d(1)

        self.use_meta_features = use_meta_features
        self.use_aux_head = use_aux_head

        feature_dim = cnn_channels[-1]
        if use_meta_features:
            feature_dim += meta_feature_dim

        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.main_head = nn.Linear(fc_hidden_dim, num_main_classes)

        if use_aux_head:
            self.aux_head = nn.Linear(fc_hidden_dim, num_aux_classes)
        else:
            self.aux_head = None

    def extract_features_before_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna o mapa convolucional final antes do pooling.
        Usado para Grad-CAM 1D.
        Shape: (B, C, T_reduzido)
        """
        return self.feature_extractor(x)

    def forward_from_conv_features(
        self,
        conv_features: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pooled = self.global_pool(conv_features)
        z = pooled.squeeze(-1) if pooled.dim() == 3 else pooled

        if self.use_meta_features:
            if meta is None:
                raise ValueError("USE_META_FEATURES=True, mas meta=None.")
            z = torch.cat([z, meta], dim=1)

        z = self.shared_fc(z)
        main_logits = self.main_head(z)
        aux_logits = self.aux_head(z) if self.aux_head is not None else None
        return main_logits, aux_logits

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        conv_features = self.extract_features_before_pool(x)
        return self.forward_from_conv_features(conv_features, meta)

# =============================================================================
# Métricas
# =============================================================================

def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    metrics["subset_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        metrics["auroc_macro"] = float(roc_auc_score(y_true, y_prob, average="macro"))
    except Exception:
        metrics["auroc_macro"] = float("nan")

    try:
        metrics["auprc_macro"] = float(average_precision_score(y_true, y_prob, average="macro"))
    except Exception:
        metrics["auprc_macro"] = float("nan")

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)
    rows = []

    for i, cls in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        prob = y_prob[:, i]

        row = {
            "class": cls,
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }

        try:
            row["auroc"] = float(roc_auc_score(yt, prob))
        except Exception:
            row["auroc"] = float("nan")

        try:
            row["auprc"] = float(average_precision_score(yt, prob))
        except Exception:
            row["auprc"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def predict_with_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_temperature: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            y = batch["y_main"].cpu().numpy()
            logits, _ = model(x, meta if USE_META_FEATURES else None)
            if use_temperature and hasattr(model, "temperature_scaler") and model.temperature_scaler is not None:
                logits = model.temperature_scaler(logits)
            logits_list.append(logits.cpu().numpy())
            y_list.append(y)
    logits = np.concatenate(logits_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)
    return {"logits": logits, "y_true": y_true, "prob": sigmoid_np(logits)}


def find_optimal_thresholds(y_val_true: np.ndarray, y_val_prob: np.ndarray) -> Dict[str, float]:
    thresholds = np.arange(0.05, 0.951, 0.01)
    best: Dict[str, float] = {}
    for i, cls in enumerate(MAIN_SUPERCLASS_LIST):
        yt = y_val_true[:, i]
        yp = y_val_prob[:, i]
        scores = [f1_score(yt, (yp >= t).astype(int), zero_division=0) for t in thresholds]
        best[cls] = float(thresholds[int(np.argmax(scores))])
    return best


def apply_thresholds(y_prob: np.ndarray, threshold_map: Dict[str, float]) -> np.ndarray:
    preds = np.zeros_like(y_prob, dtype=int)
    for i, cls in enumerate(MAIN_SUPERCLASS_LIST):
        preds[:, i] = (y_prob[:, i] >= threshold_map.get(cls, THRESHOLD_MAIN)).astype(int)
    return preds


def compute_ece_multilabel(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    eces = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(float)
        pr = y_prob[:, i].astype(float)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            mask = (pr >= lo) & (pr <= hi) if b == n_bins - 1 else (pr >= lo) & (pr < hi)
            if mask.sum() == 0:
                continue
            conf = float(pr[mask].mean())
            acc = float(yt[mask].mean())
            ece += (mask.sum() / len(pr)) * abs(acc - conf)
        eces.append(ece)
    return float(np.mean(eces)) if eces else float("nan")


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=1e-3)


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            y = batch["y_main"].to(device, non_blocking=True)
            logits, _ = model(x, meta if USE_META_FEATURES else None)
            logits_list.append(logits.detach())
            labels_list.append(y.detach())

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    scaler = TemperatureScaler().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(scaler(all_logits), all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    model.temperature_scaler = scaler
    return float(torch.clamp(scaler.temperature.detach().cpu(), min=1e-3).item())


# =============================================================================
# Treino / avaliação
# =============================================================================

def calculate_pos_weight(df: pd.DataFrame, cols: Sequence[str], class_names: Sequence[str], title: str) -> torch.Tensor:
    y = df[list(cols)].astype(float).to_numpy(dtype=np.float32)

    pos = y.sum(axis=0)
    neg = y.shape[0] - pos

    pos_weight = neg / np.clip(pos, 1.0, None)
    pos_weight = np.clip(pos_weight, 1.0, 100.0)

    print(f"\npos_weight {title}:")
    for cls, w, p in zip(class_names, pos_weight, pos):
        print(f"  {cls}: {w:.4f} | positives={int(p)}")

    return torch.tensor(pos_weight, dtype=torch.float32)


def build_criteria(train_df: pd.DataFrame, device: torch.device) -> Tuple[nn.Module, Optional[nn.Module]]:
    main_cols = [f"main_{sc}" for sc in MAIN_SUPERCLASS_LIST]
    aux_cols = [f"aux_{ac}" for ac in AUX_CLASS_LIST]

    if USE_MAIN_CLASS_WEIGHTS:
        main_pos_weight = calculate_pos_weight(train_df, main_cols, MAIN_SUPERCLASS_LIST, "main").to(device)
        main_criterion = nn.BCEWithLogitsLoss(pos_weight=main_pos_weight)
    else:
        main_criterion = nn.BCEWithLogitsLoss()

    aux_criterion = None
    if USE_AUX_HEAD:
        if USE_AUX_CLASS_WEIGHTS:
            aux_pos_weight = calculate_pos_weight(train_df, aux_cols, AUX_CLASS_LIST, "aux").to(device)
            aux_criterion = nn.BCEWithLogitsLoss(pos_weight=aux_pos_weight)
        else:
            aux_criterion = nn.BCEWithLogitsLoss()

    return main_criterion, aux_criterion


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    main_criterion: nn.Module,
    aux_criterion: Optional[nn.Module],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    aux_loss_weight: Optional[float] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_n = 0

    all_main_logits: List[np.ndarray] = []
    all_aux_logits: List[np.ndarray] = []
    all_y_main: List[np.ndarray] = []
    all_y_aux: List[np.ndarray] = []
    all_exam_ids: List[str] = []

    pbar = tqdm(loader, desc="train" if is_train else "eval", leave=False)

    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        y_main = batch["y_main"].to(device, non_blocking=True)
        y_aux = batch["y_aux"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            main_logits, aux_logits = model(x, meta if USE_META_FEATURES else None)

            main_loss = main_criterion(main_logits, y_main)
            aux_loss = torch.tensor(0.0, device=device)

            if USE_AUX_HEAD and aux_logits is not None and aux_criterion is not None:
                aux_loss = aux_criterion(aux_logits, y_aux)
                cur_aux_weight = AUX_LOSS_WEIGHT if aux_loss_weight is None else aux_loss_weight
                loss = main_loss + cur_aux_weight * aux_loss
            else:
                loss = main_loss

            if is_train:
                loss.backward()
                optimizer.step()

        n = x.size(0)
        total_loss += float(loss.item()) * n
        total_main_loss += float(main_loss.item()) * n
        total_aux_loss += float(aux_loss.item()) * n
        total_n += n

        all_main_logits.append(main_logits.detach().cpu().numpy())
        all_y_main.append(y_main.detach().cpu().numpy())
        all_y_aux.append(y_aux.detach().cpu().numpy())
        all_exam_ids.extend(batch["exam_id"])

        if aux_logits is not None:
            all_aux_logits.append(aux_logits.detach().cpu().numpy())

        pbar.set_postfix(loss=total_loss / max(1, total_n))

    result = {
        "loss": total_loss / max(1, total_n),
        "main_loss": total_main_loss / max(1, total_n),
        "aux_loss": total_aux_loss / max(1, total_n),
        "main_logits": np.concatenate(all_main_logits, axis=0),
        "y_main": np.concatenate(all_y_main, axis=0),
        "y_aux": np.concatenate(all_y_aux, axis=0),
        "exam_ids": all_exam_ids,
    }

    if all_aux_logits:
        result["aux_logits"] = np.concatenate(all_aux_logits, axis=0)
    else:
        result["aux_logits"] = None

    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_main_metrics: Dict[str, float],
    val_aux_metrics: Optional[Dict[str, float]],
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_main_metrics": val_main_metrics,
        "val_aux_metrics": val_aux_metrics,
        "main_superclass_list": MAIN_SUPERCLASS_LIST,
        "aux_class_list": AUX_CLASS_LIST,
        "config": {
            "TARGET_LEADS": TARGET_LEADS,
            "TARGET_SAMPLES": TARGET_SAMPLES,
            "USE_META_FEATURES": USE_META_FEATURES,
            "META_FEATURE_DIM": META_FEATURE_DIM,
            "USE_AUX_HEAD": USE_AUX_HEAD,
            "AUX_LOSS_WEIGHT": AUX_LOSS_WEIGHT,
            "CNN_CHANNELS": CNN_CHANNELS,
            "KERNEL_SIZE": KERNEL_SIZE,
            "POOL_SIZE": POOL_SIZE,
            "DROPOUT_RATE": DROPOUT_RATE,
            "FC_HIDDEN_DIM": FC_HIDDEN_DIM,
            "USE_RESIDUAL_BLOCKS": USE_RESIDUAL_BLOCKS,
            "N_RESIDUAL_PER_STAGE": N_RESIDUAL_PER_STAGE,
            "USE_ATTENTION_POOL": USE_ATTENTION_POOL,
        }
    }, path)


def make_loader(df: pd.DataFrame, split_name: str, shuffle: bool, device: torch.device) -> DataLoader:
    ds = make_dataset(df, split_name=split_name)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        drop_last=False,
    )


def train_model(
    df: pd.DataFrame,
    results_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, pd.DataFrame]:
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    train_loader = make_loader(train_df, "train", shuffle=True, device=device)
    val_loader = make_loader(val_df, "val", shuffle=False, device=device)

    model = SimpleECGCNNMultiTask(
        num_main_classes=NUM_MAIN_CLASSES,
        num_aux_classes=NUM_AUX_CLASSES,
        in_leads=TARGET_LEADS,
        cnn_channels=CNN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        pool_size=POOL_SIZE,
        dropout_rate=DROPOUT_RATE,
        fc_hidden_dim=FC_HIDDEN_DIM,
        use_meta_features=USE_META_FEATURES,
        meta_feature_dim=META_FEATURE_DIM,
        use_aux_head=USE_AUX_HEAD,
    ).to(device)

    main_criterion, aux_criterion = build_criteria(train_df, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    warmup_scheduler = None
    cosine_scheduler = None
    if LR_SCHEDULER:
        if LR_SCHEDULER_TYPE == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=LR_FACTOR,
                patience=LR_PATIENCE,
            )
        else:
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda ep: (0.1 + 0.9 * ((ep + 1) / max(1, WARMUP_EPOCHS))),
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=COSINE_T0,
                T_mult=COSINE_T_MULT,
                eta_min=COSINE_ETA_MIN,
            )

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    print("\nIniciando treinamento.")
    print(f"Device: {device}")
    print(f"DATA_LOADING_MODE: {DATA_LOADING_MODE}")
    print(f"LR scheduler: {'habilitado' if LR_SCHEDULER else 'desabilitado'} | tipo={LR_SCHEDULER_TYPE}")
    print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        if AUX_LOSS_DECAY:
            progress = (epoch - 1) / max(1, EPOCHS - 1)
            aux_weight = AUX_LOSS_WEIGHT + progress * (AUX_LOSS_WEIGHT_FINAL - AUX_LOSS_WEIGHT)
        else:
            aux_weight = AUX_LOSS_WEIGHT

        train_out = run_epoch(model, train_loader, main_criterion, aux_criterion, device, optimizer, aux_loss_weight=aux_weight)
        val_out = run_epoch(model, val_loader, main_criterion, aux_criterion, device, optimizer=None)

        train_main_prob = sigmoid_np(train_out["main_logits"])
        train_main_metrics = compute_metrics(train_out["y_main"], train_main_prob, threshold=THRESHOLD_MAIN)

        val_main_prob = sigmoid_np(val_out["main_logits"])
        val_main_metrics = compute_metrics(val_out["y_main"], val_main_prob, threshold=THRESHOLD_MAIN)

        train_aux_metrics = None
        val_aux_metrics = None
        if USE_AUX_HEAD and train_out["aux_logits"] is not None and val_out["aux_logits"] is not None:
            train_aux_prob = sigmoid_np(train_out["aux_logits"])
            val_aux_prob = sigmoid_np(val_out["aux_logits"])
            train_aux_metrics = compute_metrics(train_out["y_aux"], train_aux_prob, threshold=THRESHOLD_AUX)
            val_aux_metrics = compute_metrics(val_out["y_aux"], val_aux_prob, threshold=THRESHOLD_AUX)

        if LR_SCHEDULER:
            if LR_SCHEDULER_TYPE == "plateau" and scheduler is not None:
                scheduler.step(val_out["loss"])
            else:
                if epoch <= WARMUP_EPOCHS and warmup_scheduler is not None:
                    warmup_scheduler.step()
                elif cosine_scheduler is not None:
                    cosine_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss_total": train_out["loss"],
            "train_loss_main": train_out["main_loss"],
            "train_loss_aux": train_out["aux_loss"],
            "val_loss_total": val_out["loss"],
            "val_loss_main": val_out["main_loss"],
            "val_loss_aux": val_out["aux_loss"],
            "train_main_f1_macro": train_main_metrics["f1_macro"],
            "train_main_f1_micro": train_main_metrics["f1_micro"],
            "val_main_f1_macro": val_main_metrics["f1_macro"],
            "val_main_f1_micro": val_main_metrics["f1_micro"],
            "val_main_auroc_macro": val_main_metrics["auroc_macro"],
            "val_main_auprc_macro": val_main_metrics["auprc_macro"],
            "elapsed_sec": elapsed,
        }

        if val_aux_metrics is not None:
            row.update({
                "train_aux_f1_macro": train_aux_metrics["f1_macro"],
                "val_aux_f1_macro": val_aux_metrics["f1_macro"],
                "val_aux_f1_micro": val_aux_metrics["f1_micro"],
                "val_aux_auroc_macro": val_aux_metrics["auroc_macro"],
            })

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"lr={lr:.2e} | "
            f"train_loss={train_out['loss']:.5f} "
            f"(main={train_out['main_loss']:.5f}, aux={train_out['aux_loss']:.5f}) | "
            f"val_loss={val_out['loss']:.5f} "
            f"(main={val_out['main_loss']:.5f}, aux={val_out['aux_loss']:.5f}) | "
            f"val_main_f1_macro={val_main_metrics['f1_macro']:.4f} | "
            f"val_main_f1_micro={val_main_metrics['f1_micro']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_out["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_out["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                results_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                val_out["loss"],
                val_main_metrics,
                val_aux_metrics,
            )
            print(f"  -> best_model.pt salvo. val_loss={val_out['loss']:.5f}")
        else:
            epochs_without_improvement += 1

        pd.DataFrame(history).to_csv(results_dir / "train_history.csv", index=False, encoding="utf-8-sig")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping na época {epoch}. Melhor época: {best_epoch}.")
            break

    save_checkpoint(
        results_dir / "last_model.pt",
        model,
        optimizer,
        epoch,
        val_out["loss"],
        val_main_metrics,
        val_aux_metrics,
    )

    ckpt = torch.load(results_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    return model, pd.DataFrame(history)


def evaluate_test(
    model: nn.Module,
    df: pd.DataFrame,
    results_dir: Path,
    device: torch.device,
) -> None:
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    val_loader = make_loader(val_df, "val", shuffle=False, device=device)
    test_loader = make_loader(test_df, "test", shuffle=False, device=device)

    baseline_out = predict_with_model(model, test_loader, device=device, use_temperature=False)
    baseline_prob = baseline_out["prob"]
    baseline_metrics = compute_per_class_metrics(baseline_out["y_true"], baseline_prob, MAIN_SUPERCLASS_LIST, THRESHOLD_MAIN)

    main_criterion = nn.BCEWithLogitsLoss()
    aux_criterion = nn.BCEWithLogitsLoss() if USE_AUX_HEAD else None

    test_out = run_epoch(model, test_loader, main_criterion, aux_criterion, device, optimizer=None)

    if USE_TEMPERATURE_SCALING:
        t_value = calibrate_temperature(model, val_loader, device)
        print(f"[MELHORIA] Temperature scaling habilitado (T = {t_value:.3f})")
    else:
        t_value = 1.0

    main_logits_eval = test_out["main_logits"]
    if USE_TEMPERATURE_SCALING and hasattr(model, "temperature_scaler"):
        with torch.no_grad():
            logits_t = torch.from_numpy(main_logits_eval).to(device)
            main_logits_eval = model.temperature_scaler(logits_t).cpu().numpy()
    main_prob = sigmoid_np(main_logits_eval)

    threshold_map = {cls: THRESHOLD_MAIN for cls in MAIN_SUPERCLASS_LIST}
    if USE_OPTIMAL_THRESHOLDS:
        val_pred = predict_with_model(model, val_loader, device=device, use_temperature=USE_TEMPERATURE_SCALING)
        threshold_map = find_optimal_thresholds(val_pred["y_true"], val_pred["prob"])
        print("[MELHORIA] Thresholds otimizados: " + ", ".join([f"{k}={v:.2f}" for k, v in threshold_map.items()]))
        (results_dir / "optimal_thresholds.json").write_text(
            json.dumps(threshold_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    main_pred = apply_thresholds(main_prob, threshold_map)

    main_metrics = compute_metrics(test_out["y_main"], main_prob, threshold=THRESHOLD_MAIN)
    main_metrics["subset_accuracy"] = float(accuracy_score(test_out["y_main"], main_pred))
    main_metrics["f1_micro"] = float(f1_score(test_out["y_main"], main_pred, average="micro", zero_division=0))
    main_metrics["f1_macro"] = float(f1_score(test_out["y_main"], main_pred, average="macro", zero_division=0))
    main_metrics["precision_micro"] = float(precision_score(test_out["y_main"], main_pred, average="micro", zero_division=0))
    main_metrics["precision_macro"] = float(precision_score(test_out["y_main"], main_pred, average="macro", zero_division=0))
    main_metrics["recall_micro"] = float(recall_score(test_out["y_main"], main_pred, average="micro", zero_division=0))
    main_metrics["recall_macro"] = float(recall_score(test_out["y_main"], main_pred, average="macro", zero_division=0))
    main_metrics["test_loss_total"] = float(test_out["loss"])
    main_metrics["test_loss_main"] = float(test_out["main_loss"])
    main_metrics["test_loss_aux"] = float(test_out["aux_loss"])
    main_metrics["threshold_main"] = float(THRESHOLD_MAIN)
    main_metrics["ece_before"] = compute_ece_multilabel(test_out["y_main"], baseline_prob)
    main_metrics["ece_after"] = compute_ece_multilabel(test_out["y_main"], main_prob)
    main_metrics["temperature"] = float(t_value)
    main_metrics["n_test"] = int(len(test_out["y_main"]))

    aux_metrics = {}
    aux_prob = None
    aux_pred = None
    if USE_AUX_HEAD and test_out["aux_logits"] is not None:
        aux_prob = sigmoid_np(test_out["aux_logits"])
        aux_pred = (aux_prob >= THRESHOLD_AUX).astype(int)
        aux_metrics = compute_metrics(test_out["y_aux"], aux_prob, threshold=THRESHOLD_AUX)
        aux_metrics = {f"aux_{k}": v for k, v in aux_metrics.items()}

    summary = dict(main_metrics)
    summary.update(aux_metrics)
    pd.DataFrame([summary]).to_csv(results_dir / "test_metrics_summary.csv", index=False, encoding="utf-8-sig")

    per_class_main = compute_per_class_metrics(test_out["y_main"], main_prob, MAIN_SUPERCLASS_LIST, THRESHOLD_MAIN)
    for i, cls in enumerate(MAIN_SUPERCLASS_LIST):
        per_class_main.loc[per_class_main["class"] == cls, "threshold"] = threshold_map.get(cls, THRESHOLD_MAIN)
        per_class_main.loc[per_class_main["class"] == cls, "f1_thresholded"] = f1_score(
            test_out["y_main"][:, i],
            main_pred[:, i],
            zero_division=0
        )
    per_class_main.to_csv(results_dir / "test_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    if aux_prob is not None:
        per_class_aux = compute_per_class_metrics(test_out["y_aux"], aux_prob, AUX_CLASS_LIST, THRESHOLD_AUX)
        per_class_aux.to_csv(results_dir / "test_aux_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    if SAVE_PREDICTIONS:
        rows = []
        exam_ids = test_out["exam_ids"]
        for i, exam_id in enumerate(exam_ids):
            row = {"exam_id": exam_id}

            for c, sc in enumerate(MAIN_SUPERCLASS_LIST):
                row[f"true_main_{sc}"] = int(test_out["y_main"][i, c])
                row[f"prob_main_{sc}"] = float(main_prob[i, c])
                row[f"pred_main_{sc}"] = int(main_pred[i, c])

            if aux_prob is not None and aux_pred is not None:
                for c, ac in enumerate(AUX_CLASS_LIST):
                    row[f"true_aux_{ac}"] = int(test_out["y_aux"][i, c])
                    row[f"prob_aux_{ac}"] = float(aux_prob[i, c])
                    row[f"pred_aux_{ac}"] = int(aux_pred[i, c])

            rows.append(row)

        pd.DataFrame(rows).to_csv(results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    calibration_payload = {
        "temperature": float(t_value),
        "ece_before": float(main_metrics["ece_before"]),
        "ece_after": float(main_metrics["ece_after"]),
    }
    (results_dir / "calibration_temperature.json").write_text(
        json.dumps(calibration_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    baseline_metrics = baseline_metrics.rename(columns={
        "precision": "precision_baseline",
        "recall": "recall_baseline",
        "f1": "f1_baseline",
        "auroc": "auroc_baseline",
        "auprc": "auprc_baseline",
    })
    improved_metrics = per_class_main.rename(columns={
        "precision": "precision_improved",
        "recall": "recall_improved",
        "f1": "f1_improved",
        "auroc": "auroc_improved",
        "auprc": "auprc_improved",
    })
    comp = baseline_metrics.merge(improved_metrics, on=["class"], how="outer")
    comp.to_csv(results_dir / "metrics_comparison.csv", index=False, encoding="utf-8-sig")

    print("\nTeste - cabeça principal:")
    for k, v in main_metrics.items():
        print(f"  {k}: {v}")

    if aux_metrics:
        print("\nTeste - cabeça auxiliar:")
        for k, v in aux_metrics.items():
            print(f"  {k}: {v}")

    print("\nMétricas por classe principal:")
    print(per_class_main)

    plot_results(results_dir, test_out["y_main"], main_prob, MAIN_SUPERCLASS_LIST, THRESHOLD_MAIN, "main")
    plot_calibration(
        y_true=test_out["y_main"],
        y_prob=baseline_prob,
        class_names=MAIN_SUPERCLASS_LIST,
        output_path=results_dir / "plots" / "calibration_before_temperature.png",
    )
    plot_calibration(
        y_true=test_out["y_main"],
        y_prob=main_prob,
        class_names=MAIN_SUPERCLASS_LIST,
        output_path=results_dir / "plots" / "calibration_after_temperature.png",
    )

    run_advanced_evaluation_outputs(
        model=model,
        test_df=test_df,
        y_true=test_out["y_main"],
        y_prob=main_prob,
        results_dir=results_dir,
        device=device,
    )

    build_metrics_json(
        y_true=test_out["y_main"],
        y_prob=main_prob,
        class_names=MAIN_SUPERCLASS_LIST,
        threshold=THRESHOLD_MAIN,
        output_path=results_dir / "metrics_improved.json",
        extra={
            "thresholds": threshold_map,
            "temperature": t_value,
            "ece_before": main_metrics["ece_before"],
            "ece_after": main_metrics["ece_after"],
        },
    )


# =============================================================================
# Plots
# =============================================================================

def plot_history(results_dir: Path, history: pd.DataFrame) -> None:
    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)

    if history.empty:
        return

    fig = plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_loss_total"], label="train_total")
    plt.plot(history["epoch"], history["val_loss_total"], label="val_total")
    plt.plot(history["epoch"], history["train_loss_main"], label="train_main")
    plt.plot(history["epoch"], history["val_loss_main"], label="val_main")
    if "val_loss_aux" in history.columns:
        plt.plot(history["epoch"], history["val_loss_aux"], label="val_aux")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png", dpi=130)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_main_f1_macro"], label="train_main_f1_macro")
    plt.plot(history["epoch"], history["val_main_f1_macro"], label="val_main_f1_macro")
    plt.plot(history["epoch"], history["val_main_f1_micro"], label="val_main_f1_micro")
    if "val_aux_f1_macro" in history.columns:
        plt.plot(history["epoch"], history["val_aux_f1_macro"], label="val_aux_f1_macro")
    plt.xlabel("Época")
    plt.ylabel("F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "f1_curve.png", dpi=130)
    plt.close(fig)


def plot_results(
    results_dir: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
    prefix: str,
) -> None:
    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)

    y_pred = (y_prob >= threshold).astype(int)
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for i, cls in enumerate(class_names):
        ax = axes[i]
        cm = mcm[i]
        ax.imshow(cm)
        ax.set_title(cls)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center")

    fig.suptitle(f"Matrizes de confusão multi-label - {prefix}")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_multilabel_confusion_matrices.png", dpi=130)
    plt.close(fig)


# =============================================================================
# Avaliação avançada, curvas, calibração, Grad-CAM e influência dos metadados
# =============================================================================

def _safe_auc(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
    try:
        if len(np.unique(y_true_col)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true_col, y_prob_col))
    except Exception:
        return float("nan")


def _safe_auprc(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
    try:
        if len(np.unique(y_true_col)) < 2:
            return float("nan")
        return float(average_precision_score(y_true_col, y_prob_col))
    except Exception:
        return float("nan")


def build_metrics_json(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
    output_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    global_metrics = compute_metrics(y_true, y_prob, threshold=threshold)

    per_class: Dict[str, Dict[str, Any]] = {}
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    for i, cls in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()
        yt = y_true[:, i]
        yp = y_pred[:, i]
        prob = y_prob[:, i]

        try:
            brier = float(brier_score_loss(yt, prob)) if len(np.unique(yt)) > 1 else float("nan")
        except Exception:
            brier = float("nan")

        per_class[cls] = {
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "specificity": float(tn / max(1, tn + fp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "auroc": _safe_auc(yt, prob),
            "auprc": _safe_auprc(yt, prob),
            "brier": brier,
            "mean_probability_positive": float(prob[yt == 1].mean()) if np.any(yt == 1) else float("nan"),
            "mean_probability_negative": float(prob[yt == 0].mean()) if np.any(yt == 0) else float("nan"),
        }

    payload = {
        "threshold": float(threshold),
        "n_samples": int(y_true.shape[0]),
        "classes": list(class_names),
        "global": global_metrics,
        "per_class": per_class,
    }
    if extra:
        payload["extra"] = extra

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 7))

    for i, cls in enumerate(class_names):
        yt = y_true[:, i]
        prob = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, prob)
        auc = roc_auc_score(yt, prob)
        plt.plot(fpr, tpr, label=f"{cls} AUC={auc:.3f}")

    # micro-average
    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_prob.ravel())
        auc_micro = roc_auc_score(y_true, y_prob, average="micro")
        plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro AUC={auc_micro:.3f}")
    except Exception:
        pass

    plt.plot([0, 1], [0, 1], linestyle=":", label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves - superclasses multilabel")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 7))

    for i, cls in enumerate(class_names):
        yt = y_true[:, i]
        prob = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(yt, prob)
        ap = average_precision_score(yt, prob)
        plt.plot(recall, precision, label=f"{cls} AP={ap:.3f}")

    try:
        precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
        ap_micro = average_precision_score(y_true, y_prob, average="micro")
        plt.plot(recall_micro, precision_micro, linestyle="--", label=f"micro AP={ap_micro:.3f}")
    except Exception:
        pass

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves - superclasses multilabel")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    fig = plt.figure(figsize=(8, 7))

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for i, cls in enumerate(class_names):
        yt = y_true[:, i].astype(float)
        prob = y_prob[:, i].astype(float)

        bin_centers = []
        frac_pos = []
        mean_pred = []

        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            if b == n_bins - 1:
                mask = (prob >= lo) & (prob <= hi)
            else:
                mask = (prob >= lo) & (prob < hi)

            if mask.sum() == 0:
                continue

            mp = float(prob[mask].mean())
            fp = float(yt[mask].mean())
            center = float((lo + hi) / 2.0)

            bin_centers.append(center)
            mean_pred.append(mp)
            frac_pos.append(fp)

            rows.append({
                "class": cls,
                "bin": b,
                "bin_low": lo,
                "bin_high": hi,
                "n": int(mask.sum()),
                "mean_predicted_probability": mp,
                "fraction_positive": fp,
            })

        if mean_pred:
            plt.plot(mean_pred, frac_pos, marker="o", label=cls)

    plt.plot([0, 1], [0, 1], linestyle=":", label="perfect calibration")
    plt.xlabel("Probabilidade média prevista")
    plt.ylabel("Fração observada positiva")
    plt.title("Calibration plot - superclasses multilabel")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    return df


def plot_multilabel_confusion_matrices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
    output_path: Path,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for i, cls in enumerate(class_names):
        ax = axes[i]
        cm = mcm[i]
        ax.imshow(cm)
        ax.set_title(cls)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(int(cm[r, c])), ha="center", va="center")

    fig.suptitle("Matriz de confusão por superclasse - multilabel")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def get_sample_from_dataset(dataset: Dataset, index: int) -> Dict[str, Any]:
    item = dataset[index]
    return item


def smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = min(window, len(x))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def compute_gradcam_1d(
    model: SimpleECGCNNMultiTask,
    x: torch.Tensor,
    meta: torch.Tensor,
    target_class_idx: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM 1D sobre o último mapa convolucional.
    Retorna:
      cam_upsampled: shape (TARGET_SAMPLES,)
      signal_np: shape (12, TARGET_SAMPLES)
    """
    model.eval()

    x = x.unsqueeze(0).to(device)
    meta = meta.unsqueeze(0).to(device)

    conv_features = model.extract_features_before_pool(x)
    conv_features.retain_grad()

    main_logits, _ = model.forward_from_conv_features(conv_features, meta if USE_META_FEATURES else None)
    score = main_logits[0, target_class_idx]

    model.zero_grad(set_to_none=True)
    score.backward()

    grads = conv_features.grad.detach()          # (1, C, T')
    acts = conv_features.detach()                # (1, C, T')

    weights = grads.mean(dim=2, keepdim=True)    # (1, C, 1)
    cam = torch.sum(weights * acts, dim=1).squeeze(0)  # (T',)
    cam = torch.relu(cam)

    cam_np = cam.detach().cpu().numpy()
    if np.max(cam_np) > 0:
        cam_np = cam_np / (np.max(cam_np) + 1e-12)

    cam_t = torch.from_numpy(cam_np).view(1, 1, -1).float()
    cam_up = torch.nn.functional.interpolate(
        cam_t,
        size=TARGET_SAMPLES,
        mode="linear",
        align_corners=False,
    ).view(-1).numpy()

    cam_up = smooth_1d(cam_up, GRADCAM_SMOOTHING_WINDOW)
    if np.max(cam_up) > 0:
        cam_up = cam_up / (np.max(cam_up) + 1e-12)

    signal_np = x.detach().cpu().numpy()[0]
    return cam_up.astype(np.float32), signal_np.astype(np.float32)


def plot_gradcam_example(
    signal_12xT: np.ndarray,
    cam: np.ndarray,
    class_name: str,
    probability: float,
    exam_id: str,
    true_label: int,
    output_path: Path,
) -> None:
    t = np.arange(TARGET_SAMPLES) / 500.0
    fig, axes = plt.subplots(12, 1, figsize=(15, 16), sharex=True)

    for i, lead in enumerate(["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]):
        ax = axes[i]
        sig = signal_12xT[i]

        ax.plot(t, sig, linewidth=0.7)
        ymin, ymax = float(np.min(sig)), float(np.max(sig))
        if ymin == ymax:
            ymin, ymax = ymin - 1.0, ymax + 1.0

        ax.imshow(
            cam.reshape(1, -1),
            aspect="auto",
            extent=[t[0], t[-1], ymin, ymax],
            alpha=0.35,
            origin="lower",
        )

        ax.set_ylabel(lead, rotation=0, labelpad=18)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    axes[-1].set_xlabel("Tempo (s)")
    fig.suptitle(
        f"Grad-CAM 1D | classe={class_name} | prob={probability:.4f} | true={true_label} | exam_id={exam_id}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def generate_gradcam_examples(
    model: SimpleECGCNNMultiTask,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    results_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    gradcam_dir = results_dir / "plots" / "gradcam"
    ensure_dir(gradcam_dir)

    dataset = make_dataset(test_df.reset_index(drop=True), split_name="gradcam_test")
    rows: List[Dict[str, Any]] = []

    for class_idx, cls in enumerate(class_names):
        yt = y_true[:, class_idx]
        prob = y_prob[:, class_idx]

        tp_candidates = np.where((yt == 1) & (prob >= THRESHOLD_MAIN))[0]
        if len(tp_candidates) > 0:
            ordered = tp_candidates[np.argsort(prob[tp_candidates])[::-1]]
            candidate_type = "high_confidence_true_positive"
        else:
            ordered = np.argsort(prob)[::-1]
            candidate_type = "highest_probability_no_tp_available"

        selected = ordered[:GRADCAM_NUM_EXAMPLES_PER_CLASS]

        for rank, sample_idx in enumerate(selected):
            item = get_sample_from_dataset(dataset, int(sample_idx))
            cam, signal_np = compute_gradcam_1d(
                model=model,
                x=item["x"],
                meta=item["meta"],
                target_class_idx=class_idx,
                device=device,
            )

            exam_id = str(item["exam_id"])
            probability = float(prob[sample_idx])
            true_label = int(yt[sample_idx])

            out_path = gradcam_dir / f"gradcam_{cls}_rank{rank}_{exam_id}.png"
            plot_gradcam_example(
                signal_12xT=signal_np,
                cam=cam,
                class_name=cls,
                probability=probability,
                exam_id=exam_id,
                true_label=true_label,
                output_path=out_path,
            )

            rows.append({
                "class": cls,
                "rank": rank,
                "exam_id": exam_id,
                "sample_index": int(sample_idx),
                "probability": probability,
                "true_label": true_label,
                "candidate_type": candidate_type,
                "plot_path": str(out_path),
            })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "gradcam_examples.csv", index=False, encoding="utf-8-sig")
    return df


def evaluate_metadata_influence(
    model: SimpleECGCNNMultiTask,
    test_df: pd.DataFrame,
    results_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Avalia sensibilidade às features de metadados:
      prob(real meta) - prob(meta zerado)

    Também mede ablações individuais:
      age=0 mantendo sex
      sex=0 mantendo age

    Isso ajuda a ver se o modelo está dependendo demais de idade/sexo.
    """
    if not USE_META_FEATURES:
        payload = {"enabled": False, "reason": "USE_META_FEATURES=False"}
        (results_dir / "metadata_influence.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    eval_df = test_df.reset_index(drop=True).copy()
    if METADATA_INFLUENCE_MAX_SAMPLES is not None and len(eval_df) > METADATA_INFLUENCE_MAX_SAMPLES:
        eval_df = eval_df.sample(n=METADATA_INFLUENCE_MAX_SAMPLES, random_state=SEED).reset_index(drop=True)

    loader = DataLoader(
        make_dataset(eval_df, split_name="metadata_influence"),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        drop_last=False,
    )

    model.eval()

    all_real: List[np.ndarray] = []
    all_zero: List[np.ndarray] = []
    all_age_zero: List[np.ndarray] = []
    all_sex_zero: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_exam_ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="metadata influence", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            y = batch["y_main"].cpu().numpy()
            all_y.append(y)
            all_exam_ids.extend(batch["exam_id"])

            logits_real, _ = model(x, meta)
            all_real.append(torch.sigmoid(logits_real).cpu().numpy())

            meta_zero = torch.full_like(meta, float(METADATA_BASELINE_VALUE))
            logits_zero, _ = model(x, meta_zero)
            all_zero.append(torch.sigmoid(logits_zero).cpu().numpy())

            meta_age_zero = meta.clone()
            meta_age_zero[:, 0] = float(METADATA_BASELINE_VALUE)
            logits_age_zero, _ = model(x, meta_age_zero)
            all_age_zero.append(torch.sigmoid(logits_age_zero).cpu().numpy())

            meta_sex_zero = meta.clone()
            meta_sex_zero[:, 1:] = float(METADATA_BASELINE_VALUE)
            logits_sex_zero, _ = model(x, meta_sex_zero)
            all_sex_zero.append(torch.sigmoid(logits_sex_zero).cpu().numpy())

    prob_real = np.concatenate(all_real, axis=0)
    prob_zero = np.concatenate(all_zero, axis=0)
    prob_age_zero = np.concatenate(all_age_zero, axis=0)
    prob_sex_zero = np.concatenate(all_sex_zero, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    delta_all = prob_real - prob_zero
    delta_age = prob_real - prob_age_zero
    delta_sex = prob_real - prob_sex_zero

    rows = []
    summary: Dict[str, Any] = {
        "enabled": True,
        "n_samples": int(len(prob_real)),
        "interpretation": "delta = probabilidade com metadados reais - probabilidade com metadados ablacionados/zerados",
        "per_class": {},
    }

    for i, cls in enumerate(MAIN_SUPERCLASS_LIST):
        for kind, delta in [
            ("all_meta_zeroed", delta_all),
            ("age_zeroed", delta_age),
            ("sex_zeroed", delta_sex),
        ]:
            d = delta[:, i]
            row = {
                "class": cls,
                "ablation": kind,
                "mean_delta": float(np.mean(d)),
                "mean_abs_delta": float(np.mean(np.abs(d))),
                "median_abs_delta": float(np.median(np.abs(d))),
                "p95_abs_delta": float(np.quantile(np.abs(d), 0.95)),
                "max_abs_delta": float(np.max(np.abs(d))),
            }
            rows.append(row)

            summary["per_class"].setdefault(cls, {})[kind] = row

    details_rows = []
    for r, exam_id in enumerate(all_exam_ids):
        row = {"exam_id": exam_id}
        for i, cls in enumerate(MAIN_SUPERCLASS_LIST):
            row[f"true_{cls}"] = int(y_true[r, i])
            row[f"prob_real_{cls}"] = float(prob_real[r, i])
            row[f"prob_meta_zero_{cls}"] = float(prob_zero[r, i])
            row[f"delta_meta_{cls}"] = float(delta_all[r, i])
            row[f"delta_age_{cls}"] = float(delta_age[r, i])
            row[f"delta_sex_{cls}"] = float(delta_sex[r, i])
        details_rows.append(row)

    summary_df = pd.DataFrame(rows)
    details_df = pd.DataFrame(details_rows)

    summary_df.to_csv(results_dir / "metadata_influence_summary.csv", index=False, encoding="utf-8-sig")
    details_df.to_csv(results_dir / "metadata_influence_per_sample.csv", index=False, encoding="utf-8-sig")
    (results_dir / "metadata_influence.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_metadata_influence(summary_df, results_dir / "plots" / "metadata_influence.png")
    return summary


def plot_metadata_influence(summary_df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    if summary_df.empty:
        return

    fig = plt.figure(figsize=(10, 6))

    # Barras agrupadas simples por ablação.
    classes = list(MAIN_SUPERCLASS_LIST)
    ablations = ["all_meta_zeroed", "age_zeroed", "sex_zeroed"]
    x = np.arange(len(classes))
    width = 0.25

    for j, abl in enumerate(ablations):
        vals = []
        for cls in classes:
            sub = summary_df[(summary_df["class"] == cls) & (summary_df["ablation"] == abl)]
            vals.append(float(sub["mean_abs_delta"].iloc[0]) if not sub.empty else 0.0)
        plt.bar(x + (j - 1) * width, vals, width=width, label=abl)

    plt.xticks(x, classes)
    plt.ylabel("Mean absolute probability delta")
    plt.title("Influência dos metadados por ablação")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run_advanced_evaluation_outputs(
    model: SimpleECGCNNMultiTask,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    results_dir: Path,
    device: torch.device,
) -> None:
    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)

    if GENERATE_ADVANCED_EVAL:
        build_metrics_json(
            y_true=y_true,
            y_prob=y_prob,
            class_names=MAIN_SUPERCLASS_LIST,
            threshold=THRESHOLD_MAIN,
            output_path=results_dir / "metrics.json",
            extra={
                "main_task": True,
                "auxiliary_head_used_for_training": bool(USE_AUX_HEAD),
                "metadata_features_used": bool(USE_META_FEATURES),
                "data_loading_mode": DATA_LOADING_MODE,
            },
        )

    if GENERATE_ROC_CURVES:
        plot_roc_curves(
            y_true=y_true,
            y_prob=y_prob,
            class_names=MAIN_SUPERCLASS_LIST,
            output_path=plots_dir / "roc_curves.png",
        )

    if GENERATE_PR_CURVES:
        plot_pr_curves(
            y_true=y_true,
            y_prob=y_prob,
            class_names=MAIN_SUPERCLASS_LIST,
            output_path=plots_dir / "pr_curves.png",
        )

    if GENERATE_CALIBRATION_PLOT:
        plot_calibration(
            y_true=y_true,
            y_prob=y_prob,
            class_names=MAIN_SUPERCLASS_LIST,
            output_path=plots_dir / "calibration.png",
        )

    if GENERATE_MULTILABEL_CONFUSION:
        plot_multilabel_confusion_matrices(
            y_true=y_true,
            y_prob=y_prob,
            class_names=MAIN_SUPERCLASS_LIST,
            threshold=THRESHOLD_MAIN,
            output_path=plots_dir / "multilabel_confusion_by_superclass.png",
        )

    if GENERATE_GRADCAM:
        generate_gradcam_examples(
            model=model,
            test_df=test_df,
            y_true=y_true,
            y_prob=y_prob,
            class_names=MAIN_SUPERCLASS_LIST,
            results_dir=results_dir,
            device=device,
        )

    if GENERATE_METADATA_INFLUENCE:
        evaluate_metadata_influence(
            model=model,
            test_df=test_df,
            results_dir=results_dir,
            device=device,
        )

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    set_seed(SEED)

    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    results_dir = Path(RESULTS_DIR).expanduser().resolve()
    ensure_dir(results_dir)
    ensure_dir(results_dir / "plots")

    save_config(results_dir)

    device = get_device()

    df = load_and_prepare_metadata(output_dir, results_dir)

    if USE_AUGMENTATION:
        print(f"[MELHORIA] Augmentação de ECG habilitada (p={AUG_PROBABILITY}, lead_dropout={AUG_LEAD_DROPOUT_PROB})")
    if USE_RESIDUAL_BLOCKS:
        print(f"[MELHORIA] Blocos residuais habilitados (N_RESIDUAL_PER_STAGE={N_RESIDUAL_PER_STAGE})")
    if USE_ATTENTION_POOL:
        print("[MELHORIA] Pooling com atenção temporal habilitado")

    model, history = train_model(df, results_dir, device)
    plot_history(results_dir, history)

    evaluate_test(model, df, results_dir, device)

    print("\nConcluído.")
    print(f"Resultados em: {results_dir}")
    print(f"Melhor modelo: {results_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
