# -*- coding: utf-8 -*-
"""
Processamento robusto de ECGs WFDB (.hea + .mat) de múltiplas bases.

Saídas:
  - batches/batch_0000.npz, batch_0001.npz, ...
      x: shape (N, 12, TARGET_SAMPLES), dtype float32
      global_indices: índice global do exame no metadata.csv
      exam_ids: ids dos exames no batch
  - metadata.csv
  - quality_report.csv
  - discarded.csv
  - processing.log

Dependências:
  pip install wfdb numpy pandas scipy tqdm
"""

from __future__ import annotations

import gc
import logging
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from tqdm import tqdm


# =============================================================================
# Variáveis globais
# =============================================================================

# --- Paths ---
BASE_DIRS = [
    r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Brutos\Chapman-Shaoxing-Ningbo",
    r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Brutos\cpsc_2018",
    r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Brutos\cpsc_2018_extra",
    r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Brutos\georgia",
    r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Brutos\ptb",
    r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Brutos\ptb-xl",
]
MAP_CSV_PATH = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Maps.csv"
OUTPUT_DIR   = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Processados"

# --- Leads ---
TARGET_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# --- Sinal ---
TARGET_FS      = 500
TARGET_SECONDS = 10
TARGET_SAMPLES = TARGET_FS * TARGET_SECONDS

ALLOW_RESAMPLE  = True
ALLOW_TRIM_PAD  = True

# --- Filtros de sinal ---
BANDPASS_LOW_HZ  = 0.5
BANDPASS_HIGH_HZ = 40.0
FILTER_ORDER     = 4

# --- Amplitude / Qualidade ---
AMPLITUDE_PEAK_MV      = 20.0
FLATLINE_STD_THRESHOLD = 0.01
NAN_ALLOWED            = False

# --- Idade ---
AGE_MIN                     = 0
AGE_MAX                     = 120
AGE_OUT_OF_RANGE_FILL_VALUE = None
DEFAULT_AGE_IF_NO_VALID_AGES = 60

# --- Sexo ---
SEX_FILL_VALUE = "Unknown"
VALID_SEX_MAP  = {
    "male": "Male", "m": "Male", "1": "Male",
    "female": "Female", "f": "Female", "0": "Female",
}

# --- Batches ---
BATCH_SIZE = 1024

# --- Reexecução ---
# Se True, apaga saídas antigas e reprocessa tudo.
# Se False, evita reprocessar exames já presentes em metadata.csv.
FORCE_REPROCESS = False

# --- Precisão ---
OUTPUT_DTYPE = np.float32


# =============================================================================
# Logging
# =============================================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "processing.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# =============================================================================
# Estruturas
# =============================================================================

@dataclass
class HeaderInfo:
    record_path_no_ext: Path
    hea_path: Path
    exam_id: str
    source_base: str
    fs: Optional[float]
    sig_len: Optional[int]
    n_sig: Optional[int]
    sig_names: List[str]
    age_raw: Optional[str]
    sex_raw: Optional[str]
    dx_raw: Optional[str]


@dataclass
class ProcessedExam:
    signal_12xT: np.ndarray
    metadata: Dict[str, Any]
    quality_detail: str


# =============================================================================
# Utilitários de filesystem e header
# =============================================================================

def normalize_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def source_base_from_path(path: Path, base_dirs: Sequence[Path]) -> str:
    path_resolved = path.resolve()
    best_match: Optional[Path] = None

    for base in base_dirs:
        try:
            path_resolved.relative_to(base)
            if best_match is None or len(str(base)) > len(str(best_match)):
                best_match = base
        except ValueError:
            continue

    if best_match is not None:
        return best_match.name

    return "unknown"


def find_hea_files(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        logging.warning("Base não encontrada: %s", base_dir)
        return []
    return sorted(base_dir.rglob("*.hea"))


def read_header_text(hea_path: Path) -> List[str]:
    # Headers PhysioNet geralmente são ASCII/UTF-8, mas alguns arquivos podem ter variações.
    for enc in ("utf-8", "latin-1"):
        try:
            return hea_path.read_text(encoding=enc).splitlines()
        except UnicodeDecodeError:
            continue
    return hea_path.read_text(errors="ignore").splitlines()


def parse_header(hea_path: Path, base_dirs: Sequence[Path]) -> HeaderInfo:
    lines = read_header_text(hea_path)
    exam_id = hea_path.stem
    record_path_no_ext = hea_path.with_suffix("")
    source_base = source_base_from_path(hea_path, base_dirs)

    fs = None
    sig_len = None
    n_sig = None
    sig_names: List[str] = []
    age_raw = None
    sex_raw = None
    dx_raw = None

    if lines:
        first = lines[0].strip().split()
        # Exemplo: HR00001 12 500 5000
        if len(first) >= 4:
            try:
                n_sig = int(float(first[1]))
            except Exception:
                n_sig = None
            try:
                fs = float(first[2])
            except Exception:
                fs = None
            try:
                sig_len = int(float(first[3]))
            except Exception:
                sig_len = None

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if s.startswith("#"):
            m = re.match(r"^#\s*([^:]+)\s*:\s*(.*)$", s)
            if not m:
                continue
            key = m.group(1).strip().lower()
            value = m.group(2).strip()
            if key == "age":
                age_raw = value
            elif key == "sex":
                sex_raw = value
            elif key == "dx":
                dx_raw = value
            continue

        # Linhas de sinal: último token é o nome do lead.
        parts = s.split()
        if len(parts) >= 9 and not s.startswith("#"):
            possible_lead = parts[-1].strip()
            if possible_lead:
                sig_names.append(possible_lead)

    return HeaderInfo(
        record_path_no_ext=record_path_no_ext,
        hea_path=hea_path,
        exam_id=exam_id,
        source_base=source_base,
        fs=fs,
        sig_len=sig_len,
        n_sig=n_sig,
        sig_names=sig_names,
        age_raw=age_raw,
        sex_raw=sex_raw,
        dx_raw=dx_raw,
    )


def collect_headers(base_dirs: Sequence[Path]) -> List[HeaderInfo]:
    headers: List[HeaderInfo] = []

    for base_dir in base_dirs:
        hea_files = find_hea_files(base_dir)
        logging.info("Base %s: %d headers encontrados.", base_dir.name, len(hea_files))
        for hea_path in tqdm(hea_files, desc=f"Scan headers: {base_dir.name}"):
            try:
                headers.append(parse_header(hea_path, base_dirs))
            except Exception as exc:
                logging.warning("Falha lendo header %s: %s", hea_path, exc)

    return headers


# =============================================================================
# Idade, sexo e diagnóstico
# =============================================================================

def parse_age(age_raw: Optional[str]) -> Optional[float]:
    if age_raw is None:
        return None

    s = str(age_raw).strip()
    if not s or s.lower() in {"nan", "none", "unknown", "na", "n/a"}:
        return None

    try:
        value = float(s)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    if value < AGE_MIN or value > AGE_MAX:
        return None

    return value


def compute_age_mean_by_base(headers: Sequence[HeaderInfo]) -> Dict[str, float]:
    ages_by_base: Dict[str, List[float]] = {}

    for h in headers:
        age = parse_age(h.age_raw)
        if age is not None:
            ages_by_base.setdefault(h.source_base, []).append(age)

    means: Dict[str, float] = {}
    for base, ages in ages_by_base.items():
        if ages:
            means[base] = float(np.mean(ages))

    all_bases = sorted({h.source_base for h in headers})
    for base in all_bases:
        means.setdefault(base, float(DEFAULT_AGE_IF_NO_VALID_AGES))

    return means


def normalize_age(age_raw: Optional[str], source_base: str, age_mean_by_base: Dict[str, float]) -> Tuple[float, bool]:
    age = parse_age(age_raw)
    if age is not None:
        return float(age), False

    if AGE_OUT_OF_RANGE_FILL_VALUE is not None:
        return float(AGE_OUT_OF_RANGE_FILL_VALUE), True

    return float(age_mean_by_base.get(source_base, DEFAULT_AGE_IF_NO_VALID_AGES)), True


def normalize_sex(sex_raw: Optional[str]) -> Tuple[str, bool]:
    if sex_raw is None:
        return SEX_FILL_VALUE, True

    key = str(sex_raw).strip().lower()
    if key in VALID_SEX_MAP:
        return VALID_SEX_MAP[key], False

    return SEX_FILL_VALUE, True


def split_dx_codes(dx_raw: Optional[str]) -> List[str]:
    if dx_raw is None:
        return []
    s = str(dx_raw).strip()
    if not s or s.lower() in {"unknown", "none", "nan", "na", "n/a"}:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def load_map_csv(map_csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not map_csv_path.exists():
        raise FileNotFoundError(f"MAP_CSV_PATH não encontrado: {map_csv_path}")

    # Tenta separador automático; fallback para ; e ,
    read_attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ";"},
        {"sep": ","},
    ]

    last_exc: Optional[Exception] = None
    df: Optional[pd.DataFrame] = None
    for kwargs in read_attempts:
        try:
            df = pd.read_csv(map_csv_path, dtype=str, **kwargs)
            break
        except Exception as exc:
            last_exc = exc

    if df is None:
        raise RuntimeError(f"Falha ao ler Map.csv: {last_exc}")

    df.columns = [str(c).strip() for c in df.columns]

    required = {"Snomed_CT", "Acronym", "Superclass_ID", "Superclass"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Map.csv sem colunas obrigatórias: {sorted(missing)}. Colunas encontradas: {list(df.columns)}")

    df = df.fillna("")
    mapping: Dict[str, Dict[str, str]] = {}

    for _, row in df.iterrows():
        code = str(row["Snomed_CT"]).strip()
        if not code:
            continue
        # Remove .0 caso o CSV tenha sido salvo com código como float.
        if code.endswith(".0"):
            code = code[:-2]
        mapping[code] = {
            "Acronym": str(row["Acronym"]).strip(),
            "Superclass_ID": str(row["Superclass_ID"]).strip(),
            "Superclass": str(row["Superclass"]).strip(),
        }

    return mapping


def map_diagnoses(dx_codes: Sequence[str], dx_map: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    mapped_codes: List[str] = []
    acronyms: List[str] = []
    superclass_ids: List[str] = []
    superclasses: List[str] = []

    for raw_code in dx_codes:
        code = str(raw_code).strip()
        if code.endswith(".0"):
            code = code[:-2]
        item = dx_map.get(code)
        if item is None:
            continue
        mapped_codes.append(code)
        acronyms.append(item.get("Acronym", ""))
        superclass_ids.append(item.get("Superclass_ID", ""))
        superclasses.append(item.get("Superclass", ""))

    return mapped_codes, acronyms, superclass_ids, superclasses


# =============================================================================
# Processamento de sinal
# =============================================================================

def read_wfdb_signal(record_path_no_ext: Path) -> Tuple[np.ndarray, List[str], float]:
    """
    Retorna sinal como shape (n_samples, n_leads), nomes dos leads e fs.
    """
    record_name = str(record_path_no_ext)
    record = wfdb.rdrecord(record_name, physical=True)

    if record.p_signal is not None:
        sig = np.asarray(record.p_signal, dtype=np.float64)
    elif record.d_signal is not None:
        # Fallback: se p_signal não estiver disponível.
        sig = np.asarray(record.d_signal, dtype=np.float64)
    else:
        raise ValueError("wfdb.rdrecord não retornou p_signal nem d_signal.")

    sig_names = list(record.sig_name) if record.sig_name is not None else []
    fs = float(record.fs)

    if sig.ndim != 2:
        raise ValueError(f"Sinal com ndim inválido: {sig.ndim}")

    return sig, sig_names, fs


def select_and_order_leads(sig_nxL: np.ndarray, sig_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Retorna sinal shape (12, n_samples), na ordem TARGET_LEADS.
    Se algum lead estiver ausente, levanta ValueError. O descarte evita criar sinal artificial
    para bases que deveriam conter 12 derivações completas.
    """
    lead_to_idx = {str(name).strip(): i for i, name in enumerate(sig_names)}
    missing = [lead for lead in TARGET_LEADS if lead not in lead_to_idx]
    if missing:
        raise ValueError("missing_leads:" + ",".join(missing))

    ordered = np.stack([sig_nxL[:, lead_to_idx[lead]] for lead in TARGET_LEADS], axis=0)
    return ordered, []


def resample_signal(sig_12xT: np.ndarray, original_fs: float) -> np.ndarray:
    if int(round(original_fs)) == int(TARGET_FS):
        return sig_12xT

    if not ALLOW_RESAMPLE:
        raise ValueError(f"resample_not_allowed:fs_{original_fs:g}")

    if original_fs <= 0:
        raise ValueError(f"invalid_fs:{original_fs}")

    frac = Fraction(int(TARGET_FS), int(round(original_fs))).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    return signal.resample_poly(sig_12xT, up=up, down=down, axis=1)


def trim_or_pad(sig_12xT: np.ndarray) -> np.ndarray:
    n = sig_12xT.shape[1]

    if n == TARGET_SAMPLES:
        return sig_12xT

    if not ALLOW_TRIM_PAD:
        raise ValueError(f"length_not_allowed:n_{n}")

    if n > TARGET_SAMPLES:
        return sig_12xT[:, :TARGET_SAMPLES]

    pad_width = TARGET_SAMPLES - n
    return np.pad(sig_12xT, pad_width=((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def bandpass_filter(sig_12xT: np.ndarray) -> np.ndarray:
    nyq = TARGET_FS / 2.0

    low = BANDPASS_LOW_HZ / nyq
    high = BANDPASS_HIGH_HZ / nyq

    if not (0 < low < high < 1):
        raise ValueError(f"Filtro inválido: low={BANDPASS_LOW_HZ}, high={BANDPASS_HIGH_HZ}, fs={TARGET_FS}")

    sos = signal.butter(FILTER_ORDER, [low, high], btype="bandpass", output="sos")

    # sosfiltfilt é numericamente mais estável que filtfilt com coeficientes b/a.
    try:
        return signal.sosfiltfilt(sos, sig_12xT, axis=1)
    except ValueError:
        # Fallback para sinais muito curtos antes do pad; normalmente não ocorre após trim_or_pad.
        return signal.sosfilt(sos, sig_12xT, axis=1)


def validate_signal_quality(sig_12xT: np.ndarray) -> Tuple[List[str], str]:
    warnings: List[str] = []
    details: List[str] = []

    for i, lead in enumerate(TARGET_LEADS):
        x = sig_12xT[i]

        if np.any(~np.isfinite(x)):
            warnings.append("nan_inf")
            details.append(f"{lead}:nan_inf")

        finite_x = x[np.isfinite(x)]
        if finite_x.size == 0:
            continue

        max_abs = float(np.max(np.abs(finite_x)))
        std = float(np.std(finite_x))

        if max_abs > AMPLITUDE_PEAK_MV:
            warnings.append("spike_amplitude")
            details.append(f"{lead}:spike_amplitude:max_abs={max_abs:.4g}")

        if std < FLATLINE_STD_THRESHOLD:
            warnings.append("flatline")
            details.append(f"{lead}:flatline:std={std:.4g}")

    # Remove duplicatas preservando ordem.
    warnings_unique = list(dict.fromkeys(warnings))
    return warnings_unique, "|".join(details)


def sanitize_final_signal(sig_12xT: np.ndarray) -> np.ndarray:
    if NAN_ALLOWED:
        return sig_12xT.astype(OUTPUT_DTYPE, copy=False)

    # Substitui NaN/Inf após registrar warning, para não quebrar treino downstream.
    return np.nan_to_num(sig_12xT, nan=0.0, posinf=0.0, neginf=0.0).astype(OUTPUT_DTYPE, copy=False)


def process_one_exam(
    h: HeaderInfo,
    dx_map: Dict[str, Dict[str, str]],
    age_mean_by_base: Dict[str, float],
) -> ProcessedExam:
    warnings: List[str] = []

    sig_nxL, sig_names, fs = read_wfdb_signal(h.record_path_no_ext)

    try:
        sig_12xT, lead_warnings = select_and_order_leads(sig_nxL, sig_names)
        warnings.extend(lead_warnings)
    except ValueError as exc:
        # Leads ausentes são críticos para manter shape 12xT sem imputação artificial.
        msg = str(exc)
        if msg.startswith("missing_leads:"):
            raise ValueError(msg)
        raise

    sig_12xT = resample_signal(sig_12xT, fs)
    sig_12xT = trim_or_pad(sig_12xT)
    sig_12xT = bandpass_filter(sig_12xT)

    quality_warnings, quality_detail = validate_signal_quality(sig_12xT)
    warnings.extend(quality_warnings)

    age, age_filled = normalize_age(h.age_raw, h.source_base, age_mean_by_base)
    if age_filled:
        warnings.append("age_filled")

    sex, sex_filled = normalize_sex(h.sex_raw)
    if sex_filled:
        warnings.append("sex_unknown")

    dx_codes = split_dx_codes(h.dx_raw)
    mapped_codes, acronyms, superclass_ids, superclasses = map_diagnoses(dx_codes, dx_map)

    mapped_set = set(mapped_codes)
    for code in dx_codes:
        clean_code = code[:-2] if code.endswith(".0") else code
        if clean_code not in mapped_set:
            warnings.append(f"dx_not_mapped:{clean_code}")

    warnings = list(dict.fromkeys(warnings))
    warning_text = "|".join(warnings)

    sig_12xT = sanitize_final_signal(sig_12xT)

    metadata = {
        "exam_id": h.exam_id,
        "source_base": h.source_base,
        "batch_file": "",
        "batch_index": "",
        "age": age,
        "sex": sex,
        "dx_codes": "|".join(dx_codes),
        "acronym": "|".join(acronyms),
        "superclass_id": "|".join(superclass_ids),
        "superclass": "|".join(superclasses),
        "warning_exame": warning_text,
    }

    return ProcessedExam(signal_12xT=sig_12xT, metadata=metadata, quality_detail=quality_detail)


# =============================================================================
# Escrita incremental
# =============================================================================

METADATA_COLUMNS = [
    "exam_id",
    "source_base",
    "batch_file",
    "batch_index",
    "age",
    "sex",
    "dx_codes",
    "acronym",
    "superclass_id",
    "superclass",
    "warning_exame",
]

QUALITY_COLUMNS = METADATA_COLUMNS + ["quality_detail", "hea_path"]
DISCARDED_COLUMNS = ["exam_id", "source_base", "motivo", "hea_path"]


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str], append: bool = True) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    df = df[list(columns)]

    header = not (append and path.exists())
    mode = "a" if append else "w"
    df.to_csv(path, index=False, mode=mode, header=header, encoding="utf-8-sig")


def load_processed_exam_ids(metadata_path: Path) -> set[str]:
    if FORCE_REPROCESS or not metadata_path.exists():
        return set()

    try:
        df = pd.read_csv(metadata_path, usecols=["exam_id"], dtype=str)
        return set(df["exam_id"].dropna().astype(str).tolist())
    except Exception:
        logging.warning("Não foi possível ler metadata.csv existente. Reprocessamento pode duplicar linhas.")
        return set()


def next_batch_number(batches_dir: Path) -> int:
    existing = sorted(batches_dir.glob("batch_*.npz"))
    if not existing:
        return 0

    max_idx = -1
    for p in existing:
        m = re.match(r"batch_(\d+)\.npz$", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def save_batch(
    batch_signals: List[np.ndarray],
    batch_metadata: List[Dict[str, Any]],
    batch_number: int,
    global_start_index: int,
    batches_dir: Path,
    metadata_path: Path,
    quality_path: Path,
) -> int:
    batch_file = f"batch_{batch_number:04d}.npz"
    batch_path = batches_dir / batch_file

    x = np.stack(batch_signals, axis=0).astype(OUTPUT_DTYPE, copy=False)
    n = x.shape[0]

    global_indices = np.arange(global_start_index, global_start_index + n, dtype=np.int64)
    exam_ids = np.array([m["exam_id"] for m in batch_metadata], dtype=object)

    for i, meta in enumerate(batch_metadata):
        meta["batch_file"] = batch_file
        meta["batch_index"] = i

    np.savez_compressed(
        batch_path,
        x=x,
        global_indices=global_indices,
        exam_ids=exam_ids,
    )

    write_csv(metadata_path, batch_metadata, METADATA_COLUMNS, append=True)

    quality_rows = []
    for meta in batch_metadata:
        if str(meta.get("warning_exame", "")).strip():
            row = dict(meta)
            row["quality_detail"] = meta.get("_quality_detail", "")
            row["hea_path"] = meta.get("_hea_path", "")
            quality_rows.append(row)

    # Remove colunas internas do metadata já escrito.
    for meta in batch_metadata:
        meta.pop("_quality_detail", None)
        meta.pop("_hea_path", None)

    write_csv(quality_path, quality_rows, QUALITY_COLUMNS, append=True)

    logging.info("Batch salvo: %s | shape=%s", batch_path, x.shape)

    del x
    gc.collect()

    return global_start_index + n


def reset_outputs(output_dir: Path) -> None:
    if not FORCE_REPROCESS:
        return

    for filename in ["metadata.csv", "quality_report.csv", "discarded.csv", "processing.log"]:
        p = output_dir / filename
        if p.exists():
            p.unlink()

    batches_dir = output_dir / "batches"
    if batches_dir.exists():
        for p in batches_dir.glob("batch_*.npz"):
            p.unlink()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    output_dir = normalize_path(OUTPUT_DIR)
    reset_outputs(output_dir)
    setup_logging(output_dir)

    base_dirs = [normalize_path(p) for p in BASE_DIRS]
    map_csv_path = normalize_path(MAP_CSV_PATH)

    batches_dir = output_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    quality_path = output_dir / "quality_report.csv"
    discarded_path = output_dir / "discarded.csv"

    logging.info("Iniciando processamento.")
    logging.info("OUTPUT_DIR: %s", output_dir)

    logging.info("Carregando mapa diagnóstico: %s", map_csv_path)
    dx_map = load_map_csv(map_csv_path)
    logging.info("Mapa carregado: %d códigos SNOMED.", len(dx_map))

    logging.info("Primeira passagem: varredura de headers e cálculo de idade média por base.")
    headers = collect_headers(base_dirs)
    logging.info("Total de headers encontrados: %d", len(headers))

    age_mean_by_base = compute_age_mean_by_base(headers)
    logging.info("Média de idade por base: %s", age_mean_by_base)

    processed_ids = load_processed_exam_ids(metadata_path)
    if processed_ids:
        logging.info("Reexecução: %d exames já presentes em metadata.csv serão pulados.", len(processed_ids))

    if metadata_path.exists() and not FORCE_REPROCESS:
        try:
            global_index = len(pd.read_csv(metadata_path, usecols=["exam_id"]))
        except Exception:
            global_index = 0
    else:
        global_index = 0

    batch_number = next_batch_number(batches_dir)

    batch_signals: List[np.ndarray] = []
    batch_metadata: List[Dict[str, Any]] = []
    discarded_rows: List[Dict[str, Any]] = []

    # Processa agrupado por base para tqdm mais legível.
    headers_by_base: Dict[str, List[HeaderInfo]] = {}
    for h in headers:
        headers_by_base.setdefault(h.source_base, []).append(h)

    for source_base, base_headers in headers_by_base.items():
        logging.info("Processando base %s: %d exames.", source_base, len(base_headers))

        for h in tqdm(base_headers, desc=f"Process: {source_base}"):
            if h.exam_id in processed_ids:
                continue

            try:
                processed = process_one_exam(h, dx_map, age_mean_by_base)

                processed.metadata["_quality_detail"] = processed.quality_detail
                processed.metadata["_hea_path"] = str(h.hea_path)

                batch_signals.append(processed.signal_12xT)
                batch_metadata.append(processed.metadata)

                if len(batch_signals) >= BATCH_SIZE:
                    global_index = save_batch(
                        batch_signals=batch_signals,
                        batch_metadata=batch_metadata,
                        batch_number=batch_number,
                        global_start_index=global_index,
                        batches_dir=batches_dir,
                        metadata_path=metadata_path,
                        quality_path=quality_path,
                    )
                    batch_number += 1
                    batch_signals.clear()
                    batch_metadata.clear()
                    gc.collect()

            except Exception as exc:
                motivo = str(exc)
                logging.warning("Descartado %s | base=%s | motivo=%s", h.exam_id, h.source_base, motivo)
                discarded_rows.append({
                    "exam_id": h.exam_id,
                    "source_base": h.source_base,
                    "motivo": motivo,
                    "hea_path": str(h.hea_path),
                })

                if len(discarded_rows) >= 1000:
                    write_csv(discarded_path, discarded_rows, DISCARDED_COLUMNS, append=True)
                    discarded_rows.clear()

    if batch_signals:
        global_index = save_batch(
            batch_signals=batch_signals,
            batch_metadata=batch_metadata,
            batch_number=batch_number,
            global_start_index=global_index,
            batches_dir=batches_dir,
            metadata_path=metadata_path,
            quality_path=quality_path,
        )
        batch_signals.clear()
        batch_metadata.clear()

    if discarded_rows:
        write_csv(discarded_path, discarded_rows, DISCARDED_COLUMNS, append=True)
        discarded_rows.clear()

    # Garante existência dos CSVs, mesmo vazios.
    if not metadata_path.exists():
        pd.DataFrame(columns=METADATA_COLUMNS).to_csv(metadata_path, index=False, encoding="utf-8-sig")
    if not quality_path.exists():
        pd.DataFrame(columns=QUALITY_COLUMNS).to_csv(quality_path, index=False, encoding="utf-8-sig")
    if not discarded_path.exists():
        pd.DataFrame(columns=DISCARDED_COLUMNS).to_csv(discarded_path, index=False, encoding="utf-8-sig")

    logging.info("Processamento concluído.")
    logging.info("Total global processado em metadata.csv: %d", global_index)
    logging.info("Batches em: %s", batches_dir)
    logging.info("Metadata: %s", metadata_path)
    logging.info("Quality report: %s", quality_path)
    logging.info("Discarded: %s", discarded_path)


if __name__ == "__main__":
    main()
