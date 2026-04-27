# -*- coding: utf-8 -*-
"""
Teste / auditoria do pré-processamento de ECG já gerado por process_ecg_wfdb_batches.py.

Este script valida explicitamente se os batches estão em:
    (N, 12, 5000)  -> formato salvo pelo preprocessamento: exames, leads, tempo
equivalente a:
    500 Hz x 10 s x 12 leads

Também gera relatórios de metadados por base, incluindo:
    - superclasses por base
    - acrônimos por base
    - SNOMED/Dx por base
    - warnings por base
    - sexo por base
    - idade por base

Saídas em:
  OUTPUT_DIR\\preprocessing_test\\
    - preprocessing_test_summary.txt
    - batch_integrity_report.csv
    - sample_signal_stats.csv
    - metadata_distribution_report.csv
    - warning_distribution.csv
    - superclass_by_base.csv
    - acronym_by_base.csv
    - dx_by_base.csv
    - warning_by_base.csv
    - sex_by_base.csv
    - age_by_base.csv
    - plots\\*.png

Dependências:
  pip install numpy pandas matplotlib tqdm
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Configurações globais
# =============================================================================

# Diretório gerado pelo script de processamento.
OUTPUT_DIR = r"C:\Users\bruno\OneDrive\Desktop\Classification_ECG\Dados_Processados"

# Mesmos parâmetros esperados do preprocessamento.
TARGET_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_FS = 500
TARGET_SECONDS = 10
TARGET_SAMPLES = TARGET_FS * TARGET_SECONDS

# Formato esperado salvo pelo preprocessamento:
# x.shape == (N, 12, 5000)
EXPECTED_N_LEADS = len(TARGET_LEADS)
EXPECTED_SIGNAL_SHAPE_PER_EXAM = (EXPECTED_N_LEADS, TARGET_SAMPLES)

AMPLITUDE_PEAK_MV = 20.0
FLATLINE_STD_THRESHOLD = 0.01

# Amostragem para estatísticas finas.
# Use None para varrer todos os exames em todos os batches.
MAX_EXAMS_FOR_DETAILED_STATS = 5000

# Quantos exemplos plotar.
N_RANDOM_PLOTS = 12
N_WARNING_PLOTS = 12
N_PER_SUPERCLASS_PLOTS = 3

# Plot.
PLOT_SECONDS = TARGET_SECONDS
PLOT_DPI = 130
RANDOM_SEED = 42

# Se True, falhas críticas encerram o script com erro.
# Se False, registra falhas no relatório e continua quando possível.
STRICT = False


# =============================================================================
# Utilitários
# =============================================================================

def pjoin(*parts: str | Path) -> Path:
    return Path(*parts).expanduser().resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_pipe(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def explode_pipe_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Explode coluna com itens separados por '|', preservando source_base e exam_id.
    """
    if column not in df.columns:
        return pd.DataFrame(columns=["exam_id", "source_base", column])

    rows = []
    for _, row in df.iterrows():
        items = split_pipe(row.get(column, ""))
        if not items:
            rows.append({
                "exam_id": row.get("exam_id", ""),
                "source_base": row.get("source_base", ""),
                column: "<empty>",
            })
        else:
            for item in items:
                rows.append({
                    "exam_id": row.get("exam_id", ""),
                    "source_base": row.get("source_base", ""),
                    column: item,
                })

    return pd.DataFrame(rows)


def fail_or_warn(errors: List[str], message: str) -> None:
    errors.append(message)
    if STRICT:
        raise RuntimeError(message)


def safe_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def parse_batch_number(path: Path) -> int:
    m = re.match(r"batch_(\d+)\.npz$", path.name)
    if not m:
        return -1
    return int(m.group(1))


def load_csv_if_exists(path: Path, required: bool, errors: List[str]) -> pd.DataFrame:
    if not path.exists():
        msg = f"Arquivo não encontrado: {path}"
        if required:
            fail_or_warn(errors, msg)
        else:
            errors.append(msg)
        return pd.DataFrame()

    try:
        return pd.read_csv(path, dtype=str)
    except Exception as exc:
        fail_or_warn(errors, f"Falha lendo CSV {path}: {exc}")
        return pd.DataFrame()


# =============================================================================
# Validações de estrutura e metadata
# =============================================================================

def validate_structure(output_dir: Path, errors: List[str]) -> Dict[str, Path]:
    batches_dir = output_dir / "batches"
    metadata_path = output_dir / "metadata.csv"
    quality_path = output_dir / "quality_report.csv"
    discarded_path = output_dir / "discarded.csv"

    if not output_dir.exists():
        fail_or_warn(errors, f"OUTPUT_DIR não existe: {output_dir}")

    if not batches_dir.exists():
        fail_or_warn(errors, f"Pasta de batches não existe: {batches_dir}")

    if not metadata_path.exists():
        fail_or_warn(errors, f"metadata.csv não existe: {metadata_path}")

    return {
        "batches_dir": batches_dir,
        "metadata_path": metadata_path,
        "quality_path": quality_path,
        "discarded_path": discarded_path,
    }


def validate_metadata_columns(metadata: pd.DataFrame, errors: List[str]) -> None:
    required_columns = [
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

    missing = [c for c in required_columns if c not in metadata.columns]
    if missing:
        fail_or_warn(errors, f"metadata.csv sem colunas obrigatórias: {missing}")

    if metadata.empty:
        fail_or_warn(errors, "metadata.csv está vazio.")

    if "exam_id" in metadata.columns:
        duplicated = metadata["exam_id"].duplicated().sum()
        if duplicated > 0:
            errors.append(f"metadata.csv tem {duplicated} exam_id duplicados.")


def metadata_distribution_reports(metadata: pd.DataFrame, output_test_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add_counts(column: str) -> None:
        if column not in metadata.columns:
            return
        counts = metadata[column].fillna("").replace("", "<empty>").value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({
                "category": column,
                "value": value,
                "count": int(count),
                "fraction": float(count) / max(1, len(metadata)),
            })

    for col in ["source_base", "sex"]:
        add_counts(col)

    if "age" in metadata.columns:
        age = pd.to_numeric(metadata["age"], errors="coerce")
        rows.extend([
            {"category": "age_summary", "value": "valid_count", "count": int(age.notna().sum()), "fraction": float(age.notna().mean())},
            {"category": "age_summary", "value": "missing_count", "count": int(age.isna().sum()), "fraction": float(age.isna().mean())},
            {"category": "age_summary", "value": "mean", "count": float(age.mean()) if age.notna().any() else np.nan, "fraction": np.nan},
            {"category": "age_summary", "value": "std", "count": float(age.std()) if age.notna().any() else np.nan, "fraction": np.nan},
            {"category": "age_summary", "value": "min", "count": float(age.min()) if age.notna().any() else np.nan, "fraction": np.nan},
            {"category": "age_summary", "value": "max", "count": float(age.max()) if age.notna().any() else np.nan, "fraction": np.nan},
        ])

    for column in ["acronym", "superclass", "superclass_id", "dx_codes"]:
        if column not in metadata.columns:
            continue
        counter: Counter[str] = Counter()
        for v in metadata[column].fillna(""):
            for item in split_pipe(v):
                counter[item] += 1
        for value, count in counter.most_common():
            rows.append({
                "category": f"{column}_exploded",
                "value": value,
                "count": int(count),
                "fraction": float(count) / max(1, len(metadata)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_test_dir / "metadata_distribution_report.csv", index=False, encoding="utf-8-sig")
    return df


def create_metadata_by_base_reports(metadata: pd.DataFrame, output_test_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Gera os relatórios que faltavam: principalmente superclasses por base.
    Cada count representa ocorrência de label, não apenas número de exames.
    Como um exame pode ter múltiplas superclasses/acrônimos, fazemos explode por '|'.
    """
    reports: Dict[str, pd.DataFrame] = {}

    if metadata.empty or "source_base" not in metadata.columns:
        return reports

    # Superclasses por base.
    if "superclass" in metadata.columns:
        exploded = explode_pipe_column(metadata, "superclass")
        df = (
            exploded
            .groupby(["source_base", "superclass"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["source_base", "count"], ascending=[True, False])
        )
        totals = metadata.groupby("source_base").size().rename("n_exams_base").reset_index()
        df = df.merge(totals, on="source_base", how="left")
        df["fraction_over_base_exams"] = df["count"] / df["n_exams_base"].replace(0, np.nan)
        df.to_csv(output_test_dir / "superclass_by_base.csv", index=False, encoding="utf-8-sig")
        reports["superclass_by_base"] = df

    # Acrônimos por base.
    if "acronym" in metadata.columns:
        exploded = explode_pipe_column(metadata, "acronym")
        df = (
            exploded
            .groupby(["source_base", "acronym"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["source_base", "count"], ascending=[True, False])
        )
        totals = metadata.groupby("source_base").size().rename("n_exams_base").reset_index()
        df = df.merge(totals, on="source_base", how="left")
        df["fraction_over_base_exams"] = df["count"] / df["n_exams_base"].replace(0, np.nan)
        df.to_csv(output_test_dir / "acronym_by_base.csv", index=False, encoding="utf-8-sig")
        reports["acronym_by_base"] = df

    # SNOMED/Dx por base.
    if "dx_codes" in metadata.columns:
        exploded = explode_pipe_column(metadata, "dx_codes")
        df = (
            exploded
            .groupby(["source_base", "dx_codes"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["source_base", "count"], ascending=[True, False])
        )
        totals = metadata.groupby("source_base").size().rename("n_exams_base").reset_index()
        df = df.merge(totals, on="source_base", how="left")
        df["fraction_over_base_exams"] = df["count"] / df["n_exams_base"].replace(0, np.nan)
        df.to_csv(output_test_dir / "dx_by_base.csv", index=False, encoding="utf-8-sig")
        reports["dx_by_base"] = df

    # Warnings por base.
    if "warning_exame" in metadata.columns:
        exploded = explode_pipe_column(metadata, "warning_exame")
        exploded["warning_group"] = exploded["warning_exame"].astype(str).apply(
            lambda x: "dx_not_mapped" if x.startswith("dx_not_mapped:") else x
        )
        df = (
            exploded
            .groupby(["source_base", "warning_group"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["source_base", "count"], ascending=[True, False])
        )
        totals = metadata.groupby("source_base").size().rename("n_exams_base").reset_index()
        df = df.merge(totals, on="source_base", how="left")
        df["fraction_over_base_exams"] = df["count"] / df["n_exams_base"].replace(0, np.nan)
        df.to_csv(output_test_dir / "warning_by_base.csv", index=False, encoding="utf-8-sig")
        reports["warning_by_base"] = df

    # Sexo por base.
    if "sex" in metadata.columns:
        df = (
            metadata
            .assign(sex=metadata["sex"].fillna("<empty>").replace("", "<empty>"))
            .groupby(["source_base", "sex"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["source_base", "count"], ascending=[True, False])
        )
        totals = metadata.groupby("source_base").size().rename("n_exams_base").reset_index()
        df = df.merge(totals, on="source_base", how="left")
        df["fraction_over_base_exams"] = df["count"] / df["n_exams_base"].replace(0, np.nan)
        df.to_csv(output_test_dir / "sex_by_base.csv", index=False, encoding="utf-8-sig")
        reports["sex_by_base"] = df

    # Idade por base.
    if "age" in metadata.columns:
        tmp = metadata.copy()
        tmp["age_num"] = pd.to_numeric(tmp["age"], errors="coerce")
        df = (
            tmp
            .groupby("source_base", dropna=False)["age_num"]
            .agg(
                n_exams="size",
                valid_age_count="count",
                mean_age="mean",
                std_age="std",
                min_age="min",
                p25_age=lambda s: s.quantile(0.25),
                median_age="median",
                p75_age=lambda s: s.quantile(0.75),
                max_age="max",
            )
            .reset_index()
        )
        df["missing_age_count"] = df["n_exams"] - df["valid_age_count"]
        df.to_csv(output_test_dir / "age_by_base.csv", index=False, encoding="utf-8-sig")
        reports["age_by_base"] = df

    return reports


def warning_distribution(metadata: pd.DataFrame, output_test_dir: Path) -> pd.DataFrame:
    counter: Counter[str] = Counter()

    if "warning_exame" in metadata.columns:
        for v in metadata["warning_exame"].fillna(""):
            for w in split_pipe(v):
                if w.startswith("dx_not_mapped:"):
                    counter["dx_not_mapped"] += 1
                else:
                    counter[w] += 1

    rows = [{"warning": k, "count": v, "fraction_over_metadata": v / max(1, len(metadata))} for k, v in counter.most_common()]
    df = pd.DataFrame(rows)
    df.to_csv(output_test_dir / "warning_distribution.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# Validações de batches
# =============================================================================

def validate_single_batch(
    batch_path: Path,
    metadata_for_batch: pd.DataFrame,
    errors: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Retorna:
      batch_report_row
      per_exam_stats_rows
    """
    row: Dict[str, Any] = {
        "batch_file": batch_path.name,
        "exists": batch_path.exists(),
        "load_ok": False,
        "n_exams": np.nan,
        "shape": "",
        "expected_shape_per_exam": str(EXPECTED_SIGNAL_SHAPE_PER_EXAM),
        "shape_ok_N_12_5000": False,
        "shape_interpreted_as": "",
        "fs_seconds_leads_check": f"{TARGET_FS}Hz x {TARGET_SECONDS}s x {EXPECTED_N_LEADS}leads",
        "dtype": "",
        "has_x": False,
        "has_exam_ids": False,
        "has_global_indices": False,
        "nan_count": np.nan,
        "inf_count": np.nan,
        "max_abs": np.nan,
        "mean_abs": np.nan,
        "flatline_exam_count": np.nan,
        "spike_exam_count": np.nan,
        "metadata_rows": len(metadata_for_batch),
        "metadata_alignment_ok": False,
        "error": "",
    }

    per_exam_rows: List[Dict[str, Any]] = []

    if not batch_path.exists():
        row["error"] = "batch_file_not_found"
        errors.append(f"Batch referenciado no metadata não existe: {batch_path}")
        return row, per_exam_rows

    try:
        data = np.load(batch_path, allow_pickle=True)
    except Exception as exc:
        row["error"] = f"np_load_failed:{exc}"
        errors.append(f"Falha carregando batch {batch_path}: {exc}")
        return row, per_exam_rows

    keys = set(data.files)
    row["has_x"] = "x" in keys
    row["has_exam_ids"] = "exam_ids" in keys
    row["has_global_indices"] = "global_indices" in keys

    if "x" not in keys:
        row["error"] = "missing_x"
        errors.append(f"Batch sem array x: {batch_path}")
        return row, per_exam_rows

    x = data["x"]
    row["load_ok"] = True
    row["n_exams"] = int(x.shape[0]) if x.ndim >= 1 else 0
    row["shape"] = str(tuple(x.shape))
    row["dtype"] = str(x.dtype)

    # Verificação explícita do formato 500 x 10 x 12:
    # Esperado no arquivo: (N, 12, 5000), onde 5000 = 500 Hz * 10 s.
    if x.ndim == 3 and x.shape[1] == EXPECTED_N_LEADS and x.shape[2] == TARGET_SAMPLES:
        row["shape_ok_N_12_5000"] = True
        row["shape_interpreted_as"] = f"(N={x.shape[0]}, leads=12, samples=5000)"
    else:
        row["shape_ok_N_12_5000"] = False
        row["shape_interpreted_as"] = f"invalid:{tuple(x.shape)}"
        errors.append(
            f"{batch_path.name}: shape inválido. Esperado (N, 12, {TARGET_SAMPLES}) "
            f"= N exames x 12 leads x 500Hz*10s. Recebido {tuple(x.shape)}."
        )

    if x.ndim != 3:
        errors.append(f"{batch_path.name}: x deveria ter ndim=3, recebido {x.ndim}.")
    else:
        if x.shape[1] != len(TARGET_LEADS):
            errors.append(f"{batch_path.name}: número de leads inválido: {x.shape[1]} != {len(TARGET_LEADS)}.")
        if x.shape[2] != TARGET_SAMPLES:
            errors.append(f"{batch_path.name}: número de amostras inválido: {x.shape[2]} != {TARGET_SAMPLES}.")

    nan_count = int(np.isnan(x).sum()) if np.issubdtype(x.dtype, np.floating) else 0
    inf_count = int(np.isinf(x).sum()) if np.issubdtype(x.dtype, np.floating) else 0
    row["nan_count"] = nan_count
    row["inf_count"] = inf_count

    finite = x[np.isfinite(x)]
    if finite.size > 0:
        row["max_abs"] = float(np.max(np.abs(finite)))
        row["mean_abs"] = float(np.mean(np.abs(finite)))

    if nan_count > 0:
        errors.append(f"{batch_path.name}: contém {nan_count} NaN.")
    if inf_count > 0:
        errors.append(f"{batch_path.name}: contém {inf_count} Inf.")

    # Alinhamento batch_index e exam_ids.
    if "exam_ids" in keys and "batch_index" in metadata_for_batch.columns and "exam_id" in metadata_for_batch.columns:
        try:
            exam_ids = list(data["exam_ids"].astype(str))
            ok = True
            for _, mrow in metadata_for_batch.iterrows():
                idx = int(float(mrow["batch_index"]))
                expected_exam = str(mrow["exam_id"])
                if idx < 0 or idx >= len(exam_ids) or exam_ids[idx] != expected_exam:
                    ok = False
                    errors.append(
                        f"Alinhamento inválido em {batch_path.name}: batch_index={idx}, "
                        f"metadata_exam={expected_exam}, npz_exam={exam_ids[idx] if 0 <= idx < len(exam_ids) else '<out_of_range>'}"
                    )
                    break
            row["metadata_alignment_ok"] = ok
        except Exception as exc:
            row["metadata_alignment_ok"] = False
            errors.append(f"Falha validando alinhamento metadata/npz em {batch_path.name}: {exc}")

    # Estatísticas por exame.
    if x.ndim == 3:
        x64 = safe_float_array(x)
        exam_max_abs = np.nanmax(np.abs(x64), axis=(1, 2))
        lead_std = np.nanstd(x64, axis=2)
        exam_min_std = np.nanmin(lead_std, axis=1)
        exam_mean = np.nanmean(x64, axis=(1, 2))
        exam_std = np.nanstd(x64, axis=(1, 2))

        spike_mask = exam_max_abs > AMPLITUDE_PEAK_MV
        flatline_mask = np.any(lead_std < FLATLINE_STD_THRESHOLD, axis=1)

        row["spike_exam_count"] = int(np.sum(spike_mask))
        row["flatline_exam_count"] = int(np.sum(flatline_mask))

        exam_ids = data["exam_ids"].astype(str) if "exam_ids" in keys else np.array([f"{batch_path.stem}_{i}" for i in range(x.shape[0])])

        for i in range(x.shape[0]):
            per_exam_rows.append({
                "batch_file": batch_path.name,
                "batch_index": i,
                "exam_id": str(exam_ids[i]),
                "max_abs": float(exam_max_abs[i]),
                "min_lead_std": float(exam_min_std[i]),
                "signal_mean": float(exam_mean[i]),
                "signal_std": float(exam_std[i]),
                "has_spike": bool(spike_mask[i]),
                "has_flatline": bool(flatline_mask[i]),
                "has_nan": bool(np.isnan(x64[i]).any()),
                "has_inf": bool(np.isinf(x64[i]).any()),
            })

    data.close()
    return row, per_exam_rows


def validate_batches(
    batches_dir: Path,
    metadata: pd.DataFrame,
    output_test_dir: Path,
    errors: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    batch_paths = sorted(batches_dir.glob("batch_*.npz"), key=parse_batch_number)

    if not batch_paths:
        fail_or_warn(errors, f"Nenhum batch_*.npz encontrado em {batches_dir}")

    referenced = set(metadata["batch_file"].dropna().astype(str).tolist()) if "batch_file" in metadata.columns else set()
    referenced_paths = {batches_dir / x for x in referenced if x}
    all_paths = sorted(set(batch_paths).union(referenced_paths), key=lambda p: (parse_batch_number(p), p.name))

    batch_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    rng = random.Random(RANDOM_SEED)

    for batch_path in tqdm(all_paths, desc="Validando batches"):
        if "batch_file" in metadata.columns:
            meta_batch = metadata[metadata["batch_file"].astype(str) == batch_path.name].copy()
        else:
            meta_batch = pd.DataFrame()

        batch_row, per_exam_rows = validate_single_batch(batch_path, meta_batch, errors)
        batch_rows.append(batch_row)

        if MAX_EXAMS_FOR_DETAILED_STATS is None:
            sample_rows.extend(per_exam_rows)
        else:
            remaining = max(0, MAX_EXAMS_FOR_DETAILED_STATS - len(sample_rows))
            if remaining > 0:
                if len(per_exam_rows) <= remaining:
                    sample_rows.extend(per_exam_rows)
                else:
                    sample_rows.extend(rng.sample(per_exam_rows, remaining))

    batch_df = pd.DataFrame(batch_rows)
    sample_df = pd.DataFrame(sample_rows)

    batch_df.to_csv(output_test_dir / "batch_integrity_report.csv", index=False, encoding="utf-8-sig")
    sample_df.to_csv(output_test_dir / "sample_signal_stats.csv", index=False, encoding="utf-8-sig")

    return batch_df, sample_df


# =============================================================================
# Plotagem
# =============================================================================

def get_signal_from_metadata_row(row: pd.Series, batches_dir: Path) -> Optional[np.ndarray]:
    try:
        batch_file = str(row["batch_file"])
        batch_index = int(float(row["batch_index"]))
        data = np.load(batches_dir / batch_file, allow_pickle=True)
        x = data["x"][batch_index]
        data.close()
        return x
    except Exception:
        return None


def plot_ecg_12lead(
    signal_12xT: np.ndarray,
    title: str,
    out_path: Path,
    max_seconds: float = PLOT_SECONDS,
) -> None:
    ensure_dir(out_path.parent)

    x = np.asarray(signal_12xT)
    n = min(x.shape[1], int(max_seconds * TARGET_FS))
    t = np.arange(n) / TARGET_FS

    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)

    for i, lead in enumerate(TARGET_LEADS):
        axes[i].plot(t, x[i, :n], linewidth=0.8)
        axes[i].set_ylabel(lead, rotation=0, labelpad=18)
        axes[i].grid(True, linewidth=0.3, alpha=0.5)

    axes[-1].set_xlabel("Tempo (s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def create_plots(
    metadata: pd.DataFrame,
    batches_dir: Path,
    output_test_dir: Path,
    errors: List[str],
) -> List[Path]:
    plots_dir = output_test_dir / "plots"
    ensure_dir(plots_dir)

    created: List[Path] = []
    rng = random.Random(RANDOM_SEED)

    if metadata.empty:
        return created

    valid_meta = metadata.dropna(subset=["batch_file", "batch_index"]).copy()
    if valid_meta.empty:
        errors.append("Sem linhas válidas em metadata para plotagem.")
        return created

    n_random = min(N_RANDOM_PLOTS, len(valid_meta))
    random_rows = valid_meta.sample(n=n_random, random_state=RANDOM_SEED) if n_random > 0 else pd.DataFrame()

    for k, (_, row) in enumerate(random_rows.iterrows()):
        sig = get_signal_from_metadata_row(row, batches_dir)
        if sig is None:
            continue
        title = f"Random {k:02d} | exam_id={row.get('exam_id', '')} | base={row.get('source_base', '')}"
        out = plots_dir / f"random_{k:02d}_{row.get('exam_id', 'exam')}.png"
        plot_ecg_12lead(sig, title, out)
        created.append(out)

    if "warning_exame" in valid_meta.columns:
        warning_meta = valid_meta[valid_meta["warning_exame"].fillna("").astype(str).str.strip() != ""]
        n_warning = min(N_WARNING_PLOTS, len(warning_meta))
        if n_warning > 0:
            warning_rows = warning_meta.sample(n=n_warning, random_state=RANDOM_SEED)
            for k, (_, row) in enumerate(warning_rows.iterrows()):
                sig = get_signal_from_metadata_row(row, batches_dir)
                if sig is None:
                    continue
                warning = str(row.get("warning_exame", ""))[:120]
                title = f"Warning {k:02d} | exam_id={row.get('exam_id', '')} | {warning}"
                out = plots_dir / f"warning_{k:02d}_{row.get('exam_id', 'exam')}.png"
                plot_ecg_12lead(sig, title, out)
                created.append(out)

    if "superclass" in valid_meta.columns:
        superclass_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, row in valid_meta.iterrows():
            for sc in split_pipe(row.get("superclass", "")):
                superclass_to_indices[sc].append(idx)

        for superclass, indices in sorted(superclass_to_indices.items()):
            if not superclass:
                continue
            selected = rng.sample(indices, min(N_PER_SUPERCLASS_PLOTS, len(indices)))
            clean_sc = re.sub(r"[^A-Za-z0-9_\-]+", "_", superclass)[:60]
            for k, idx in enumerate(selected):
                row = valid_meta.loc[idx]
                sig = get_signal_from_metadata_row(row, batches_dir)
                if sig is None:
                    continue
                title = f"Superclass={superclass} | sample={k:02d} | exam_id={row.get('exam_id', '')}"
                out = plots_dir / f"superclass_{clean_sc}_{k:02d}_{row.get('exam_id', 'exam')}.png"
                plot_ecg_12lead(sig, title, out)
                created.append(out)

    return created


# =============================================================================
# Relatório final
# =============================================================================

def append_superclass_by_base_summary(lines: List[str], by_base_reports: Dict[str, pd.DataFrame]) -> None:
    lines.append("5) Superclasses por base")
    df = by_base_reports.get("superclass_by_base")
    if df is None or df.empty:
        lines.append("- Relatório superclass_by_base.csv não gerado ou vazio.")
        lines.append("")
        return

    for base in sorted(df["source_base"].dropna().unique()):
        sub = df[df["source_base"] == base].head(15)
        lines.append(f"- Base: {base}")
        for _, row in sub.iterrows():
            lines.append(
                f"  - {row['superclass']}: {int(row['count'])} "
                f"({float(row['fraction_over_base_exams']):.4%} sobre exames da base)"
            )
    lines.append("")


def summarize(
    output_test_dir: Path,
    metadata: pd.DataFrame,
    quality: pd.DataFrame,
    discarded: pd.DataFrame,
    batch_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    warning_df: pd.DataFrame,
    by_base_reports: Dict[str, pd.DataFrame],
    created_plots: Sequence[Path],
    errors: Sequence[str],
) -> Path:
    lines: List[str] = []
    lines.append("RELATÓRIO DE TESTE DO PRÉ-PROCESSAMENTO")
    lines.append("=" * 70)
    lines.append("")

    lines.append("1) Arquivos principais")
    lines.append(f"- metadata.csv: {len(metadata)} linhas")
    lines.append(f"- quality_report.csv: {len(quality)} linhas")
    lines.append(f"- discarded.csv: {len(discarded)} linhas")
    lines.append(f"- batches avaliados: {len(batch_df)}")
    lines.append(f"- plots gerados: {len(created_plots)}")
    lines.append("")

    lines.append("2) Verificação explícita de formato: 500 Hz x 10 s x 12 leads")
    lines.append(f"- Formato esperado por exame no .npz: (12, {TARGET_SAMPLES})")
    lines.append(f"- Formato esperado por batch no .npz: (N, 12, {TARGET_SAMPLES})")
    lines.append(f"- Interpretação: N exames x 12 leads x {TARGET_FS} Hz * {TARGET_SECONDS} s")
    if not batch_df.empty and "shape_ok_N_12_5000" in batch_df.columns:
        ok_count = int(batch_df["shape_ok_N_12_5000"].fillna(False).astype(bool).sum())
        total_count = len(batch_df)
        lines.append(f"- Batches com shape correto (N,12,5000): {ok_count}/{total_count}")
        bad = batch_df[~batch_df["shape_ok_N_12_5000"].fillna(False).astype(bool)]
        if not bad.empty:
            lines.append("- Batches com shape incorreto:")
            for _, row in bad.iterrows():
                lines.append(f"  - {row['batch_file']}: {row['shape']}")
    lines.append("")

    lines.append("3) Integridade dos batches")
    if not batch_df.empty:
        total_npz_exams = pd.to_numeric(batch_df["n_exams"], errors="coerce").sum()
        total_nan = pd.to_numeric(batch_df["nan_count"], errors="coerce").sum()
        total_inf = pd.to_numeric(batch_df["inf_count"], errors="coerce").sum()
        max_abs_global = pd.to_numeric(batch_df["max_abs"], errors="coerce").max()
        alignment_ok = batch_df["metadata_alignment_ok"].fillna(False).astype(bool).sum() if "metadata_alignment_ok" in batch_df.columns else 0

        lines.append(f"- Total de exames nos .npz: {int(total_npz_exams)}")
        lines.append(f"- Total de NaN: {int(total_nan)}")
        lines.append(f"- Total de Inf: {int(total_inf)}")
        lines.append(f"- Máximo absoluto global observado: {max_abs_global:.6g}" if pd.notna(max_abs_global) else "- Máximo absoluto global observado: n/a")
        lines.append(f"- Batches com alinhamento metadata/npz OK: {alignment_ok}/{len(batch_df)}")
    else:
        lines.append("- Sem dados de batch.")
    lines.append("")

    lines.append("4) Estatísticas amostrais do sinal")
    if not sample_df.empty:
        for col in ["max_abs", "min_lead_std", "signal_mean", "signal_std"]:
            s = pd.to_numeric(sample_df[col], errors="coerce")
            lines.append(
                f"- {col}: mean={s.mean():.6g}, std={s.std():.6g}, "
                f"min={s.min():.6g}, p50={s.quantile(0.50):.6g}, "
                f"p95={s.quantile(0.95):.6g}, max={s.max():.6g}"
            )
        lines.append(f"- Amostras com spike_amplitude pelo teste: {int(sample_df['has_spike'].sum())}")
        lines.append(f"- Amostras com flatline pelo teste: {int(sample_df['has_flatline'].sum())}")
        lines.append(f"- Amostras com NaN pelo teste: {int(sample_df['has_nan'].sum())}")
        lines.append(f"- Amostras com Inf pelo teste: {int(sample_df['has_inf'].sum())}")
    else:
        lines.append("- Estatísticas amostrais não geradas.")
    lines.append("")

    append_superclass_by_base_summary(lines, by_base_reports)

    lines.append("6) Distribuição por base")
    if "source_base" in metadata.columns and not metadata.empty:
        vc = metadata["source_base"].fillna("<empty>").value_counts()
        for k, v in vc.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- Indisponível.")
    lines.append("")

    lines.append("7) Warnings mais frequentes")
    if not warning_df.empty:
        for _, row in warning_df.head(20).iterrows():
            lines.append(f"- {row['warning']}: {row['count']} ({float(row['fraction_over_metadata']):.4%})")
    else:
        lines.append("- Nenhum warning em metadata.warning_exame.")
    lines.append("")

    lines.append("8) Falhas / inconsistências detectadas")
    if errors:
        for e in errors[:200]:
            lines.append(f"- {e}")
        if len(errors) > 200:
            lines.append(f"- ... mais {len(errors) - 200} mensagens omitidas.")
    else:
        lines.append("- Nenhuma falha crítica detectada.")
    lines.append("")

    lines.append("9) Arquivos gerados")
    for fname in [
        "batch_integrity_report.csv",
        "sample_signal_stats.csv",
        "metadata_distribution_report.csv",
        "warning_distribution.csv",
        "superclass_by_base.csv",
        "acronym_by_base.csv",
        "dx_by_base.csv",
        "warning_by_base.csv",
        "sex_by_base.csv",
        "age_by_base.csv",
    ]:
        lines.append(f"- {output_test_dir / fname}")
    lines.append(f"- plots: {output_test_dir / 'plots'}")

    report_path = output_test_dir / "preprocessing_test_summary.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    output_dir = pjoin(OUTPUT_DIR)
    output_test_dir = output_dir / "preprocessing_test"
    ensure_dir(output_test_dir)

    errors: List[str] = []

    paths = validate_structure(output_dir, errors)

    metadata = load_csv_if_exists(paths["metadata_path"], required=True, errors=errors)
    quality = load_csv_if_exists(paths["quality_path"], required=False, errors=errors)
    discarded = load_csv_if_exists(paths["discarded_path"], required=False, errors=errors)

    validate_metadata_columns(metadata, errors)

    dist_df = metadata_distribution_reports(metadata, output_test_dir) if not metadata.empty else pd.DataFrame()
    warning_df = warning_distribution(metadata, output_test_dir) if not metadata.empty else pd.DataFrame()
    by_base_reports = create_metadata_by_base_reports(metadata, output_test_dir) if not metadata.empty else {}

    batch_df, sample_df = validate_batches(paths["batches_dir"], metadata, output_test_dir, errors)

    created_plots = create_plots(metadata, paths["batches_dir"], output_test_dir, errors)

    report_path = summarize(
        output_test_dir=output_test_dir,
        metadata=metadata,
        quality=quality,
        discarded=discarded,
        batch_df=batch_df,
        sample_df=sample_df,
        dist_df=dist_df,
        warning_df=warning_df,
        by_base_reports=by_base_reports,
        created_plots=created_plots,
        errors=errors,
    )

    print("\nTeste do pré-processamento concluído.")
    print(f"Relatório: {report_path}")
    print(f"Diretório de teste: {output_test_dir}")
    print(f"Verificação de shape esperada: (N, 12, {TARGET_SAMPLES}) = N x 12 leads x 500 Hz * 10 s")
    print("Relatório de superclasses por base: superclass_by_base.csv")

    if errors:
        print(f"\nForam detectadas {len(errors)} inconsistências/avisos. Veja o relatório.")
    else:
        print("\nNenhuma inconsistência crítica detectada.")


if __name__ == "__main__":
    main()
