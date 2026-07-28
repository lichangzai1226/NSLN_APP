from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_config(path: Path | None = None) -> dict:
    config_path = path or ROOT / "model_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _finite_float(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}不是有效数值。") from error
    if not np.isfinite(number):
        raise ValueError(f"{field_name}必须为有限数值。")
    return number


def extract_required_features(
    feature_values: Mapping[str, object], config: dict
) -> tuple[dict[str, float], list[str]]:
    extracted: dict[str, float] = {}
    warnings: list[str] = []

    for canonical_name, spec in config["radscore"]["features"].items():
        found_value = None
        found_alias = None
        for alias in spec["aliases"]:
            if alias in feature_values:
                found_value = feature_values[alias]
                found_alias = alias
                break

        if found_alias is None:
            raise KeyError(f"缺少模型必需特征：{canonical_name}")

        try:
            extracted[canonical_name] = _finite_float(found_value, canonical_name)
        except ValueError:
            extracted[canonical_name] = float(spec["median"])
            warnings.append(
                f"{canonical_name}为缺失或异常值，已使用开发队列固定中位数填补。"
            )

    return extracted, warnings


@dataclass(frozen=True)
class PredictionResult:
    rad_score: float
    clinical_probability: float
    fusion_probability: float
    linear_predictor: float
    standardized_features: dict[str, float]


def calculate_radscore(
    feature_values: Mapping[str, object], config: dict | None = None
) -> tuple[float, dict[str, float], list[str]]:
    cfg = config or load_config()
    extracted, warnings = extract_required_features(feature_values, cfg)
    score = float(cfg["radscore"]["intercept"])
    standardized: dict[str, float] = {}

    for canonical_name, value in extracted.items():
        spec = cfg["radscore"]["features"][canonical_name]
        z_value = (value - float(spec["mean"])) / float(spec["sd_ddof0"])
        standardized[canonical_name] = z_value
        score += float(spec["coefficient"]) * z_value

    return score, standardized, warnings


def encode_positive_sln(number_positive_sln: int) -> tuple[int, int]:
    if not isinstance(number_positive_sln, (int, np.integer)):
        raise ValueError("阳性SLN枚数必须为整数。")
    if number_positive_sln < 1:
        raise ValueError("本模型仅适用于至少1枚SLN阳性的患者。")
    return int(number_positive_sln == 2), int(number_positive_sln >= 3)


def sigmoid(linear_predictor: float) -> float:
    if linear_predictor >= 0:
        return 1.0 / (1.0 + math.exp(-linear_predictor))
    exp_value = math.exp(linear_predictor)
    return exp_value / (1.0 + exp_value)


def predict_from_features(
    feature_values: Mapping[str, object],
    number_positive_sln: int,
    config: dict | None = None,
) -> tuple[PredictionResult, list[str]]:
    cfg = config or load_config()
    rad_score, standardized, warnings = calculate_radscore(feature_values, cfg)
    sln_2, sln_3plus = encode_positive_sln(number_positive_sln)

    clinical = cfg["clinical_model"]
    clinical_lp = (
        float(clinical["intercept"])
        + float(clinical["coefficients"]["SLN_2_vs_1"]) * sln_2
        + float(clinical["coefficients"]["SLN_3plus_vs_1"]) * sln_3plus
    )

    fusion = cfg["fusion_model"]
    fusion_lp = (
        float(fusion["intercept"])
        + float(fusion["coefficients"]["Intratumoral_Rad_score"]) * rad_score
        + float(fusion["coefficients"]["SLN_2_vs_1"]) * sln_2
        + float(fusion["coefficients"]["SLN_3plus_vs_1"]) * sln_3plus
    )

    return (
        PredictionResult(
            rad_score=rad_score,
            clinical_probability=sigmoid(clinical_lp),
            fusion_probability=sigmoid(fusion_lp),
            linear_predictor=fusion_lp,
            standardized_features=standardized,
        ),
        warnings,
    )
