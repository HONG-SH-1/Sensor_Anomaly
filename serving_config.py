"""
체크포인트와 함께 배포되는 서빙 설정(Kaggle 튜닝 결과와 동기화).
`models/checkpoints/model_serving_config.json`이 있으면 병합하고, 없거나 오류면 아래 기본값 사용.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from typing import Any, Dict

CONFIG_FILENAME = "model_serving_config.json"

# 노트북/앱에서 쓰던 초기 기본값 (JSON 없을 때)
DEFAULT_SERVING: Dict[str, Any] = {
    "best_params": {
        "LSTM-AE": {
            "hidden": 128,
            "latent": 64,
            "layers": 2,
            "dropout": 0.15,
        },
        "CNN1D-AE": {
            "hidden": 64,
            "latent": 8,
            "layers": 2,
            "dropout": 0.35,
        },
    },
    "window_configs": {
        "LSTM-AE": {"window_size": 30, "threshold_pct": 93},
        "CNN1D-AE": {"window_size": 50, "threshold_pct": 93},
    },
    "step_size": 10,
}


def _merge_serving(overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(DEFAULT_SERVING)
    for family in ("best_params", "window_configs"):
        if family not in overlay or not isinstance(overlay[family], dict):
            continue
        for model_name, patch in overlay[family].items():
            if not isinstance(patch, dict):
                continue
            if model_name not in out[family]:
                out[family][model_name] = {}
            out[family][model_name].update(patch)
    if "step_size" in overlay:
        try:
            out["step_size"] = int(overlay["step_size"])
        except (TypeError, ValueError):
            pass
    return out


def load_serving_config(checkpoint_dir: str) -> Dict[str, Any]:
    """
    checkpoint_dir 옆의 model_serving_config.json을 읽어 DEFAULT와 병합.
    파일이 없으면 DEFAULT_SERVING만 반환.
    """
    path = os.path.join(checkpoint_dir, CONFIG_FILENAME)
    if not os.path.isfile(path):
        return copy.deepcopy(DEFAULT_SERVING)
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        warnings.warn(f"{CONFIG_FILENAME} JSON 오류 — 기본값 사용: {e}", UserWarning)
        return copy.deepcopy(DEFAULT_SERVING)
    except OSError as e:
        warnings.warn(f"{CONFIG_FILENAME} 읽기 실패 — 기본값 사용: {e}", UserWarning)
        return copy.deepcopy(DEFAULT_SERVING)
    if not isinstance(raw, dict):
        return copy.deepcopy(DEFAULT_SERVING)
    return _merge_serving(raw)
