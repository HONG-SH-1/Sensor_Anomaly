"""
업로드 CSV와 학습 시 저장된 피처 목록(final_features.pkl) 정합 검사.
비지도 AE 추론: 라벨(machine_status/label)은 선택, 센서 피처는 체크포인트와 동일해야 함.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

# 학습·추론에서 피처로 쓰지 않아도 되는 열 (메타/평가용)
OPTIONAL_COLUMNS = frozenset({
    'machine_status',
    'label',
    'timestamp',
    'time',
    'datetime',
    'Unnamed: 0',
})


def validate_feature_columns(df: pd.DataFrame, features: List[str]) -> Tuple[str, List[str]]:
    """
    Returns:
        (info_message, extra_non_optional_columns) — extra는 참고용 이름 목록
    Raises:
        ValueError: 필수 피처 누락
    """
    missing = [c for c in features if c not in df.columns]
    if missing:
        show = missing[:25]
        more = f' 외 {len(missing) - len(show)}개' if len(missing) > len(show) else ''
        raise ValueError(
            '체크포인트(final_features.pkl)에 맞는 센서 컬럼이 부족합니다.\n'
            f'누락 ({len(missing)}개): {show}{more}\n'
            '학습에 사용한 것과 동일한 컬럼명·개수가 필요합니다. '
            '`data/sample_upload/README.txt`를 참고하세요.'
        )

    extra = [
        c for c in df.columns
        if c not in features and c not in OPTIONAL_COLUMNS
    ]
    msg = ''
    if extra:
        ex_show = extra[:15]
        tail = ' …' if len(extra) > 15 else ''
        msg = (
            f'참고: 피처로 사용되지 않는 열 {len(extra)}개 무시됨 '
            f'({", ".join(ex_show)}{tail})'
        )
    return msg, extra
