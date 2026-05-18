"""
clean 노트북 → Kaggle용 no-viz 복제.

실행 (레포 루트에서):
  python scripts/build_kaggle_noviz.py
"""
import copy
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"
CLEAN = NB / "clean"
KAGGLE = NB / "kaggle"

SECTION2_START = re.compile(r"##\s*Section\s*2\.\s*Deep\s*EDA", re.I)
SECTION3_START = re.compile(r"##\s*Section\s*3\.", re.I)

REPLACEMENT_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## Section 2. Deep EDA — omitted (Kaggle no-viz)\n",
        "\n",
        "원본 `notebooks/clean/` 에는 7종 시각화가 있습니다. 본 빌드는 Kaggle에서 학습·튜닝만 수행합니다. "
        "`valid_sensors` / `df_vis` 는 아래 셀에서 clean과 동일 로직으로만 정의합니다.\n",
    ],
}

REPLACEMENT_CODE = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Section 2 시각화 생략 — 결측 기준만 동일 (clean 노트북과 동일)\n",
        "missing = df_raw[sensor_cols].isnull().mean().sort_values(ascending=False)\n",
        "drop_missing = missing[missing > MISSING_THRESH].index.tolist()\n",
        "valid_sensors = [c for c in sensor_cols if c not in drop_missing]\n",
        "df_vis = df_raw[valid_sensors + ['machine_status']].copy()\n",
        "print(f'Dropped by missing: {drop_missing}')\n",
        "print(f'Remaining sensors : {len(valid_sensors)}')\n",
    ],
}


def join_src(cell):
    return "".join(cell.get("source", []))


def set_source_string(cell, text: str):
    cell["source"] = [ln + "\n" for ln in text.split("\n")[:-1]]
    last = text.split("\n")[-1]
    if text.endswith("\n"):
        cell["source"].append(last + "\n")
    else:
        cell["source"].append(last)


def strip_code_plots(src: str) -> str:
    s = src

    s = re.sub(r"# VIF 비교 시각화\n.*?plt\.show\(\)\s*", "", s, flags=re.DOTALL)

    s = re.sub(
        r"\nfig, ax = plt\.subplots\(figsize=\(10, 3\)\).*?plt\.show\(\)\s*",
        "\n",
        s,
        flags=re.DOTALL,
    )

    s = s.replace("# ── 성능 비교 시각화 ──", "# ── 성능 비교 (표만; Kaggle no-viz) ──")
    s = re.sub(r"^mtitles\s*=.*\n", "", s, flags=re.M)
    s = re.sub(r"^plot_keys\s*=.*\n", "", s, flags=re.M)
    s = re.sub(r"^plot_titles\s*=.*\n", "", s, flags=re.M)
    s = re.sub(r"\nfig = plt\.figure\(figsize=\(18,9\)\).*?plt\.show\(\)\s*", "\n", s, flags=re.DOTALL)
    s = re.sub(r"\n# ROC 곡선\nfig, ax = plt\.subplots.*?plt\.show\(\)\s*", "\n", s, flags=re.DOTALL)

    s = re.sub(
        r"# Optuna 결과 시각화\n.*?plt\.show\(\)\s*",
        "",
        s,
        flags=re.DOTALL,
    )

    s = re.sub(
        r"# 최종 결과 종합 시각화\n.*?plt\.show\(\)\s*",
        "",
        s,
        flags=re.DOTALL,
    )

    s = re.sub(
        r"    fig, axes = plt\.subplots\(2,1, figsize=\(14,8\), sharex=True\)\n"
        r"    idx = np\.arange\(len\(errs\)\)\n"
        r"    axes\[0\]\.plot\(idx, errs.*?"
        r"    plt\.tight_layout\(\)\n",
        "    # Kaggle no-viz: Gradio 플롯 생략\n"
        "    fig = plt.figure(figsize=(6, 1))\n"
        "    fig.text(0.5, 0.5, 'Plot omitted (Kaggle no-viz)', ha='center', va='center')\n",
        s,
        flags=re.DOTALL,
    )

    return s


def build_nb(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    out = []
    i = 0
    while i < len(cells):
        c = cells[i]
        s = join_src(c)
        if c["cell_type"] == "markdown" and SECTION2_START.search(s):
            out.append(copy.deepcopy(REPLACEMENT_MD))
            out.append(copy.deepcopy(REPLACEMENT_CODE))
            i += 1
            while i < len(cells):
                s3 = join_src(cells[i])
                if cells[i]["cell_type"] == "markdown" and SECTION3_START.search(s3):
                    break
                i += 1
            continue
        out.append(copy.deepcopy(c))
        i += 1

    for c in out:
        if c["cell_type"] != "code":
            continue
        old = join_src(c)
        new = strip_code_plots(old)
        if new != old:
            set_source_string(c, new)

    nb["cells"] = out
    if out and out[0]["cell_type"] == "markdown":
        t = join_src(out[0])
        note = (
            "\n\n---\n**Kaggle 빌드:** 시각화 제거·축약. 전체 그림은 `notebooks/clean/` 로컬 실행.\n"
        )
        set_source_string(out[0], t + note)

    return nb


def main():
    pairs = [("vif_clean", "vif_kaggle"), ("all_clean", "all_kaggle")]
    for stem_in, stem_out in pairs:
        src = CLEAN / f"{stem_in}.ipynb"
        out = KAGGLE / f"{stem_out}.ipynb"
        nb = build_nb(src)
        KAGGLE.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Wrote", out.relative_to(REPO), "cells", len(nb["cells"]))


if __name__ == "__main__":
    main()
