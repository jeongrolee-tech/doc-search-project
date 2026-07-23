"""
week1/main.py
기술 문서 검색 시스템 - 1주차: 데이터 탐색 (EDA, Exploratory Data Analysis)

실행 방법 (저장소 루트에서):
    python week1/main.py
"""

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 상수: 경로는 한 곳에서만 관리한다 (매직 스트링 금지)
# ---------------------------------------------------------------------------
DATA_PATH = "data/tech_docs.csv"

LINE = "=" * 62
SHORT_DOC_THRESHOLD = 50  # '짧은 문서' 판정 기준 (단어 수)


def section(title: str) -> None:
    """출력 블록마다 구분선과 제목을 붙여 가독성을 확보한다."""
    print("\n" + LINE)
    print(title)
    print(LINE)


# ---------------------------------------------------------------------------
# 기능 1 — 데이터 불러오기
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """CSV 파일을 읽어 DataFrame으로 반환한다. 파일이 없으면 프로그램을 종료한다."""
    section("[1] 데이터 불러오기")

    if not os.path.exists(path):
        print(f"[오류] 파일을 찾을 수 없습니다: {path}")
        print("       - data/ 폴더에 tech_docs.csv 가 있는지 확인하세요.")
        print("       - 저장소 루트에서 `python week1/main.py` 로 실행했는지 확인하세요.")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")

    rows, cols = df.shape
    print(f"데이터 로드 완료: {rows}행 × {cols}열")
    return df


# ---------------------------------------------------------------------------
# 기능 2 — 데이터 구조 확인
# ---------------------------------------------------------------------------
def explore_structure(df: pd.DataFrame) -> None:
    """DataFrame의 크기, 컬럼명·자료형, 상위 5행을 출력한다."""
    section("[2] 데이터 구조 확인")

    rows, cols = df.shape
    print(f"전체 행 수 : {rows}")
    print(f"전체 열 수 : {cols}")

    print("\n--- 컬럼명 및 자료형 ---")
    for name, dtype in df.dtypes.items():
        kind = "문자형" if dtype == "object" else "수치형"
        print(f"  {name:<10} : {str(dtype):<8} ({kind})")

    print("\n--- 상위 5행 미리보기 ---")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 30,
    ):
        print(df.head(5))


# ---------------------------------------------------------------------------
# 기능 3 — 카테고리 분포 확인
# ---------------------------------------------------------------------------
def show_category_distribution(df: pd.DataFrame) -> dict:
    """카테고리별 문서 수·비율·평균 단어 수를 계산해 출력하고 딕셔너리로 반환한다."""
    section("[3] 카테고리 분포 확인")

    total = len(df)
    counts = df["category"].value_counts()  # 많은 순으로 정렬되어 반환됨

    print("--- 카테고리별 문서 수 / 비율 ---")
    for category, count in counts.items():
        ratio = count / total * 100
        print(f"  {category:<10} : {count:>3}개  ({ratio:5.1f}%)")

    # 반복문 + 딕셔너리로 카테고리별 평균 단어 수 계산
    result: dict = {}
    print("\n--- 카테고리별 평균 문서 길이 (단어 수) ---")
    for category in counts.index:
        subset = df[df["category"] == category]
        word_counts = subset["content"].apply(lambda text: len(str(text).split()))
        avg_words = word_counts.mean()

        result[category] = {
            "count": int(counts[category]),
            "ratio": round(counts[category] / total * 100, 1),
            "avg_words": round(float(avg_words), 1),
        }
        print(f"  {category:<10} : 평균 {avg_words:6.1f}단어")

    return result


# ---------------------------------------------------------------------------
# 기능 4 — 결측치 현황 파악
# ---------------------------------------------------------------------------
def _severity(ratio: float) -> str:
    """결측치 비율(%)을 심각도 라벨로 변환한다."""
    if ratio < 5:
        return "낮음"
    elif ratio < 20:
        return "주의"
    return "높음"


def check_missing(df: pd.DataFrame) -> dict:
    """컬럼별 결측치 수·비율·심각도를 계산해 출력하고 딕셔너리로 반환한다."""
    section("[4] 결측치 현황")

    total = len(df)
    missing_counts = df.isnull().sum()  # 컬럼별 결측치 개수 (Series)

    result: dict = {}
    has_missing = []
    no_missing = []

    for column, count in missing_counts.items():
        count = int(count)
        ratio = count / total * 100
        result[column] = {
            "missing": count,
            "ratio": round(ratio, 2),
            "severity": _severity(ratio) if count > 0 else "없음",
        }
        (has_missing if count > 0 else no_missing).append(column)

    print("--- 결측치가 있는 컬럼 ---")
    if has_missing:
        for column in has_missing:
            info = result[column]
            print(
                f"  {column:<10} : {info['missing']:>3}개 "
                f"({info['ratio']:5.2f}%)  심각도: {info['severity']}"
            )
    else:
        print("  결측치가 있는 컬럼: 없음")

    print("\n--- 결측치가 없는 컬럼 ---")
    print(f"  {', '.join(no_missing) if no_missing else '없음'}")

    return result


# ---------------------------------------------------------------------------
# 기능 5 — NumPy로 문서 길이 통계량 계산
# ---------------------------------------------------------------------------
def numpy_doc_stats(df: pd.DataFrame) -> dict:
    """content 컬럼의 단어 수를 NumPy 배열로 만들어 통계량을 계산·검증한다."""
    section("[5] NumPy 문서 길이 통계량")

    # 배열을 만들기 전에 결측치 행을 먼저 제거한다
    valid = df.dropna(subset=["content"]).copy()
    dropped = len(df) - len(valid)
    print(f"결측치 제거: {dropped}행 제외 → {len(valid)}행으로 계산\n")

    valid["word_count"] = valid["content"].apply(lambda text: len(str(text).split()))
    lengths = np.array(valid["word_count"].tolist())  # 리스트 → NumPy 배열

    mean_ = np.mean(lengths)
    std_ = np.std(lengths, ddof=1)  # 표본표준편차 (pandas와 동일 방식)
    median_ = np.median(lengths)
    min_ = np.min(lengths)
    max_ = np.max(lengths)

    print("--- 문서 길이 통계량 (NumPy) ---")
    print(f"  평균     (mean)   : {mean_:8.2f} 단어")
    print(f"  표준편차 (std)    : {std_:8.2f} 단어")
    print(f"  중앙값   (median) : {median_:8.2f} 단어")
    print(f"  최솟값   (min)    : {min_:8.0f} 단어")
    print(f"  최댓값   (max)    : {max_:8.0f} 단어")

    # 조건 필터링(불리언 마스킹)으로 짧은 문서 추출
    short_mask = lengths < SHORT_DOC_THRESHOLD
    short_lengths = lengths[short_mask]

    print(f"\n--- {SHORT_DOC_THRESHOLD}단어 미만 문서 ---")
    print(f"  개수: {len(short_lengths)}개 "
          f"(전체의 {len(short_lengths) / len(lengths) * 100:.1f}%)")
    if len(short_lengths) > 0:
        short_docs = valid[short_mask]
        for _, row in short_docs.iterrows():
            print(f"    - [{row['doc_id']}] {row['title'][:40]:<40} : {row['word_count']}단어")
    else:
        print("    해당 문서 없음")

    # pandas describe() 결과와 대조 검증
    desc = valid["word_count"].describe()
    print("\n--- pandas describe() 와 수치 비교 ---")
    print(f"  {'항목':<18}{'NumPy':>12}{'pandas':>12}   일치")
    comparisons = [
        ("mean",   mean_,   desc["mean"]),
        ("std",    std_,    desc["std"]),
        ("median", median_, desc["50%"]),
        ("min",    min_,    desc["min"]),
        ("max",    max_,    desc["max"]),
    ]
    for label, np_value, pd_value in comparisons:
        match = "일치" if np.isclose(np_value, pd_value) else "불일치"
        print(f"  {label:<18}{np_value:>12.4f}{pd_value:>12.4f}   {match}")

    print(f"\n  (참고) ddof 미지정 시 np.std() = {np.std(lengths):.4f} "
          f"→ pandas와 다름. 반드시 ddof=1 을 지정할 것.")

    return {
        "mean": float(mean_),
        "std": float(std_),
        "median": float(median_),
        "min": int(min_),
        "max": int(max_),
        "short_docs": int(len(short_lengths)),
    }


# ---------------------------------------------------------------------------
# 기능 6 — 전체 연결
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data(DATA_PATH)

    explore_structure(df)
    category_stats = show_category_distribution(df)
    missing_stats = check_missing(df)
    length_stats = numpy_doc_stats(df)

    section("[6] main() 함수로 전체 연결")
    print(f"  카테고리 수     : {len(category_stats)}개")
    print(f"  결측치 보유 컬럼: "
          f"{sum(1 for v in missing_stats.values() if v['missing'] > 0)}개")
    print(f"  평균 문서 길이  : {length_stats['mean']:.1f}단어")


if __name__ == "__main__":
    main()