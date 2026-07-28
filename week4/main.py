"""
week4/main.py — 과제 4: 전체 파이프라인 통합
과제 1~3의 모든 단계(로드 → 전처리 → 벡터화 → 검색 → 평가 → 오류분석)를
하나의 실행 가능한 main() 흐름으로 잇는다.

week3 대비 바뀐 점(이 파일에서 새로 한 것):
  (A) build_tfidf 에 text_column 인자 추가 → 본문/제목가중본문을 같은 함수로 벡터화
  (B) demo_search 추가 → "예시 검색 1건"으로 파이프라인이 실제로 도는지 눈으로 확인 (기능 1 요건)
  (C) main() 끝에 기능 2(선택) 제목 가중치 개선 후 재측정 블록 추가

실행 (저장소 루트 doc-search-project/ 에서):
    python week4/main.py            # 통합 파이프라인 실행
    python week4/main.py --catalog  # 문서 목록 출력 (평가셋 만들 때 사용)
"""

import os                                          # 파일 존재 여부 확인(os.path.exists)에 사용
import sys                                         # 강제 종료(sys.exit), 실행 인자(sys.argv) 읽기에 사용
import re                                          # 전처리 정규식용

import numpy as np                                 # 벡터 내적·크기 계산 및 평균(np.mean) 계산
import pandas as pd                                # CSV 로드와 표 형태(DataFrame) 처리
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF 벡터화 도구

DATA_PATH = "data/tech_docs.csv"                   # 데이터 경로(프로젝트 루트 기준 상대경로)
TOP_K = 3                                          # 평가 기준 상위 k개 (Precision@3, MRR 모두 이 값을 씀)
LINE = "=" * 62                                    # 출력 구분선


def section(title: str) -> None:
    """출력 가독성을 위해 제목을 구분선으로 감싸 출력한다."""
    print("\n" + LINE)                             # 위쪽 구분선
    print(title)                                   # 제목
    print(LINE)                                    # 아래쪽 구분선


# ══════════════════════════════════════════════════════════════
#  과제 1~2 재사용 구간 (로직 동일 — 정렬 안정성만 보강)
# ══════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """CSV 파일을 읽어 DataFrame으로 반환한다. 파일이 없으면 종료한다."""
    section("[1] 데이터 불러오기")

    if not os.path.exists(path):                   # 파일이 없으면 뒤 로직이 전부 무의미하므로 즉시 중단
        print(f"[오류] 파일을 찾을 수 없습니다: {path}")
        print("       - 저장소 루트에서 `python week4/main.py` 로 실행했는지 확인하세요.")
        sys.exit(1)                                # 종료코드 1 = 비정상 종료(쉘/CI가 실패로 인식)

    df = pd.read_csv(path, encoding="utf-8-sig")   # utf-8-sig: 엑셀이 붙이는 BOM 문자를 자동 제거
    rows, cols = df.shape                          # shape = (행 수, 열 수) 튜플
    print(f"데이터 로드 완료: {rows}행 × {cols}열")
    return df


def preprocess(text: str) -> str:
    """텍스트를 소문자 영숫자만 남긴 형태로 정제한다."""
    text = text.lower()                            # 대소문자 통일 ("Python"과 "python"을 같은 단어로 취급)
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # 소문자·숫자·공백을 제외한 모든 문자를 공백으로 치환
    text = re.sub(r"\s+", " ", text).strip()       # 연속 공백을 하나로 합치고 앞뒤 공백 제거
    return text


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """유사도 공식 = (A · B) / (||A|| × ||B||)
    내적이 클수록, 두 벡터의 방향이 비슷할수록 값이 1에 가까워진다.
    """
    a = np.asarray(a, dtype=float).ravel()         # a를 1차원 실수 배열로 변환
    b = np.asarray(b, dtype=float).ravel()         # b를 1차원 실수 배열로 변환

    dot = np.dot(a, b)                             # 두 벡터의 내적
    norm_a = np.linalg.norm(a)                     # a 벡터의 크기(길이)
    norm_b = np.linalg.norm(b)                     # b 벡터의 크기(길이)

    if norm_a == 0.0 or norm_b == 0.0:             # 길이 0 벡터는 방향이 없어 각도 계산 불가 → 0으로 방어
        return 0.0

    return float(dot / (norm_a * norm_b))          # 코사인 유사도 반환


def keyword_search(query: str, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """TF-IDF 없이, 질문 단어가 문서에 몇 개나 겹치는지만으로 점수를 매기는 단순 검색"""
    q_words = set(preprocess(query).split())       # 질문을 전처리 후 단어 '집합'으로 변환(중복 제거)

    scores = [len(q_words & set(doc.split()))      # &는 교집합 → 질문과 문서의 공통 단어 개수
              for doc in df["content_clean"]]      # 모든 문서에 대해 반복

    result = df.copy()                             # 원본 훼손을 막기 위해 복사본에서 작업
    result["score"] = scores                       # 계산한 점수를 score 열로 추가

    # kind="mergesort" = 안정 정렬(stable sort).
    # Baseline 점수는 정수라 동점이 대량 발생하는데, 기본 quicksort는 불안정 정렬이라
    # 동점 문서의 순서가 실행마다 뒤바뀔 수 있다 → 평가 점수가 흔들려 재현이 안 된다.
    result = result.sort_values(
        "score", ascending=False, kind="mergesort" # 점수 내림차순 + 동점 시 원래 순서 유지
    ).head(top_k)                                  # 상위 top_k개만 남김

    return result[["doc_id", "title", "category", "score"]]   # 필요한 열만 반환


def build_tfidf(df: pd.DataFrame, text_column: str = "content_clean"):
    """text_column 전체를 TF-IDF 행렬로 변환하고, 행렬과 벡터라이저를 함께 반환한다.

    ★ week4 변경점: text_column 인자를 받도록 일반화했다.
      기본값 "content_clean"이면 과제 2와 완전히 동일하게 동작하고,
      "content_weighted"를 넘기면 제목 가중치 버전을 같은 코드로 벡터화할 수 있다.
      (설정값은 그대로 둬야 기능 2에서 '입력 텍스트만 바꾼' 효과를 순수하게 측정할 수 있다.)
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,      # 빈도 상위 5000개 단어만 사용(차원 폭발 방지)
        min_df=2,               # 2개 미만 문서에 등장한 단어는 제외 ⚠️ 희귀 핵심어까지 잘려나감(실패 분석 참고)
        stop_words="english",   # the, is, how 같은 영어 불용어 제거
    )

    matrix = vectorizer.fit_transform(df[text_column])      # fit(어휘집 학습) + transform(벡터화)을 한 번에
    n_terms = len(vectorizer.get_feature_names_out())       # 최종적으로 열(feature)이 된 단어 개수

    print(f"TF-IDF 행렬 크기: {matrix.shape} | 사용된 단어 수: {n_terms}")

    return matrix, vectorizer                      # 검색 때 재사용해야 하므로 둘 다 반환


def tfidf_search(query, df, matrix, vectorizer, top_k=3) -> pd.DataFrame:
    """질문을 같은 어휘집으로 벡터화한 뒤 모든 문서와 코사인 유사도를 계산해 Top-k를 반환한다."""
    # transform만 쓰는 이유: fit_transform을 다시 부르면 어휘집이 새로 만들어져
    # 문서 벡터와 질문 벡터의 차원·의미가 어긋나 유사도 계산이 불가능해진다.
    q_vec = vectorizer.transform([preprocess(query)]).toarray()[0]  # 질문 → TF-IDF 벡터 → 1차원 배열

    doc_matrix = matrix.toarray()                  # 희소행렬을 밀집행렬로 변환(문서 수가 커지면 메모리 폭증)

    sims = [
        cosine_similarity_numpy(q_vec, doc_matrix[i])   # 질문 vs 각 문서의 코사인 유사도
        for i in range(doc_matrix.shape[0])             # shape[0] = 문서 수
    ]

    result = df.copy()                             # 원본 보호용 복사
    result["similarity"] = sims                    # 유사도 점수를 열로 추가

    result = result.sort_values(
        "similarity", ascending=False, kind="mergesort"  # 유사도 내림차순 + 동점 순서 고정
    ).head(top_k)                                  # 상위 top_k개 선택

    return result[["doc_id", "title", "category", "similarity"]]


def demo_search(query, df, matrix, vectorizer, k: int = TOP_K) -> None:
    """★ week4 신규: 예시 질문 1건으로 tfidf_search가 실제로 도는지 눈으로 확인한다.

    기능 1 요건 "예시 질문 하나로 tfidf_search를 실행해 검색이 동작하는지 확인"을 담당.
    평가 지표(숫자)만 보면 파이프라인이 도는지 감이 안 오므로, 사람 눈으로
    '이 질문에 이런 문서가 이 순서로 나온다'를 먼저 확인시켜 주는 것이다.
    """
    section(f"=== 예시 검색: {query} ===")
    result = tfidf_search(query, df, matrix, vectorizer, top_k=k)
    print(result.to_string(index=False))           # index=False → 판다스 행 인덱스 숨김


# ══════════════════════════════════════════════════════════════
#  평가셋 구성 (EVAL_SET) — 과제 3 재사용
# ══════════════════════════════════════════════════════════════
# tech_docs.csv 실제 내용과 대조해 작성. 5개 카테고리(Python/Git/AI기초/NumPy/pandas)에 3개씩 균등 배분.
# 일부 질문은 문서의 표현과 일부러 다른 단어를 써서(동의어·어형 변화) 한계가 드러나도록 설계했다.
EVAL_SET = [
    # ── Python ──
    {"query": "how to write a list comprehension in python", "relevant_doc_ids": ["D001", "D059"]},  # 정답 2개(정석 + 유사 문서)
    {"query": "python decorators explained",                 "relevant_doc_ids": ["D010"]},          # decorators가 D010에만 등장 → min_df 함정
    {"query": "handling errors with try except",             "relevant_doc_ids": ["D005"]},          # 문서는 "exception handling" → 표현 불일치 함정
    # ── Git ──
    {"query": "git branching basics",                        "relevant_doc_ids": ["D013"]},          # 제목과 거의 일치하는 쉬운 질문
    {"query": "how to resolve merge conflicts in git",       "relevant_doc_ids": ["D018"]},          # 자연어 문장형 질문
    {"query": "temporarily save uncommitted changes",        "relevant_doc_ids": ["D019"]},          # "stash"라는 단어를 안 쓴 의도 기반 질문
    # ── AI기초 ──
    {"query": "how does backpropagation work in neural networks", "relevant_doc_ids": ["D030"]},     # 명세 예시와 동일한 실패 케이스
    {"query": "what is gradient descent optimization",       "relevant_doc_ids": ["D023"]},          # 핵심어가 제목에 그대로 존재
    {"query": "preventing overfitting with regularization",  "relevant_doc_ids": ["D026", "D027", "D056"]},  # 정답 3개(관련 문서 묶음)
    # ── NumPy ──
    {"query": "numpy broadcasting rules",                    "relevant_doc_ids": ["D034"]},          # 전문 용어 그대로 사용
    {"query": "how to reshape a numpy array",                "relevant_doc_ids": ["D037"]},          # 동사형 질문
    {"query": "boolean masking to filter numpy arrays",      "relevant_doc_ids": ["D039"]},          # 용어 + 목적 혼합
    # ── pandas ──
    {"query": "pandas groupby aggregation",                  "relevant_doc_ids": ["D044"]},          # 함수명 기반 질문
    {"query": "dealing with missing values nan in dataframe", "relevant_doc_ids": ["D043"]},         # 결측치 = missing values = NaN 동의어 테스트
    {"query": "sql style join between dataframes",           "relevant_doc_ids": ["D045"]},          # 타 도메인 용어(SQL)로 물어보기
]


def print_catalog(df: pd.DataFrame) -> None:
    """평가셋을 만들 때 참고할 문서 목록을 출력한다."""
    section("[문서 카탈로그] 이 목록을 보고 EVAL_SET 의 doc_id 를 채우세요")
    print(df[["doc_id", "title", "category"]].to_string(index=False))  # index=False → 판다스 인덱스 숨김


def validate_eval_set(eval_set, df: pd.DataFrame) -> None:
    """평가셋의 doc_id가 실제 데이터에 존재하는지 검증한다.

    존재하지 않는 ID는 '영원히 못 맞히는 문제'가 되어 점수를 조용히 깎는다.
    코드는 오류 없이 잘 돌아가면서 성능만 나빠지므로, 가장 발견하기 어려운 버그다.
    """
    valid_ids = set(df["doc_id"])                  # 실제 존재하는 ID 집합 (조회 속도 O(1))
    broken = []                                    # 잘못된 항목을 모아둘 리스트

    for item in eval_set:                          # 평가셋 항목 하나씩
        for doc_id in item["relevant_doc_ids"]:    # 정답 ID 하나씩
            if doc_id not in valid_ids:            # 실제 데이터에 없는 ID라면
                broken.append((item["query"], doc_id))  # 어떤 질문의 어떤 ID가 문제인지 기록

    if broken:                                     # 하나라도 잘못됐으면 실행을 막는다(fail fast)
        print("\n[오류] 평가셋에 존재하지 않는 doc_id가 있습니다:")
        for q, d in broken:
            print(f"  - '{q}' → {d}")
        print("\n  `python week4/main.py --catalog` 로 실제 doc_id를 확인해 EVAL_SET을 고치세요.")
        sys.exit(1)

    print(f"평가셋 크기: {len(eval_set)}개 질문 (doc_id 검증 통과)")


# ══════════════════════════════════════════════════════════════
#  Precision@k — 과제 3 재사용
# ══════════════════════════════════════════════════════════════
def precision_at_k(retrieved_ids, relevant_ids, k: int = 3) -> float:
    """상위 k개 검색 결과 중 정답이 차지하는 비율.

    분모가 k로 고정이라, 정답이 1개뿐인 질문은 k=3일 때 최대값이 1/3(≈0.333)이다.
    즉 0.333이 나왔다면 '33점'이 아니라 '사실상 만점'일 수 있으니 해석에 주의.
    """
    if k <= 0:                                     # 0으로 나누기 방어
        return 0.0

    top_k = retrieved_ids[:k]                      # 리스트 슬라이싱으로 상위 k개만 자른다
    hits = len(set(top_k) & set(relevant_ids))     # 교집합 크기 = 상위 k개 안에 든 정답 개수

    return hits / k                                # 비율로 환산해 반환


# ══════════════════════════════════════════════════════════════
#  Reciprocal Rank (이것의 평균이 MRR = Mean Reciprocal Rank) — 과제 3 재사용
# ══════════════════════════════════════════════════════════════
def reciprocal_rank(retrieved_ids, relevant_ids, k: int = 3) -> float:
    """첫 정답이 등장한 순위의 역수. 1위→1.0, 2위→0.5, 3위→0.333, Top-k에 없으면 0.0"""
    relevant = set(relevant_ids)                   # 매 반복마다 리스트를 훑지 않도록 집합으로 변환

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):  # start=1 → 순위를 0이 아닌 1부터 매김
        if doc_id in relevant:                     # 정답을 만나면
            return 1.0 / rank                      # 그 순위의 역수를 반환하고 즉시 종료(첫 정답만 본다)

    return 0.0                                     # 끝까지 못 찾으면 0점


# ══════════════════════════════════════════════════════════════
#  Baseline vs TF-IDF 성능 비교 — 과제 3 재사용
# ══════════════════════════════════════════════════════════════
def run_evaluation(eval_set, search_fn, k: int = 3) -> dict:
    """평가셋 전체를 돌며 Precision@k와 MRR의 평균을 낸다.

    search_fn: (query, k) -> DataFrame(doc_id 컬럼 포함) 형태의 래퍼 함수.
    검색기 종류를 몰라도 채점할 수 있게 인터페이스를 통일한 것(의존성 역전).
    """
    precisions, rrs = [], []                       # 질문별 점수를 쌓아둘 리스트

    for item in eval_set:                          # 평가셋 질문 하나씩
        result = search_fn(item["query"], k)       # 검색 실행 → DataFrame
        retrieved_ids = result["doc_id"].tolist()  # doc_id 열만 파이썬 리스트로 추출(순위 순서 유지)

        precisions.append(precision_at_k(retrieved_ids, item["relevant_doc_ids"], k))  # 정확도 누적
        rrs.append(reciprocal_rank(retrieved_ids, item["relevant_doc_ids"], k))        # 역순위 누적

    return {
        "k": k,                                    # 어떤 k로 잰 값인지 함께 기록(해석에 필수)
        "precision@k": float(np.mean(precisions)), # 전체 평균 정확도
        "MRR": float(np.mean(rrs)),                # 역순위의 평균 = MRR
    }


def print_comparison(results: dict, k: int) -> None:
    """{검색기이름: 지표dict} 를 표로 나란히 출력한다."""
    section("=== 성능 비교 ===")
    print(f"{'':<20}{'Precision@' + str(k):>14}{'MRR':>9}")   # 헤더 (<는 좌측정렬, >는 우측정렬)
    for name, m in results.items():
        print(f"{name:<20}{m['precision@k']:>14.4f}{m['MRR']:>9.4f}")  # .4f = 소수점 4자리 고정


# ══════════════════════════════════════════════════════════════
#  실패 케이스 분석 — 과제 3 재사용
# ══════════════════════════════════════════════════════════════
def analyze_failures(eval_set, search_fn, k: int = 3, vectorizer=None, n_docs=None) -> None:
    """Top-k 안에 정답이 하나도 없는 질문을 골라 원인 단서와 함께 출력한다."""
    section(f"=== 실패 케이스 (Top-{k} 안에 정답 없음) ===")

    failed = 0                                     # 실패 건수 카운터
    for item in eval_set:
        result = search_fn(item["query"], k)       # 해당 질문으로 검색
        retrieved_ids = result["doc_id"].tolist()  # 검색된 doc_id 목록

        if reciprocal_rank(retrieved_ids, item["relevant_doc_ids"], k) == 0.0:  # RR이 0 = 완전 실패
            failed += 1
            print(f"질문: {item['query']}")
            print(f"  정답 doc_id : {item['relevant_doc_ids']}")
            print(f"  검색 결과   : {retrieved_ids}")

            # 진단 ①: 정답이 전체 순위에서 실제 몇 위였나?
            # 4위(아깝게 밀림)와 40위(아예 못 찾음)는 원인도 처방도 완전히 다르다.
            if n_docs:
                full_ids = search_fn(item["query"], n_docs)["doc_id"].tolist()  # 전체 문서를 순위대로
                ranks = [full_ids.index(doc_id) + 1                                  # index는 0부터라 +1
                         for doc_id in item["relevant_doc_ids"] if doc_id in full_ids]
                print(f"  → 정답의 실제 순위: {ranks} / 전체 {n_docs}건")

            # 진단 ②: 질문의 어떤 단어가 TF-IDF 어휘집에서 통째로 빠졌는가?
            # min_df·stop_words 때문에 잘려나간 단어는 유사도에 0으로 기여한다.
            if vectorizer is not None:
                vocab = set(vectorizer.get_feature_names_out())                 # 실제 사용된 단어 집합
                q_terms = preprocess(item["query"]).split()                     # 질문을 같은 방식으로 전처리
                oov = [t for t in q_terms if t not in vocab]                    # OOV = Out-Of-Vocabulary
                if oov:
                    print(f"  → 어휘집에 없는 단어: {oov} (점수 계산에 전혀 기여 못 함)")
            print()

    if failed == 0:
        print("실패 케이스 없음")
    else:
        print(f"총 {failed}건 실패 / 전체 {len(eval_set)}건")


# ══════════════════════════════════════════════════════════════
#  main() — 전체 파이프라인 통합 (과제 4의 핵심)
# ══════════════════════════════════════════════════════════════
def main():
    # ── [1] 로드 (과제 1) ──
    df = load_data(DATA_PATH)                                   # ① 데이터 로드
    df = df.dropna(subset=["content"]).reset_index(drop=True)  # 내용이 빈 문서 제거 + 인덱스 0부터 재정렬
    df["content_clean"] = df["content"].apply(preprocess)      # ② 전처리 결과를 새 열로 저장
    print("전처리 완료: content_clean 컬럼 생성")

    if "--catalog" in sys.argv:                                # 실행 인자에 --catalog가 있으면
        print_catalog(df)                                      # 문서 목록만 출력하고
        return                                                 # 평가는 건너뛰고 종료

    # ── [2] 벡터화 (과제 2) ──
    matrix, vectorizer = build_tfidf(df)                       # ③ TF-IDF 행렬·벡터라이저 생성 (기본 content_clean)

    # ── [3] 검색 동작 확인: 예시 1건 (기능 1 요건) ──
    demo_search("git merge conflicts", df, matrix, vectorizer) # ④ 파이프라인이 실제로 검색을 수행하는지 눈으로 확인

    # ── [4] 평가셋 검증 (과제 3) ──
    section("[평가셋 검증]")
    validate_eval_set(EVAL_SET, df)                            # ⑤ 평가셋 doc_id 유효성 검사

    # lambda 래퍼: 시그니처가 다른 두 검색 함수를 (query, k) 형태 하나로 통일한다.
    # 덕분에 run_evaluation은 "어떤 검색기인지" 몰라도 동일한 방식으로 채점할 수 있다.
    baseline_fn = lambda q, k: keyword_search(q, df, top_k=k)                    # 키워드 검색 래퍼
    tfidf_fn = lambda q, k: tfidf_search(q, df, matrix, vectorizer, top_k=k)     # TF-IDF 검색 래퍼

    # ── [5] Baseline vs TF-IDF 채점·비교 (과제 3) ──
    results = {
        "Keyword Baseline": run_evaluation(EVAL_SET, baseline_fn, k=TOP_K),      # ⑥ Baseline 채점
        "TF-IDF": run_evaluation(EVAL_SET, tfidf_fn, k=TOP_K),                   #    TF-IDF 채점
    }
    print_comparison(results, TOP_K)                                            # 표로 비교 출력

    # ── [6] 실패 케이스 분석 (과제 3) ──
    analyze_failures(EVAL_SET, tfidf_fn, k=TOP_K,
                     vectorizer=vectorizer, n_docs=len(df))                     # ⑦ TF-IDF 실패 케이스 분석

    # ══════════════════════════════════════════════════════════
    #  기능 2 (선택) — 제목 가중치 개선 후 성능 재측정
    #  아이디어: 제목엔 핵심 주제어가 압축돼 있다. 제목을 3회 반복해 본문 앞에 붙이면
    #  그 단어들의 TF(단어 빈도)가 커져 TF-IDF 점수가 오르고, 질문이 제목 단어와
    #  겹칠 때 코사인 유사도가 상승한다. (Elasticsearch의 title^3 필드 가중치의 원시 버전)
    # ══════════════════════════════════════════════════════════
    section("[선택] 제목 가중치 개선 후 재측정")

    # title이 NaN(float)이면 preprocess가 .lower()에서 터지므로 빈 문자열로 방어
    df["title_clean"] = df["title"].fillna("").apply(preprocess)               # 제목도 본문과 같은 방식으로 정제

    # (제목 + 공백)을 3번 반복해 본문 앞에 붙인다 → "title title title 본문..."
    # 판다스 문자열 Series는 * 정수 = 반복, + = 이어붙이기를 원소별로 지원한다.
    df["content_weighted"] = (df["title_clean"] + " ") * 3 + df["content_clean"]

    w_matrix, w_vectorizer = build_tfidf(df, text_column="content_weighted")   # 제목가중본문으로 TF-IDF 재구성
    w_tfidf_fn = lambda q, k: tfidf_search(q, df, w_matrix, w_vectorizer, top_k=k)  # 개선판 검색 래퍼

    improved = {
        "TF-IDF (기본)":     results["TF-IDF"],                                 # 위에서 이미 잰 값 재사용(중복 계산 X)
        "TF-IDF (제목가중)": run_evaluation(EVAL_SET, w_tfidf_fn, k=TOP_K),      # 개선판 채점
    }
    print_comparison(improved, TOP_K)                                          # 기본 vs 개선 나란히 비교

    '''
    [측정 결과로부터 도출한 개선 방향]
    - 동의어·관련어를 구분하지 못하는 한계 확인 (network vs networks, exception vs try/except)
    - min_df=2 → 희귀 핵심어 삭제 → min_df=1 또는 BM25로 교체
    - 어형 불일치 → Porter Stemmer / lemmatization 도입
    - 쿼리 희석 & 동의어 → 임베딩 기반 의미 검색 (다음 단계)
      * 제목 가중치는 '제목에 핵심어가 담긴 질문'에서만 효과가 큼(그 외엔 변화 미미)
    '''


if __name__ == "__main__":                         # 이 파일을 직접 실행할 때만 main() 호출
    main()                                         # (다른 파일이 import할 때는 실행되지 않음)