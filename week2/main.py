"""
week2/main.py — 과제 2: 벡터화 및 코사인 유사도 구현
1주차 코드를 이어서 확장. load_data 는 1주차 것을 그대로 재사용한다.

실행 (저장소 루트에서):
    python week2/main.py
"""

import os
import sys
import re #전처리 정규식용

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data/tech_docs.csv"
LINE = "=" * 62


def section(title: str) -> None:
    print("\n" + LINE)
    print(title)
    print(LINE)


# ── 1주차 재사용: 데이터 불러오기 (그대로 유지, utf-8-sig 중요) ──
def load_data(path: str) -> pd.DataFrame:
    """CSV 파일을 읽어 DataFrame으로 반환한다. 파일이 없으면 종료한다."""
    section("[1] 데이터 불러오기")

    if not os.path.exists(path):
        print(f"[오류] 파일을 찾을 수 없습니다: {path}")
        print("       - data/ 폴더에 tech_docs.csv 가 있는지 확인하세요.")
        print("       - 저장소 루트에서 `python week2/main.py` 로 실행했는지 확인하세요.")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")
    rows, cols = df.shape
    print(f"데이터 로드 완료: {rows}행 × {cols}열")
    return df


# ── 기능 1: 전처리(텍스트 정제) ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text) #소문자(a-z), 숫자(0-9), 공백을 제외한 모든 문자를 찾으라
    text = re.sub(r"\s+", " ", text).strip() #연속된 공백들을 하나로 바꾸고, 문자열의 맨 앞과 맨 뒤 공백을 제거
    return text


# ── 기능 2: 코사인 유사도 직접 구현 ────────────────────────────
def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    '''유사도 공식 = (A · B) / (||A|| × ||B||)
    내적(A·B)이 클수록, 두 벡터의 방향이 비슷할수록 값이 1에 가까워집니다.
    '''
    a = np.asarray(a, dtype=float).ravel()   # a를 1차원 실수 배열로 변환
    b = np.asarray(b, dtype=float).ravel()   # b를 1차원 실수 배열로 변환

    dot = np.dot(a, b)                       # 두 벡터의 내적 계산
    norm_a = np.linalg.norm(a)               # a 벡터의 크기(길이)
    norm_b = np.linalg.norm(b)               # b 벡터의 크기(길이)

    if norm_a == 0.0 or norm_b == 0.0:       # 길이가 0인 벡터는 유사도 계산 불가
        return 0.0

    return float(dot / (norm_a * norm_b))    # 코사인 유사도 계산 후 반환


# ── 기능 3: 키워드 Baseline 검색 ──────────────────────────────
def keyword_search(query: str, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    '''TF-IDF 없이, 질문 단어가 문서에 몇 개나 겹치는지만으로 점수를 매기는 단순 검색'''

    q_words = set(preprocess(query).split())          # 질문을 전처리한 뒤 단어 '집합'으로 변환(중복단어 제거)

    scores = [len(q_words & set(doc.split()))           # 질문과 문서의 공통 단어 개수 계산(set)
              for doc in df["content_clean"]]

    result = df.copy()                               # 원본 데이터프레임 복사
    result["score"] = scores                         # 계산한 점수를 score 열에 추가

    result = result.sort_values("score",             # 점수가 높은 순으로 정렬
                                ascending=False).head(top_k)  # 상위 top_k개 선택

    return result[["doc_id", "title", "category", "score"]]   # 필요한 열만 반환


# ── 기능 4: TF-IDF 벡터화 ─────────────────────────────────────
'''
구현 요건:
TfidfVectorizer로 content_clean 전체를 행렬로 변환합니다
행렬 크기(문서 수 × 단어 수)와 사용된 단어 수를 출력합니다
변환된 행렬과 vectorizer를 함께 반환합니다 (검색에서 재사용)
'''
def build_tfidf(df: pd.DataFrame):
    # TF-IDF 벡터 생성기 생성
    vectorizer = TfidfVectorizer(
        max_features=5000,      # 최대 5000개의 단어만 사용
        min_df=2,               # 2개 이상의 문서에 등장한 단어만 사용
        stop_words="english"    # 영어 불용어 제거
    )

    matrix = vectorizer.fit_transform(df["content_clean"])  # content_clean 전체를 TF-IDF 행렬로 변환

    # 모든 전처리와 필터링을 거친 뒤 TF-IDF 행렬의 열(feature)로 실제 사용된 단어들을 반환하는 메서드
    n_terms = len(vectorizer.get_feature_names_out())       # 사용된 단어(특징) 개수

    print(f"TF-IDF 행렬 크기: {matrix.shape} | 사용된 단어 수: {n_terms}")

    return matrix, vectorizer    # TF-IDF 행렬과 벡터라이저 반환


# ── 기능 5: TF-IDF Top-k 검색 ─────────────────────────────────
'''
구현 요건:
질문을 전처리한 뒤 vectorizer로 벡터화합니다
모든 문서 벡터와 cosine_similarity_numpy로 유사도를 계산합니다
유사도가 높은 순으로 Top-k를 반환합니다 (doc_id, title, category, similarity)
'''
def tfidf_search(query, df, matrix, vectorizer, top_k=3) -> pd.DataFrame:

    # fit_transform을 사용하지 않는 이유: 문서벡터와 질문벡터의 차원이 달라져 코사인 유사도를 계산할 수 없기 때문
    q_vec = vectorizer.transform([preprocess(query)]).toarray()[0]  # 질문을 TF-IDF 벡터로 변환 후 np배열로 변환

    doc_matrix = matrix.toarray()                                   # 희소행렬을 np배열로 변환(희소 행렬을 밀집 행렬로 바꿔 메모리를 많이 씁니다)

    sims = [
        cosine_similarity_numpy(q_vec, doc_matrix[i])               # 질문과 각 문서의 코사인 유사도 계산
        for i in range(doc_matrix.shape[0])
    ]

    result = df.copy()                                              # 원본 데이터 복사
    result["similarity"] = sims                                     # 유사도 점수 추가

    result = result.sort_values(                                    # 유사도가 높은 순으로 정렬
        "similarity", ascending=False
    ).head(top_k)                                                   # 상위 top_k개 선택

    return result[["doc_id", "title", "category", "similarity"]]    # 필요한 열만 반환


# ── 기능 6: 전체 연결 ─────────────────────────────────────────
def main():
    df = load_data(DATA_PATH)  # 데이터 불러오기

    df = df.dropna(subset=["content"]).reset_index(drop=True)  # 내용이 없는 문서 제거 및 인덱스 재정렬
    df["content_clean"] = df["content"].apply(preprocess)       # 문서 전처리 수행
    print("전처리 완료: content_clean 컬럼 생성")

    matrix, vectorizer = build_tfidf(df)                        # TF-IDF 행렬과 벡터라이저 생성

    # query = "python list comprehension"  # 검색할 질문
    query = "how does gradient descent work in machine learning"  # 검색할 질문
    print(f"\n질문: {query}\n")

    print("=== Keyword Baseline ===")
    print(keyword_search(query, df, top_k=3).to_string(index=False))  # 키워드 검색 결과 출력

    print("\n=== TF-IDF Search ===")
    print(tfidf_search(query, df, matrix, vectorizer, top_k=3).to_string(index=False))  # TF-IDF 검색 결과 출력

    # Baseline은 겹치는 단어 개수만 세므로 흔한 단어가 우연히 겹치는 문서도 상위에 올릴 수 있다.
    # TF-IDF는 중요한 단어에 더 높은 가중치를 주어 관련 문서를 더 정확하게 찾는다.


if __name__ == "__main__":
    main()