import pandas as pd
import re

def preprocess_sentence(sentence):
    """
    단일 문장을 전처리하는 함수
    목적: 텍스트 데이터를 정제하여 모델 학습에 적합한 형태로 만듦
    """
    # 입력 검증: 비어있거나 None인 경우 빈 문자열 반환
    if pd.isna(sentence) or sentence is None:
        return ""

    # 문자열로 변환 (안전장치)
    sentence = str(sentence)

    # 정규표현식으로 필요한 문자만 남기기
    # ㄱ-ㅎ: 자음, ㅏ-ㅣ: 모음, 가-힣: 완성형 한글
    # a-zA-Z: 영어, 0-9: 숫자, \s: 공백
    # .,!?~: 문장부호, ㅠㅜ: 이모티콘
    # 나머지는 모두 공백으로 치환
    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s.,!?~ㅠㅜ]', ' ', sentence) #re.sub는 정규표현식(regex)을 사용하여 특정 패턴의 문자를 찾아 다른 문자로 바꿉니다.

    # 연속된 여러 공백을 하나의 공백으로 통일
    sentence = re.sub(r'\s+', ' ', sentence)

    # 설명: \s는 공백, +는 "1개 이상"을 의미합니다. 즉, \s+는 "1개 이상의 연속된 공백"(예: " ", " ", " ")을 찾습니다.

    # 동작: 이렇게 찾은 연속된 공백 뭉치를 **하나의 공백(' ')**으로 압축합니다.

    # 이유: 바로 앞 단계에서 Hi!!^^가 Hi 처럼 여러 공백으로 바뀔 수 있습니다. 단어 사이처럼 불필요하게 공백이 많은 것은 단어 사이와 동일하게 취급되어야 합니다.

    # 문장 앞뒤 공백 제거
    sentence = sentence.strip()

    # 연속된 문장부호 정리 (예: !!! -> !, ??? -> ?)
    # ([!?.])를 캡처하고 \1+로 반복을 찾아서 r'\1'로 하나만 남김
    sentence = re.sub(r'([!?.])\1+', r'\1', sentence)

    return sentence


def load_and_preprocess_data(file_path):
    """
    CSV 파일에서 질문-답변 데이터를 로드하고 전처리
    """
    print("=" * 50)
    print("데이터 로드 및 전처리 중...")
    print("=" * 50)

    # pandas로 CSV 파일 읽기
    df = pd.read_csv(file_path)
    print(f"전체 데이터: {len(df)} 쌍")

    # 전처리된 질문과 답변을 저장할 리스트 초기화
    questions = []
    answers = []

    # 모든 질문-답변 쌍을 순회
    for i, (q, a) in enumerate(zip(df['Q'], df['A'])):
        # 각각 전처리 적용
        clean_q = preprocess_sentence(q)
        clean_a = preprocess_sentence(a)

        # 둘 다 유효한 문장인 경우만 저장
        if clean_q and clean_a:
            questions.append(clean_q)
            answers.append(clean_a)

        # 진행 상황 출력 (매 1000개마다)
        if (i + 1) % 1000 == 0:
            print(f"진행: {i + 1}/{len(df)}")

    print(f"\n전처리 후 유효한 쌍: {len(questions)}")
    print("\n샘플 데이터:")

    # 처음 3개의 샘플 데이터 출력하여 확인
    for i in range(min(3, len(questions))):
        print(f"Q: {questions[i]}")
        print(f"A: {answers[i]}\n")

    return questions, answers
