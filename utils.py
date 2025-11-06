'''
데이터 분석 및 시각화를 위한 유틸리티 함수 모음
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import sentencepiece as spm

from preprocessing import load_and_preprocess_data
from tokenization import SentencePieceVocab


def _train_spm_for_analysis(sentences, model_prefix='./temp_spm_for_analysis/tokenizer', vocab_size=1200):
    """
    분석을 위해 임시 SentencePiece 모델을 학습시키는 내부 함수.
    """
    # 모델 저장 디렉토리 생성
    model_dir = os.path.dirname(model_prefix)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 학습용 임시 문장 파일 생성
    temp_sentences_path = f"{model_prefix}_sentences.txt"
    with open(temp_sentences_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
        
    # SentencePiece 학습 명령어
    cmd = (
        f'--input={temp_sentences_path} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type=unigram '
        f'--max_sentence_length=999999 '
        f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
        f'--user_defined_symbols=[SEP],[CLS],[MASK]'
    )
    
    # 모델 학습
    spm.SentencePieceTrainer.Train(cmd)
    
    # 임시 파일 삭제
    os.remove(temp_sentences_path)
    
    model_file = f"{model_prefix}.model"
    return model_file

def analyze_token_lengths():
    """
    데이터를 로드하고 토큰화하여, 토큰 시퀀스 길이의 분포를 분석하고 시각화합니다.
    데이터셋의 `max_length` 파라미터를 결정하는 데 도움을 줍니다.
    """
    # 1. 데이터 로드 및 고유 문장 추출
    print("데이터 로드 및 전처리 중...")
    questions, answers = load_and_preprocess_data("./Data/aiffel-dl-thon-dktc-online-15/train.csv")
    print(f"총 {len(questions)}개의 질문-답변 쌍 로드 완료.")

    # 2. 분석용 임시 토크나이저 학습
    print("\n분석용 임시 토크나이저 학습 중...")
    # 고유 문장으로 토크나이저를 학습시켜 어휘를 구축합니다.
    unique_sentences = list(set(questions + answers))
    sp_model_path = _train_spm_for_analysis(unique_sentences)
    vocab = SentencePieceVocab(sp_model_path)
    print("임시 토크나이저 학습 완료.")

    # 3. 각 질문-답변 쌍을 토큰화하고 시퀀스 길이 계산
    print("\nQ-A 쌍을 토큰 시퀀스로 변환 및 길이 계산 중...")
    sequence_lengths = []
    for q, a in zip(questions, answers):
        # 데이터셋에서 생성될 실제 시퀀스 길이를 계산합니다: [BOS] + Q + [SEP] + A + [EOS]
        q_tokens = vocab.encode(q)
        a_tokens = vocab.encode(a)
        seq_len = 1 + len(q_tokens) + 1 + len(a_tokens) + 1  # BOS, SEP, EOS 토큰 3개 추가
        sequence_lengths.append(seq_len)

    # 4. 히스토그램 시각화 및 통계 출력
    print("결과 시각화 및 통계 출력...")
    plt.figure(figsize=(12, 6))
    plt.hist(sequence_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Token Sequence Length Histogram (BOS + Q + SEP + A + EOS)')
    plt.xlabel('Token Sequence Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # 통계량 계산 및 시각화
    max_len = np.max(sequence_lengths)
    mean_len = np.mean(sequence_lengths)
    p95 = np.percentile(sequence_lengths, 95)
    p99 = np.percentile(sequence_lengths, 99)

    plt.axvline(mean_len, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(p95, color='orange', linestyle='dashed', linewidth=2)
    plt.axvline(p99, color='green', linestyle='dashed', linewidth=2)
    
    plt.legend([f'Mean: {mean_len:.2f}', f'95th Percentile: {p95:.2f}', f'99th Percentile: {p99:.2f}'])
    plt.show()

    print("\n--- 토큰 시퀀스 길이 통계 ---")
    print(f"분석된 시퀀스 수: {len(sequence_lengths)}")
    print(f"최대 길이: {max_len}")
    print(f"평균 길이: {mean_len:.2f}")
    print(f"50 백분위수 (중앙값): {np.median(sequence_lengths)}")
    print(f"90 백분위수: {np.percentile(sequence_lengths, 90)}")
    print(f"95 백분위수: {p95}")
    print(f"99 백분위수: {p99}")
    print("\n이 분석은 데이터셋의 `max_length`를 결정하는 데 도움이 됩니다.")
    print(f"예를 들어, `max_length`를 {int(p99)} 정도로 설정하면 약 99%의 시퀀스를 포함할 수 있습니다.")

if __name__ == '__main__':
    analyze_token_lengths()
