"""
SentencePiece 토크나이제이션 모듈
- SentencePiece 모델 학습
- Vocab 래퍼 클래스
- Dataset 클래스 (GPT-1 스타일)
"""

import sentencepiece as spm


def train_sentencepiece_model(questions, answers, model_prefix='./configs/spm_model', vocab_size=1200):
    """
    SentencePiece 모델 학습
  GPT-1에 맞춤: [SEP] 토큰을 user_defined_symbols에 추가
    - [SEP]: 질문과 답변을 구분하는 특수 토큰
    - [CLS]: 시퀀스 시작을 나타내는 토큰 (사용 안 함)
    - [MASK]: 마스킹용 토큰 (사용 안 함)    
    """
    print("=" * 50)
    print("SentencePiece 모델 학습 중...")
    print("=" * 50)

    # 모든 문장을 하나의 텍스트 파일로 저장 (SentencePiece 입력 형식)
    all_sentences_path = './configs/sentences'
    with open(all_sentences_path, 'w', encoding='utf-8') as f:
        # 질문과 답변을 모두 합쳐서 줄바꿈으로 구분하여 저장
        f.write('\n'.join(questions))
        f.write(answers[len(answers)-1])

    # SentencePiece 학습 명령어 설정
    # GPT-1: user_defined_symbols에 [SEP], [CLS], [MASK] 추가
    cmd = f'--input={all_sentences_path} \
           --model_prefix={model_prefix} \
           --vocab_size={vocab_size} \
           --model_type=unigram \
           --max_sentence_length=999999 \
           --pad_id=0 \
           --unk_id=1 \
           --bos_id=2 \
           --eos_id=3 \
           --user_defined_symbols=[SEP],[CLS],[MASK]' # GPT-1: 특수 토큰 정의
    # --input: 학습 데이터 경로
    # --model_prefix: 저장될 모델 파일명 접두사
    # --vocab_size: 어휘 사전 크기 (8000개의 서브워드)
    # --model_type: unigram 언어 모델 사용
    # --pad_id: 패딩 토큰 ID (0)
    # --unk_id: 미등록 토큰 ID (1)
    # --bos_id: 문장 시작 토큰 ID (2)
    # --eos_id: 문장 종료 토큰 ID (3)

    # SentencePiece 모델 학습 실행
    spm.SentencePieceTrainer.Train(cmd)

    # 학습된 모델 파일 경로 생성
    model_file = f"{model_prefix}.model"
    print(f"\n모델 저장됨: {model_file}")
    return model_file


class SentencePieceVocab:
    """
    SentencePiece 모델 래퍼 클래스
    목적: SentencePiece 모델을 쉽게 사용하기 위한 인터페이스 제공
    """
    def __init__(self, sp_model_path):
        # SentencePiece 프로세서 초기화
        self.sp = spm.SentencePieceProcessor()
        # 학습된 모델 로드
        self.sp.Load(sp_model_path)

        # 특수 토큰 ID 정의
        self.PAD_ID = 0  # 패딩 (빈 공간 채우기)
        self.UNK_ID = 1  # 미등록 단어
        self.BOS_ID = 2  # 문장 시작 (Beginning Of Sentence)
        self.EOS_ID = 3  # 문장 끝 (End Of Sentence)
        self.SEP_ID = 4

        # 토큰 문자열 -> ID 매핑
        self.stoi = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, '[SEP]': 4} # GPT-1: SEP 토큰

        # ID -> 토큰 문자열 매핑 (전체 어휘)
        self.itos = [self.sp.IdToPiece(i) for i in range(self.sp.GetPieceSize())]

    def encode(self, sentence):
        """문장을 토큰 ID 리스트로 인코딩"""
        return self.sp.EncodeAsIds(sentence)

    def decode(self, ids):
        """
        토큰 ID 리스트를 문장으로 디코딩
        특수 토큰(pad, bos, eos)은 제외하고 디코딩
        """
        return self.sp.DecodeIds([i for i in ids if i not in [0, 2, 3]])

    def __len__(self):
        """어휘 사전 크기 반환"""
        return self.sp.GetPieceSize()
