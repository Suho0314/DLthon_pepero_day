"""
SentencePiece 토크나이제이션 모듈
- SentencePiece 모델 학습
- Vocab 래퍼 클래스
- Dataset 클래스 (GPT-1 스타일)
"""

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


def train_sentencepiece_model(questions, answers, model_prefix='/Users/wansookim/Downloads/code_implementation/transformer_project_submit', vocab_size=1200):
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
    all_sentences_path = '/Users/wansookim/Downloads/code_implementation/transformer_project_submit/sentencepiece'
    with open(all_sentences_path, 'w', encoding='utf-8') as f:
        # 질문과 답변을 모두 합쳐서 줄바꿈으로 구분하여 저장
        f.write('\n'.join(questions + answers))

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


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset 클래스
    - 질문과 답변을 [SEP] 토큰으로 연결하여 하나의 시퀀스로 처리
    - input_ids: 전체 시퀀스 (질문 + [SEP] + 답변)
    - target_ids: input_ids를 한 칸 뒤로 shift (다음 토큰 예측)
    """
    def __init__(self, questions, answers, vocab, max_length=40):
        # 데이터 저장
        self.vocab = vocab  # SentencePiece vocab 객체
        self.max_length = max_length  # 최대 시퀀스 길이 (잘림 방지)
        self.sequences = []

        #모든 질문 답변 쌍을 시퀀스로 합쳐버리기
        for q, a in zip(questions, answers):
            sequence = (  [self.vocab.BOS_ID] + self.vocab.encode(q) + [self.vocab.SEP_ID] +  self.vocab.encode(a) + [self.vocab.EOS_ID])
        
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            else: 
                pad_length = max_length - len(sequence)
                sequence = sequence + [self.vocab.PAD_ID] * pad_length
            self.sequences.append(sequence)

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.sequences)


    def __getitem__(self, idx):
        """
    Shifted sequences 반환
        - input_ids:  [BOS, w1, w2, ..., wN]
        - target_ids: [w1, w2, ..., wN, EOS]
        """
        sequence = self.sequences[idx]
        tokens = torch.tensor(sequence, dtype=torch.long)
        # Next-token prediction을 위한 shifted sequences
        input_ids = tokens[:-1]   # 마지막 토큰 제외
        target_ids = tokens[1:]   # 첫 토큰 제외
        
        return {
            'input_ids': input_ids,    # ← SRC 대신
            'target_ids': target_ids   # ← TRG 대신
        }       


def collate_fn(batch, pad_idx=0):
    """
    DataLoader의 배치 생성 함수
    목적: 서로 다른 길이의 시퀀스를 같은 길이로 패딩하여 배치 생성
    """
    # 배치에서 SRC(질문)와 TRG(답변) 분리
    input_batch = [item['input_ids'] for item in batch]
    target_batch = [item['target_ids'] for item in batch]

    # 리스트의 텐서들을 하나의 텐서로 쌓기 (batch_size, seq_len)
    return {'input_ids': torch.stack(input_batch), 
            'target_ids': torch.stack(target_batch)}