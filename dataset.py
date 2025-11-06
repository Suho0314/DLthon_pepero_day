from preprocessing import load_and_preprocess_data
from tokenization import train_sentencepiece_model
from tokenization import SentencePieceVocab

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DLThonDataset(Dataset):
    """
    PyTorch Dataset 클래스
    - 질문과 답변을 [SEP] 토큰으로 연결하여 하나의 시퀀스로 처리
    - input_ids: 전체 시퀀스 (질문 + [SEP] + 답변)
    - target_ids: input_ids를 한 칸 뒤로 shift (다음 토큰 예측)
    """
    def __init__(self, questions, answers, vocab, max_length=120):
        # 데이터 저장
        self.vocab = vocab  # SentencePiece vocab 객체
        self.max_length = max_length  # 최대 시퀀스 길이 (잘림 방지)
        self.sequences = []

        # 모든 질문 답변 쌍을 시퀀스로 합쳐버리기
        for q, a in zip(questions, answers):
            sequence = ([self.vocab.BOS_ID] 
                        + self.vocab.encode(q) 
                        + [self.vocab.SEP_ID] 
                        +  self.vocab.encode(a) 
                        + [self.vocab.EOS_ID])
        
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

def prepare_dataloader(file_path, vocab_size=1200, max_length=120, batch_size=64):
    '''
    데이터를 로드, 전처리, 토큰화하고 PyTorch DataLoader를 생성하는 메인 함수.
    '''
    # 1. 데이터 로드 및 전처리
    questions, answers = load_and_preprocess_data(file_path)

    # 2. SentencePiece 토크나이저 모델 학습 (경로 수정된 버전 사용)
    model_prefix = './configs/sentences'
    sp_model_path = train_sentencepiece_model(
        questions, answers, model_prefix=model_prefix, vocab_size=vocab_size
    )

    # 3. SentencePiece Vocab 로드
    vocab = SentencePieceVocab(sp_model_path)

    # 4. PyTorch Dataset 생성 (이 파일에 정의된 클래스 사용)
    dataset = DLThonDataset(questions, answers, vocab, max_length=max_length)

    # 5. PyTorch DataLoader 생성
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=vocab.PAD_ID),
    )

    print(f"\nDataLoader 준비 완료. 총 {len(dataset)}개의 샘플.")

    return data_loader, vocab


# 스크립트 직접 실행 시 테스트 코드
if __name__ == '__main__':
    train_file_path = "./Data/aiffel-dl-thon-dktc-online-15/train.csv"

    # 데이터로더와 vocab 준비 (작은 배치 사이즈로 테스트)
    train_loader, vocab = prepare_dataloader(train_file_path, batch_size=4)

    # 샘플 배치 확인
    print("\n--- 샘플 배치 확인 ---")
    try:
        sample_batch = next(iter(train_loader))
        print("Input IDs Shape:", sample_batch['input_ids'].shape)
        print("Target IDs Shape:", sample_batch['target_ids'].shape)
        print("\nSample 1 Input (Token IDs):", sample_batch['input_ids'][0])
        print("Sample 1 Decoded:", vocab.decode(sample_batch['input_ids'][0].tolist()))
        print("\nSample 1 Target (Token IDs):", sample_batch['target_ids'][0])
        print("Sample 1 Decoded:", vocab.decode(sample_batch['target_ids'][0].tolist()))
        print("=" * 25)
    except StopIteration:
        print("데이터로더가 비어있습니다. 데이터셋 크기를 확인해주세요.")
