from preprocessing import load_and_preprocess_data
from tokenization import train_sentencepiece_model
from tokenization import SentencePieceVocab

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split, random_split

class DKTCDataset(Dataset):
    """
    Pytorch Dataset class를 상속받아, conversations, label을 모델 학습에 적합한 텐서 형태로 변환하는 클래스
    """

    def __init__(self, conversations, labels, vocab, max_length=400, is_test=False):
        """
        Args:
            conversations
            labels
            vocab: SentencePieceVocab 객체
            max_length: 시퀀스의 최대 길이(tokenization 이후의 길이). default=400
            is_test: 테스트 데이터셋 여부. default=False
        """
        self.vocab = vocab            # SentencePiece vocab 객체
        self.max_length = max_length  # 최대 시퀀스 길이 (잘림 방지)
        self.is_test = is_test        # 테스트 데이터 여부
        self.sequences = []
        self.labels = labels if not is_test else None

        # conversation -> sequence
        for conv in conversations:
            # [CLS] + conversation tokens + [EOS]
            sequence = [self.vocab.CLS_ID] \
                        + self.vocab.encode(conv) \
                        + [self.vocab.EOS_ID]

            # sequence의 길이 조절
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            else:
                pad_length = max_length - len(sequence)
                sequence = sequence + [self.vocab.PAD_ID] * pad_length

            self.sequences.append(sequence)

    def __len__(self):
        """데이터셋에 포함된 총 샘플의 개수 반환"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        GPT-1 방식: Next-token prediction을 위한 shifted sequences
        + classification label도 함께 반환
        
        Returns:
            dict: {
                'input_ids': 모델 입력으로 사용될 텐서 (마지막 토큰 제외),
                'target_ids': 예측 대상이 되는 텐서 (첫 토큰 제외),
                'labels': 분류 레이블 (test data인 경우 제외)
            }
        """
        sequence = self.sequences[idx]
        tokens = torch.tensor(sequence, dtype=torch.long)
        input_ids = tokens[:-1]   # 마지막 토큰 제외
        target_ids = tokens[1:]   # 첫 토큰 제외
        
        result = {
            'input_ids': input_ids,
            'target_ids': target_ids
        }
        
        # Test가 아닌 경우 레이블 추가
        if not self.is_test:
            result["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return result

def collate_fn(batch, pad_idx=0):
    """
    DataLoader의 배치 생성 함수
    """
    input_batch = [item['input_ids'] for item in batch]
    target_batch = [item['target_ids'] for item in batch]

    result = {
        'input_ids': torch.stack(input_batch),
        'target_ids': torch.stack(target_batch)
    }

    # labels가 있는 경우(즉 Test가 아닌 경우) result에 labels 추가
    if 'labels' in batch[0]:
        label_batch = [item['labels'] for item in batch]
        result['labels'] = torch.stack(label_batch)

    return result

def create_dataloaders(train_path, test_path, vocab_size=1320, max_length=400, batch_size=64, validation_split=0.1):
    """
    데이터를 로드, 전처리, 토큰화하고 PyTorch train/validation/test DataLoader를 생성하는 메인 함수.
    """
    # 1. 데이터 로드 및 전처리
    train_conversations, \
    train_labels, \
    test_conversations, \
    test_ids, \
    class_to_idx = load_and_preprocess_data(train_path, test_path)

    # 2. SentencePiece 토크나이저 모델 학습
    model_prefix = './configs/sentences'
    sp_model_path = train_sentencepiece_model(
        train_conversations, model_prefix=model_prefix, vocab_size=vocab_size
    )

    # 3. SentencePiece Vocab 로드
    vocab = SentencePieceVocab(sp_model_path)

    # 4. 전체 학습 데이터셋 생성
    full_train_dataset = DKTCDataset(
        train_conversations,
        train_labels,
        vocab,
        max_length=max_length,
        is_test=False
    )

    # 5. Train / Validation 데이터셋으로 분리
    num_train = len(full_train_dataset)
    val_size = int(num_train * validation_split)
    train_size = num_train - val_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 6. 테스트 데이터셋 생성
    test_dataset = DKTCDataset(
        test_conversations,
        labels=None,  # 테스트 데이터에는 레이블이 없습니다.
        vocab=vocab,
        max_length=max_length,
        is_test=True
    )

    # 7. DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=vocab.PAD_ID),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=vocab.PAD_ID),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 테스트 데이터는 셔플 X
        collate_fn=lambda batch: collate_fn(batch, pad_idx=vocab.PAD_ID),
    )

    print(f"\nTrain DataLoader 준비 완료: 총 {len(train_dataset)}개 conversations")
    print(f"Validation DataLoader 준비 완료: 총 {len(val_dataset)}개 conversations")
    print(f"Test DataLoader 준비 완료: 총 {len(test_dataset)}개 conversations.")

    return train_loader, val_loader, test_loader, vocab

