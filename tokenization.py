import sentencepiece as spm

def train_sentencepiece_model(conversations, 
                              model_prefix='./configs/spm_dktc', 
                              all_sentences_path='./configs/sentences.txt', 
                              vocab_size=1300):
    """
    주어진 conversations를 통해 SentencePiece 모델 학습
    
    Args:
        conversations
        model_prefix
        vocab_size: 개발자 지정 vocab의 크기. default=1200
    Return:
        model_file (str): 학습된 SentencePiece 모델 파일의 path
    """
    print("=" * 50)
    print("SentencePiece 모델 학습 중...")
    print("=" * 50)

    # 모든 문장을 하나의 텍스트 파일로 저장
    with open(all_sentences_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(conversations))

    # SentencePiece 학습 명령어 설정
    # [CLS] 토큰 추가 : 분류를 위한 시작 토큰 
    # (user_defined_symbols 명령어로 특수 토큰을 설정하면 자동으로 기본 특수 토큰 다음에 ID를 할당함)
    # --minloglevel=1: INFO 로그를 제외하고 WARNING, ERROR만 출력
    cmd = f'--input={all_sentences_path} \
           --model_prefix={model_prefix} \
           --vocab_size={vocab_size} \
           --model_type=unigram \
           --max_sentence_length=999999 \
           --pad_id=0 \
           --unk_id=1 \
           --bos_id=2 \
           --eos_id=3 \
           --user_defined_symbols=[CLS] \
           --minloglevel=1'

    # SentencePiece 모델 학습 실행
    spm.SentencePieceTrainer.Train(cmd)

    # 학습된 모델 파일 경로 생성
    model_file = f"{model_prefix}.model"
    print(f"\n모델 저장됨: {model_file}")
    print(f"Vocab 크기: {vocab_size}")
    return model_file


class SentencePieceVocab:
    """
    SentencePiece 모델을 쉽게 사용하기 위한 wrapper 클래스
    텍스트를 토큰 ID로 encoding하거나 토큰 ID를 다시 텍스트로 decoding하는 기능 제공
    """
    def __init__(self, sp_model_path):
        """
        Args:
            sp_model_path: 학습된 SentencePiece 모델 파일의 path
        """
        # SentencePiece 프로세서 초기화
        self.sp = spm.SentencePieceProcessor()
        # 학습된 모델 로드
        self.sp.Load(sp_model_path)

        # 특수 토큰 ID 정의
        self.PAD_ID = 0  # 패딩
        self.UNK_ID = 1  # 미등록 단어
        self.BOS_ID = 2  # 문장 시작 (BOS)
        self.EOS_ID = 3  # 문장 끝 (EOS)
        self.CLS_ID = 4  # Classification 토큰

        # 토큰 문자열 -> ID 매핑
        # <s>, </s> : 각각 BOS, EOS를 의미
        self.stoi = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, '[CLS]': 4}

        # ID -> 토큰 문자열 매핑 (전체 어휘)
        self.itos = [self.sp.IdToPiece(i) for i in range(self.sp.GetPieceSize())]

    def encode(self, sentence):
        """
        문장을 토큰 ID 리스트로 인코딩
        
        Args:
            sentence: 인코딩할 문자열
        """
        return self.sp.EncodeAsIds(sentence)

    def decode(self, ids):
        """
        토큰 ID 리스트를 문장으로 디코딩(특수 토큰은 제외)

        Args:
            ids: 인코딩된 토큰 ID list
        """
        return self.sp.DecodeIds([i for i in ids if i not in [0, 2, 3, 4]])

    def __len__(self):
        """어휘 사전 크기 반환"""
        return self.sp.GetPieceSize()