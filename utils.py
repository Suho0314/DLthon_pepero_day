import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Add seaborn import
import tempfile
import shutil

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

from preprocessing import load_and_preprocess_data
from tokenization import train_sentencepiece_model, SentencePieceVocab


def hist_conversations_length(train_path, test_path, vocab_size=1500):
    """
    학습 데이터에 대해 각 레이블별 토큰화 후 conversations의 length 분포를
    시각화하고, 시각화 이미지를 Images 디렉토리에 저장

    Args:
        train_path: train data csv path
        test_path: test data csv path
        vocab_size
    """
    print("데이터 로드 및 전처리 중...")
    train_conversations, train_labels, _, _, class_to_idx = \
        load_and_preprocess_data(train_path, test_path)

    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    model_prefix = os.path.join(temp_dir, 'temp_spm_for_hist')
    
    print("SentencePiece 모델 학습 중 (학습 데이터 기준)...")
    try:
        # 학습 데이터 기준으로 SentencePiece 모델 학습
        sp_model_path = train_sentencepiece_model(
            train_conversations, model_prefix=model_prefix, vocab_size=vocab_size
        )
        vocab = SentencePieceVocab(sp_model_path)

        print("토큰화 및 길이 계산 중...")
        # 학습 데이터의 길이 계산 ([CLS], [EOS] 포함)
        # [CLS], [EOS] 토큰은 encode함수에서 생성하지 않고 Dataset 클래스에서 추가하므로 +2 해야함
        all_train_lengths = [len(vocab.encode(conv)) + 2 for conv in train_conversations]

        # 레이블별 길이 분포 시각화
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        unique_labels = sorted(list(set(train_labels)))
        num_labels = len(unique_labels)

        # 서브플롯 그리드 크기 조정 (예: 2x3 또는 3x2)
        nrows = (num_labels + 1) // 2 if num_labels > 1 else 1
        ncols = 2 if num_labels > 0 else 1
        if num_labels == 0: # Handle case with no labels
            print("No labels found in training data to plot.")
            return

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        fig.suptitle('Distribution of Conversation Lengths by Label (Train Data)', fontsize=16)
        
        # axes가 1차원 배열일 경우를 대비하여 평탄화
        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, label_idx in enumerate(unique_labels):
            ax = axes[i]
            label_name = idx_to_class[label_idx]
            
            # 해당 레이블에 속하는 대화 길이 필터링
            label_lengths = [all_train_lengths[j] for j, lbl in enumerate(train_labels) if lbl == label_idx]

            if not label_lengths:
                ax.set_title(f'{label_name} (No data)')
                ax.set_xlabel('Length of Conversation (tokens)')
                ax.set_ylabel('Frequency')
                continue

            sns.histplot(label_lengths, bins=50, kde=True, ax=ax, color=sns.color_palette("tab10")[i % 10])
            
            mean_len = np.mean(label_lengths)
            median_len = np.median(label_lengths)
            percentile_95 = np.percentile(label_lengths, 95)

            ax.axvline(mean_len, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_len:.2f}')
            ax.axvline(median_len, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_len:.2f}')
            ax.axvline(percentile_95, color='purple', linestyle='dotted', linewidth=2, label=f'95th: {percentile_95:.2f}')

            ax.set_title(f'{label_name} (count: {len(label_lengths)})')
            ax.set_xlabel('Length of Conversation (tokens)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)
        
        # 사용하지 않는 서브플롯 숨기기
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout() # 전체 타이틀을 위한 여백 확보
        plt.show()

        # --- 그래프 이미지 저장 --- #
        images_dir = './Images'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        save_path = os.path.join(images_dir, 'conversation_length_distributions_by_label.png')
        fig.savefig(save_path)
        print(f"그래프가 {save_path} 에 저장되었습니다.")

    finally:
        # 임시 디렉토리 및 파일 삭제
        print(f"임시 디렉토리({temp_dir}) 및 파일을 삭제합니다.")
        shutil.rmtree(temp_dir)