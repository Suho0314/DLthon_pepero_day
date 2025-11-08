from dataset import create_dataloaders
from utils import hist_conversations_length

# 파일 경로 설정
train_file_path = "./Data/aiffel-dl-thon-dktc-online-15/train_3.csv"
test_file_path = "./Data/aiffel-dl-thon-dktc-online-15/test.csv"

# max_length 확인하기
hist_conversations_length(train_file_path, test_file_path)