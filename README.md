# DLthon_pepero_day
This is a repository for Aiffel DLthon  
---
- Team name : pepero_day  
- Teamates : 김완수, 이수호, 이영석, 최원진  
---
# Structure of this repository
```bash
DLthon_pepero_day/
├── configs/
├── Data/
│   └── aiffel-dl-thon-dktc-online-15.zip
├── Images/
├── models/
│   ├── model1.py
│   ├── model2.py
│   ├── model3.py
│   └── model4.py
├── dataset.py
├── evaluate.py
├── model.py
├── preprocessing.py
├── README.md
├── requirements.txt
├── tokenization.py
├── train.py
└── utils.py
```
---
- folders
    - Data : Kaggle에서 다운받은 원본 데이터셋
    - configs : 모델 설정, 데이터 경로 등 프로젝트의 주요 설정 값들을 저장하는 파일을 담는 디렉토리
    - Images : 결과 리포트나 발표 자료에 사용될 이미지 파일을 담는 디렉토리
    - modles : 다양한 모델 아키텍처 실험 및 관리를 위한 디렉토리
- files
    - dataset.py : 데이터셋을 불러오고, 모델 입력으로 사용할 수 있는 형태로 변환
    - evaluate.py : 학습된 모델의 성능 평가
    - model.py : 메인 모델 정의
    - preprocessing.py : 원본 데이터에 대한 전처리
    - requirements.txt : 프로젝트 실행에 필요한 파이썬 라이브러리 목록과 버전 명시
    - tokenization.py : 텍스트 데이터에 대한 토큰화 진행
    - train.py : 모델 학습 과정을 총괄하는 메인 스크립트
    - utils.py : 여러 파일에서 공통적으로 사용되는 유용한 함수를 모아놓은 모듈