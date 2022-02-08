# 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회

### Private score 7위 0.953044
- 주최 : LG AI Research
- 주관 : 데이콘
- 주제 : "작물 환경 데이터"와 "작물 병해 이미지"를 이용해 "작물의 종류", "병해의 종류", "병해의 진행 정도"를 진단하는 AI 모델 개발
- https://dacon.io/competitions/official/235870/overview/description

## Development environment. 
os : Windows 10 Pro

##  Library version
- numpy==1.19.5
- opencv-python==4.5.4-dev
- tqdm==4.43.0
- scikit-learn==0.24.0
- torch==1.10.0.dev20210803+cu111
- torchvision==0.11.0.dev20210803+cu111
- albumentations==0.5.2
- timm==0.4.12

## Model architecture
<img src =https://user-images.githubusercontent.com/44361199/152677855-dd5b40a8-0175-4804-bce2-e6c699c61186.png    style="width:50%">

## Key augmentation
<img src =https://user-images.githubusercontent.com/44361199/152676731-c8a645e1-3ca5-4284-b7d1-e90b9d5d74c2.png    style="width:30%">

## Directory tree

```
root
│───Dacon_LG.ipynb
│───csv_minmax.pickle
│───utils.py
└───data
│   │───train
│   │   │───10027
│   │   │   │───10027.csv
│   │   │   │───10027.jpg
│   │   │   └───10027.json
│   │   │───10027_focus
│   │   │   │───10027_focus.csv
│   │   │   │───10027_focus.jpg
│   │   │   └───10027_focus.json
│   │   │   ...
│   │
│   └───test
│   │   │───10000
│   │   │   │───10000.csv
│   │   │   └───10000.jpg
│   │   │───10001
│   │   │   │───10001.csv
│   │   │   └───10001.jpg
│       │   ...
│   
└───sample_submission.csv
```


## Training

- Dacon_LG.ipynb 파일이 존재하는 경로에 "data" 폴더와 utils.py파일이 존재 해야한다.
- Dacon_LG.ipynb 파일의 Model Training까지 수행

## Evaluation
- Dacon_LG.ipynb 파일에서 Model Training을 제외한 나머지 코드 수행
