# SemiNER
+ 개체명 인식기
  - 국립 국어원의 NER 데이터셋을 이용한 음절기반으로 태깅

## 데이터 전처리
```
python token_set.py
```
+ Output 
  - tf 토크나이저 pickle file
  - data폴더 하위에 학습용 numpy 파일 생성

## NER 학습
```
python train.py
```

## 모델 평가
1. 루트 폴더에 검증데이터를 둘것 (파일명은 answer.txt 로)
2. test 데이터 추론 시작
```
python test.py
```
3. 모델 성능 평가 시작
```
python accuracy.py
```

## 사용자 입력 데이터 개체명 인식 결과 확인
```
python inference.py
```
