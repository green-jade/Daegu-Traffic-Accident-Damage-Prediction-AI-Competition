## 대구 교통사고 피해 예측 AI 경진대회

### 경진대회 개요
1. 사고 발생 시간, 공간 등의 정보를 활용하여 사고위험도(ECLO)를 예측하는 AI 알고리즘 개발

        ECLO(Equivalent Casualty Loss Only) : 인명피해 심각도
        - ECLO = 사망자수 * 10 + 중상자수 * 5 + 경상자수 * 3 + 부상자수 * 1
        - 본 대회에서는 사고의 위험도를 인명피해 심각도로 측정
2. 평가 방법
: RMSLE(Root Mean Squared Logarithmic Error) of ECLO


### 참여자
빅데이터 연합동아리 Bitamin 12기 구준회, 김지원, 서은서, 송규헌, 최시훈
<br>

### 결과
- RMSLE : 0.42706
- 리더보드 상위 5%


### 프로젝트 구조
```bash
.root
|   README.md
|   
+---EDA
|       대구동성로_EDA.ipynb
|       
+---Encoding&Scaling
|       endcoding,scaling.py
|       readme.md
|       
+---Feature Engineering
|       add_features.py
|       outliers.py
|       Smote.ipynb
|       smote.py
|       readme.md
|       
\---Model Engineering
        AUTOML_baseline.ipynb
```  

### Modeling 기법
- AutoML : https://github.com/mljar/mljar-supervised/
  - CatBoost, XGB regressor, Light GBM regressor 앙상블
- MIT license

### 선택한 Feature Engineering 방법
1. 시간정보 : 요일, 연, 월, 일, cos_hour, season
2. 공간정보 : 구, 동, 사고발생횟수(동별)
3. 기타 
   - 외부 데이터 : 어린어보호구역개수, 급지구분(주차장), 제한속도(cctv), 횡단보도개수, 보안등개수
   - 파생변수 : 기상상태, 도로형태, 노면상태, 사고유형
