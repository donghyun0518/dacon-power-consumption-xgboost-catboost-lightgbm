# <h1 style="text-align: center;">⚡2025 전력사용량 예측 AI 경진대회</h1>
<hr>
<p style="text-align: center;">
    <a href="https://github.com/donghyun0518/power-usage-prediction/blob/main/presentation.pdf" target="_blank">
        <img src="https://github.com/donghyun0518/power-usage-prediction/blob/main/project_overview.png" alt="Project Cover" style="width: 1000px; border: 1px solid #c9d1d9; border-radius: 8px;">
    </a>
</p>

[프로젝트 발표 자료](https://github.com/donghyun0518/dacon-power-consumption-xgboost-catboost-lightgbm/blob/main/%EC%A0%84%EB%A0%A5.pdf)

## 🔍 프로젝트 개요
- **목적**: 안정적이고 효율적인 에너지 공급을 위한 전력 사용량 예측 시스템 개발
- **주제**: 건물의 전력사용량 데이터와 기상데이터를 활용한 전력 사용량 예측 AI 모델 개발
- **기간**: 2025.07.14 ~ 2025.08.25 (약 6주간)
- **팀 구성**: 2인
- **평가지표**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **성과**: Public Score 5.16585, Private Score 5.82404 달성 (상위권 입상, 상위 1.9%, 934팀 중 18위)

## 📊 데이터셋 구성

### 입력 데이터
- **건물 특성**: 건물 유형, 연면적(m²), 냉방면적(m²), 태양광용량(kW), ESS저장용량(kWh), PCS용량(kW)
- **기상 정보**: 기온(°C), 강수량(mm), 풍속(m/s), 습도(%), 일조(hr), 일사량(MJ/m²)

### 데이터 파일
| 파일명 | 내용 | 데이터 수 |
|--------|------|-----------|
| train.csv | 2024.06.01~08.24 100개 건물의 기상 데이터, 실제 전력사용량 | 204,000 |
| building_info.csv | 건물 번호, 건물 유형별 연면적, 냉방면적, 태양광 ESS, PCS 용량 | 100 |
| test.csv | 2024.08.25~08.31 100개 건물의 기상 데이터 | 16,800 |
| sample_submission.csv | 제출 양식(ID: 건물번호 + 시간 / answer: 예측값) | 16,800 |

## ⚙️ 주요 수행 과정

### 1. 문제 정의 및 전략 수립
- **배경**: 기후 변화와 에너지 전환 정책 가속화에 따른 수요 예측 기반 에너지 관리 역량의 중요성 증대
- **핵심 전략**:
  - Feature Engineering: 기상학적, 공학적, 시계열 피쳐 생성
  - Modeling: XGBoost, CatBoost, LightGBM
  - Ensemble: 시드, 12-fold ensemble, 모델별 최종 파일 Ensemble
  - 건물별 개별 모델링 (100개 건물)
  - 유형별 개별 모델

### 2. 탐색적 데이터 분석 (EDA)
- **주요 발견사항**:
  - 유형별 전력 소비량 패턴의 차이 존재
  - 여름철 냉방 수요로 인한 전반적인 우상향 그래프
  - 일부 건물에서 이상치 및 불규칙 패턴 존재
  - 소비 패턴의 차이로 인한 유형별, 건물별 모델링 전략 필요성 확인

### 3. 데이터 전처리
- **향상된 스플라인 보간 (Enhanced Spline Interpolation)**
  - 공식: S(w) = a_i + b_i(w- w_i) + c_i(w- w_i)² + d_i(w- w_i)³
  - 0값 보간: 전력소비량이 0인 값들을 패턴 기반으로 보간
  - 이상치 보간: 특정 건물별 이상치 패턴을 정의하여 보간
  - 패턴 매칭: CubicSpline을 사용한 주차 단위 패턴 매칭

### 4. 피쳐 엔지니어링
- **시간 순환 피쳐**: sin/cos 인코딩 (시간, 월-일, 월, 요일)
- **특별 패턴**: 건물별 휴무 여부, 날씨 영향, 월내 격주 일요일 패턴
- **계절 패턴**: 여름철 sin/cos 패턴
- **온도 통계**: 3시간 간격 일평균 온도, 일교차
- **복합 지수**: CDH(냉방도), THI(온도-습도 지수), WCT(체감온도)
- **전력 통계**: 요일-시간별, 휴일-시간별, 시간별 과거 평균/표준편차
- **환경 조건**: 고온다습 조건

#### 건물별 특별 휴무 패턴
- 매주 일요일 휴무: building_18
- 격주 일요일 휴무: building_27, 40, 54, 59, 63
- 격주 월요일 휴무: building_34
- 특정 날짜 휴무: building_19, 45, 79, 95
- 연중무휴: building_73, 88
- 복합 휴무: building_74 (격주 일요일 + 추가 월요일)

### 5. 모델 훈련

#### 5.1 XGBoost 모델
- **모델 파라미터**:
  ```python
  learning_rate: 0.1
  n_estimators: 1000
  max_depth: 5
  subsample: 0.9
  colsample_bytree: 0.8
  min_child_weight: 6
  reg_alpha: 1
  reg_lambda: 1
  ```
- **2단계 앙상블 전략**:
  - 1단계: 각 폴드별 시드 앙상블 (5개 시드: 2025, 42, 123, 777, 999)
  - 2단계: 12-Fold 교차검증 (7일 단위 역순)
- **성능**:
  - 건물별 모델 LB Score 5.39160
  - 유형별 모델 LB Score 5.7983

#### 5.2 LightGBM 모델
- **모델 파라미터**:
  ```python
  learning_rate: 0.05
  n_estimators: 2000
  max_depth: 5
  num_leaves: 50
  subsample: 0.9
  colsample_bytree: 0.8
  min_child_samples: 15
  reg_alpha: 1
  reg_lambda: 1
  ```
- **성능**:
  - 건물별 모델 LB Score 5.39090
  - 유형별 모델 LB Score 5.81680

#### 5.3 CatBoost 모델
- **모델 파라미터**:
  ```python
  learning_rate: 0.1
  iterations: 1000
  depth: 5
  subsample: 0.9
  colsample_bylevel: 0.8
  ```
- **로그 변환**: log(y + 1e-6), 음수 방지 적용
- **성능**:
  - 건물별 모델 LB Score 5.29539
  - 유형별 모델 LB Score 5.78100

### 6. 앙상블 전략

#### 6.1 Comparison 모델
- 각 건물별 XGBoost, LightGBM, CatBoost 중 SMAPE가 가장 낮은 모델 선택
- **성능**: LB Score 5.22

#### 6.2 Stacking 모델
- **Base Models**: XGBoost, LightGBM, CatBoost
- **Meta Model**: Ridge Regression
- Base Model들의 예측을 선형 결합하면서 정규화를 통해 안정성 확보
- **성능**: LB Score 5.2633

### 7. 최종 앙상블
리더보드 기준으로 앙상블 후보를 선정 후 가중평균 앙상블을 통해 최종 제출물 생성

| 앙상블 목록 | LB Score |
|-------------|----------|
| Comparison1 | 5.56768 |
| Comparison2 | 5.22 |
| Stacking | 5.2633 |
| Catboost 알파값 앙상블 | 5.3765 |
| Catboost 단일 모델 | 5.29539 |
| LightGBM 단일 모델 | 5.39090 |
| XGBoost 단일 모델 | 5.39160 |
| XGBoost 유형별 모델 | 5.7983 |
| Catboost 유형별 모델 | 5.78100 |
| LightGBM 유형별 모델 | 5.81680 |

## 🛠️ 기술 스택
- **언어**: Python
- **라이브러리**: 
  - 머신러닝: scikit-learn, XGBoost, LightGBM, CatBoost
  - 데이터 처리: pandas, numpy
  - 시각화: matplotlib, seaborn
  - 통계: scipy
- **개발 환경**: Jupyter Notebook, Google Colab

## 📈 모델 성능 평가
- **최종 성과**: Public Score 5.16585, Private Score 5.82404 (SMAPE 기준)
- **검증 방식**: 시계열 12-Fold 교차검증
- **특징**:
  - 건물별 개별 모델링으로 각 건물의 특성 반영
  - 다양한 부스팅 모델의 앙상블로 안정성 확보
  - 과소추정 문제 해결을 위한 가중치 적용
  - 시계열 특성을 고려한 피쳐 엔지니어링

## 🔮 기대효과
- **에너지 효율성 향상**: 정확한 전력 수요 예측을 통한 효율적인 에너지 공급
- **비용 절감**: 과잉 발전 및 부족 발전으로 인한 비용 최소화
- **그리드 안정성**: 전력망의 안정적 운영 지원
- **탄소 중립**: 효율적인 에너지 관리를 통한 탄소 배출 감소
- **산업 응용**: 다양한 건물 유형에 대한 맞춤형 에너지 관리 시스템 구축
