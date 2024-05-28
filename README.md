# Deep_Knowledge_Tracing

## **Abstract**

- Deep Knowledge Tracing(이하 DKT)란 교육기관에서 시험을 실시하고 성적에 따라 얼마만큼 아는지 평가하는 할 때 개개인에 맞춤화된 피드백을 받기가 어려운 문제점을 해결하는 방법이다.
- DKT를 활용하면 우리는 학생 개개인에게 수학의 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능 따라서 DKT는 교육 AI의 추천이라고 불린다. DKT는 맞춤화된 교육을 제공하기 위해 아주 중요한 역할을 맡는다.
- 필자는 이러한 인공지능 모델을 설계하기 위해 여러 종류의 모델(lightgbm, catboost, lightgcn, SASRec)을 사용하는 Multi model 아키텍처를 설계하였다.
- 최종적으로 public / private 에서 모두 3등을 기록하며 Public AUC기준 0.8255를 달성하였다.

## Introduction
<img width="700" alt="Introduce" src="https://github.com/dohyun96305/Level2-DKT/assets/75681704/3f89f50e-f536-442b-8656-7cb165bcd088">

- DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론  
- 시험에 대해 과목을 얼마만큼 이해하고 있는지 측정 및 활용하여 아직 풀지 않은 미래의 문제에 대한 정답 여부를 확인할 수 있다.  
- DKT를 활용하면 우리는 학생 개개인에게 과목에 대한 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능하다.  
- 이번 대회에서는 Iscream 데이터셋을 이용, 모델 구축 및 평가를 진행하였다.  
- 또한 이번 대회는 학생 개개인의 이해도를 나타내는 지식 상태가 아닌 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중한다.  

## Experiment Setting

- HW : V100 GPU  
- Tool : Visual Studio, Wandb  
- SW : Python 3.10.13, torch 2.1.0, transformers 4.35.2, numpy 1.26.0, pandas 2.1.3, lightgbm 3.2.1  

## Dataset

- Input : 총 7,442명의 사용자들과 총 9,454개의 고유 문항간의 Interaction  
    - userID ( 사용자 ID )  
    - assessmentItemID ( 문제 번호 )  
    - testId ( 시험지 번호 )  
    - Timestamp ( 문제를 풀기 시작한 시간 )  
    - KnowledgeTag ( 지식 유형 )  
    - answerCode (정답 여부)  
- output : test_data 사용자들의 마지막 문제의 정답 여부(0 또는 1)  

## Team members

1. 김진용 : LightGCN, UltraGCN 구현  
2. 박치언 : XLNet 모델링 및 구현, SASRec 모델 참조  
3. 배홍섭 : [Team Leader] Feature Engineering 및 Lightgbm 고도화  
4. 안제준 : Catboost, BERT4REC 구현  
5. 윤도현 : 데이터 EDA, TabNet, LastQuery 구현  

## Schedule

- competition : 2024-01-03 ~ 2024-01-25

<img width="700" alt="Schedule" src="https://github.com/dohyun96305/Level2-DKT/assets/75681704/e80a26b1-54f0-4bdb-aa55-1fe6eb1a586d">


1. 프로젝트 개발환경 구축 (Server, Github, Google Drive, WandB)
2. EDA를 통해 데이터 분포 파악 및 Feature Engineering에 필요한 Insight 도출
3. 베이스라인 코드에 대한 이해를 바탕으로 모델 개선점 파악
4. Feature Engineering을 통해 데이터 설명력 증진
5. 리서치를 바탕으로 Baseline보다 뛰어난 성능의 모델 구현 및 Fine-Tuning
6. Post-Processing 및 Public Score을 기준으로 고성능 모델 간 Ensemble 진행
7. 최종 제출

## Feature Engineering

- 데이터 전처리 및 EDA
    - test data 전체 372, 정답 맞춘 사람 176명 public acc:0.4731
    - train set에 0과1 비율을 872368:826794, 즉 1 비율이 0.4865로 조정

<img width="700" alt="EDA_1" src="https://github.com/dohyun96305/Level2-DKT/assets/75681704/4244110e-400b-41b9-b93b-0bcc53ad4757">
 
    
- Users Feature Engineering
    - 유저별 정답률, 푼 문제 수, 정답 맞춘 횟수
    - 유저별 평균(중앙값) 소요 시간
    - 유저별 푼 문제 수
    - 유저별 누적 푼 문제수, 누적 맞춘 문제 갯수, 누적 정답률, 누적 푼 문제시간
- Time Feature Engineering
    - 각 문제 푸는 시간
    - 시간 관련 feature 추가(hour, weekofyear)
    - 대분류별 누적 풀린 횟수, 대분류별 누적 정답수, 대분류별 누적 정답률, 누적 풀이 시간
    - 태그별 모든 문제, 맞춘 문제, 틀린 문제별 풀이 시간의 평균
    - 문항별 모든 문제, 맞춘 문제, 틀린 문제별 풀이 시간의 평균
    - 문제 번호별 모든 문제, 맞춘 문제, 틀린 문제별 풀이 시간의 평균
    - 이 전에 정답을 맞췄는지로 시간적 요소 반영
- Test Feature Engineering
    - 시험지별 푼 문제 개수, 푼 사용자 수
    - 시험지 별 문제 수와 태그 수
    - 시험지 별 안 푼 문제 개수, 문제를 푼 비율
    - 시험지별 정답 평균, 개수, 분산, 표준편차
    - 문항별 정답 평균, 개수, 분산, 표준편차
    - 문제 번호별 정답 평균, 개수, 분산, 표준편차
- Tag Feature Engineering
    - 태그별 정답 평균, 개수, 분산, 표준편차
    - 태그별 모든 문제, 맞춘 문제, 틀린 문제별 풀이 시간의 평균
    
- 모델 선정 및 분석 : public score기준 각 데이터에서 좋은 성능을 보이는 모델을 선정
    - LightGBM : 0.8255
    - GCN : 0.7888
    - XLNet: 0.7308
    - Catboostregressor : 0.8057
    - TabNet : 0.7732
    - SASRec: 0.8099

## Result(public 3st, private 3st)

- Ensemble 구현
    - Ensemble 진행할 모델 선정
        - Public Score 0.8 이상 Model 4개 선정 (lightGBM, LightGCN, UltraGCN, SASRec)
    - Ouput 값에 대해 Model 다른 가중치를 부여, 가중 평균을 통해 최종 Output 도출
        - LightGBM * 0.7 + Catboost * 0.1 + SASRec * 0.1 + UltraGCN * 0.05 + LightGCN * 0.05
- 모델 평가 및 개선
    - 모델 평가 방식 : **AUROC**, ACC
- 시연 결과
    - Ensemble을 통해 전반적으로 Public Score가 상승하였으며 단일 모델보다 Public Score가 떨어진 경우에도 Private Score가 상승하는 것을 대회 종료 후 확인할 수 있었음.
 
**public : 0.8255**

<img width="700" alt="Score_public" src="https://github.com/dohyun96305/Level2-DKT/assets/75681704/f2df3094-dadd-4e49-886f-b3b51ad0dd91"> <br/><br/>


**private : 0.8523**

<img width="700" alt="Score_private" src="https://github.com/dohyun96305/Level2-DKT/assets/75681704/18fff6d1-2f7e-4dc3-aeac-d73556372014">

