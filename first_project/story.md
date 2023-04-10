# project_story
- 총 인원 : 7명
- 주제 : 금융 데이터 분석(신용등급 정도, 불법 신용카드 사용자, 주식 가격 예측 등)
- Velog(시각화 자료 포함) : [링크](https://velog.io/@xswer19/Financeproject)

## 1일차
- 프로젝트 주제 선정 미팅(화상)
 >1. 팀원 : 6명, 머신러닝 프로젝트 주제 선정 시작
 >2. 주제 후보 : 신용카드 사용 데이터 분석, 감귤 착즙량 예측 모델링, 뉴욕 증시 데이터 분석, 음악 추천 시스템 모델
 >3. 주제 순위 
 >>**1순위** : 신용카드 사용 데이터 분석 
 >>2순위 : 감귤 착즙량 예측 모델링
\ >>3순위 : 음악 추천 시스템 모델
- 계획 수립
>1. 3개의 주제에 대한 데이터셋의 목적 및 컬럼을 분석 후 회의
>2. 후보 주제 중 머신러닝 기법을 사용하기 좋고 실용적인 주제를 투표
>3. 최대 투표 주제 선정할 예정 
<br><br>
## 2일차, 프로젝트 사전 회의
- 프로젝트 주제 확정 미팅(화상)
>1. 대출 연체자 신용등급 판멸 모델링 선정
>2. 차후 더 많은 금융 데이터 분석 예정
- 프로젝트 관련 규칙
1. 사용한 모든 모듈 1행에 모두 from, import
2. 작업에 필요한 중요한 변수 및 함수 기능은 2행에 마크다운으로 주석처리
- 프로젝트 환경 및 작업방식
>1. 오전 10시 30분 미팅 후 작업 시작
>2. 사용 환경 결정 : colab
>3. 코드 공유 방식 : github
- 차후 계획
> 1. 4.3일까지 데이터 컬럼 및 데이터 자체를 분석
> 2. 4.3일부터 데이터 EDA분석 및 분석 시각화
> 3. 4 5일 중간발표까지 EDA작업 완료 예정
- 2일차 분석 내용
>1. 일단 각 컬럼의 분포를 파악함
>2. 범주형 자료의 경우 seaborn의 countplot으로 각 unique값의 분포를 파악하고, 연속형 자료의 경우 seaborn의 kdeplot 그래프를 통해 분포도를 파악
>3. 몇몇의 컬럼 데이터가 음수값으로 표시되어 있어서(데이터 수집 기준 당시 지난 정도) 양수값으로 데이터 가공이 필요해보임 그리고 연, 월, 일 등 날짜 데이터는 가공이 필요해보임
>4. 하나의 컬럼은 unique값이 하나만 존재해서 머신러닝 모델링을 위한 데이터로 부적합하기 drop
- 의문점 
>1. 회귀분석의 경우 $f(x) = y$라는 하나의 회귀직선을 구하는 것이 중요한데, 해당 모델의 성능을 검증하는 지표 중 하나가 RMSE이다. 
>2. 하지만, 다중회귀을 사용할 경우 여러 차원의 값을 분석하고 cost function이 가장 적은 모델을 구하는데 연속형 자료와 범주형 자료가 섞여있어도 괜찮은가?
## 3일차, 프로젝트 시작
- 프로젝트 진행 방향
> 1. 데이터 정리 후 컬럼의 속성 파악
> 2. 시각화 결과 전처리 및 가공이 필요한 데이터 파악
- 진행 상황
> 1. 가장 먼저 데이터 전처리 작업 시작
> 2. 문제점 발견 : 데이터 중 연속형 자료인데, unique값이 두 개인 항목 발견
> 3. 문제점 발견2 : 연속형 자료와 범주형 자료가 섞인 데이터셋으로 다중회귀분석 모델링 시 문제가 있지 않을까 생각
> 4. 문제2의 해결 : 소수의 연속형 자료 컬럼을 등급별로 구분 후 범주형 데이터로 만들고, 랜덤포레스트 및 DesicionTree 등의 방법으로 모델링 구축 계
> 5. 차후 계획 : 데이터 전처리 이후 Label값이 범주형 자료이기 때문에 랜덤 포레스트, 결정나무 3가지 모델을 통해 성능 테스트 
> 6. 컬럼 분석 : 총 수입, 근무 경력, 자녀의 수 3개 컬럼의 데이터 전처리에서 문제가 발생
>> 데이터 한쪽으로 극단적이게 몰려있어서 범주형 자료로 어떻게 분류할지 등급 결정에 문제
> 7. 데이터 전처리 및 EDA 결과 RandomForest의 경우 Scalerling에 영향을 받지 않기에, 모든 자료 형태를 범주형으로 변환하고(numpy.histogram 사용, 도수분표) 모델링 계획
- 진행 결과
> 1. 일단 먼저 데이터 전처리 과정을 끝냈다. Date 컬럼들은 음수를 양수값으로 변경하고,
> 2. 연속형 자료인 수입, 나이, 카드 사용 기간 등을 numpy.histgram 함수를 통해 bins를 나누고 해당 bins를 통해 각 데이터의 등급을 구분해 범주형 자료로 구분했다.
> 3. RandomForest의 경우 Scaler 영향을 받지 않기 때문에 Scaler작업은 하지 않았으며, 이후 데이터 x, y(credit, 라벨값)로 구분하고 Train_split으로 x_train, x_test, y_train, y_test로 구분했다
> 4. 이후 모든 자료를 sklearn의 Labelencorder 기능을 통해 Label작업을 했다.
> 5. RandomForest작업 결과 성능이 65%가 나와 추가적인 작업이 필요할듯하다.
- 3일차 결과
>1. 랜덤포레스트로 accuracy 결과값을 확인한 결과 정확도 65%으로 낮은 성능을 보였다
>2. 하지만, 데이터셋 제공한 대회에서 원하는 출력값을 log_loss로 결과값을 출력해보니 0.95에 가까운 값을 얻었다.
>3. 낮은 값이 나올수록 성능이 높기에 3일차에 데이터의 스케일과 여러 머신러닝 기법을 사용해 성능을 높여야겠다.
>4. 3일차 계획은 일단 principal을 통해 차원을 축소해서 머신러닝을 돌려보고, ligthGBN, X부스트까지 사용해서 
>5. 해당 결과를 github를 통해 공유하고, 내일 미팅으로 의견을 조율 후 EDA 진행 예정이다.
## 4일차, 성능의 향상
- 4일차 시작
>1. 가장 먼저 한 행동은 이전에 했던 데이터셋을 다시 한 번 살펴봤다.
>2. 성능이 낮은 이유가 무엇인지 고민한 결과 임의로 내가 연속형 자료를 등급으로 나누어 범주형으로 만든 것이 잘못이 아닐까 생각이 들고, 다른 방법을 모색했다.
>3. 3가지 방법을 생각했다.
>>1. Pricipal 기법으로 19개의 컬럼의 차원을 $n$으로 축소해서 머신러닝을 사용하기
>>2. 다중요인분석을 사용해서 필요한 컬럼값을 특정하기
>>3. OLS기법으로 연속형 자료와 1~2개 범주형 자료만 뽑아서 분석하기
- 처차한 실패
>1. 3가지 모두 원하던 결과가 나오지는 않았다.
>2. OLS기법의 경우 proba(회귀분석인데 있을 수가 없는데...)가 없어 log_loss를 사용할 수 없었고, 나머지 두 방법의 경우 원하던 값이 나오지 않았다.
>3. 하지만 좋은 성과도 있었다.
- 올라간 성능
>1. OLS기법을 사용하기 위해 연속형 자료와 상관계수가 높은 범주형 자료 하나를 모델링했는데, 그 과정에서 상관계수의 순위 1~3위가 연속형 자료였다.
>2. 나머지 연속형 자료 하나도 상관계수가 높았으며, 범주형 자료하나도 상관계수가 높았다.
>3. 해당 결과를 통해 5개의 컬럼으로 RandomForest를 실행한 결과 accuracy 70%, log_loss 0.7의 값을 얻었다. 어제 얻은 성능보다 올라갔으며 log_loss의 값도(낮을수록 좋음) 낮아졌다.
- 차후 계획
>1. DecisionTree를 통해 성능의 값을 올려봐야겠다. 하지만 트리나무의 경우 과적합 문제가 발생할 수 있기 때문에 전적으로 성능을 믿지 말아야하는 것을 기억하자
>2. 이제 해당 5개의 컬럼값을 통해 여러 시도를 해볼 텐데, 먼저 boost계열의 머신러닝 기법을 사용해야겠다.
>3. 지금 생각해둔 기법은 lightgbm, xgboost, catboost 세 모듈이다.
>4. 만약에 4 방법으로 성능을 테스트해서 결과가 마음에 들지 않는다면, 다시 EDA과정을 시작해서 다른 컬럼값을 뽑고 Scale 과정도 가져보자
## 5일차, 의견 조율
- 화상 미팅
>1. 각자의 의견을 듣고 조율함
>>1. 범주형 자료의 경우 Labelincorder를 사용할지, onehotEncorder를 사용할지 회의 결과 자료의 순서가 데이터에 영향을 안 미치기에 Labelencorder로 사용 결정
>>2. 각자의 모델링 기법을 통해 Log_loss 결과를 확인 결과 MinMaxScaler로 voting기법으로 분석 시 0.74로 가장 낮게 나옴
>>3. 의견 조율 결과 데이터 분석 과정에서 데이터를 어떻게 가공하냐에 따라 성능이 차이가 크기 때문에 각자 데이터 전처리, 분석 과정을 거친 후 성능이 좋은 결과를 베이스로 다시 작업 하기로 결정
>2. 역할 분담
>> 데이터 분석의 효율성을 위한 팀원의 일부는 연속형 데이터를 도수분표를 기준으로 등급별로 나눠서 머신러닝 모델의 성능을 테스트하고, 팀원의 일부는 스케일을 조정하고 연속형 자료를 그대로 사용해서 머신러닝 모델링의 성능을 테스트했다.
- 오늘의 진행 예정
> 1. 일단 Pipeline으로 여러 모델링 기법을 함수로 만든 후 모델링마다 성능을 테스트 하기
> 2. 성능 테스트 후 데이터 전처리 과정 및 분석 재작업 후 모델링 재작업 
- 진행 결과
> 1. 성능 테스트 결과 xgbcBoost, randomForest 두 모델이 좋은 성능을 보여서 2개를 기준으로 GridSerch 및 log_loss값을 만드는 함수 생성
```
def log_loss_gridCv(x_train, y_train, x_test, y_test):
    rf_clf = RandomForestClassifier()
    xg = XGBClassifier()
    
    # RandomFroest
    params = {
    "n_estimators" : [100, 500, 1000,2000],
    "random_state" : [13]}
    clf = GridSearchCV(rf_clf, param_grid=params, cv = 5, scoring="neg_log_loss")
    clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_test)
    print("RandomForest : ",clf.best_estimator_)
    print("RandomForest_result : ",log_loss(y_test, pred))
    
    # xgbcBoost
    xg = XGBClassifier()
    params = {
        "n_estimators" : [100, 500, 1000,2000],
        "max_depth" : [3, 5, 7, 9],
        "random_state" : [13]
    }
    clf = GridSearchCV(xg, param_grid=params, cv=2, scoring="neg_log_loss")
    clf.fit(x_train, y_train)

    pred = clf.predict_proba(x_test)
    print("xgbm_best : ",clf.best_estimator_)
    print("xgbm_result : ",log_loss(y_test, pred))
```
>2. 해당 함수를 통해 데이터 분석 및 전처리 후 성능 테스트 후 결과 확인하고, 분석 작업 반복 예정
>3. randomForest와 xgbcBoost 성능 테스트 결과 0.77과 0.75 출력
>4. 이번에는 범주형 자료를 합쳐서 새로운 컬럼을 만들어서 시도해봐야겠다. -> 차와 부동산 유무를 합치고. 수입형태와 교육형태와 가족형태와 집형태를 합쳐서 새로운 컬럼을 만들어서 성능을 확인해보자
>5. 시각화 자료를 보면서 떠오른 생각이 **여러 컬럼을 하나의 컬럼으로 합쳐서(str형태) 새로운 고유값 컬럼을 만든다면 다른 결과가 나오지 않을까 생각**
>6. 5번의 작업을 위해 여러 컬럼을 조합하는 과정에서 **카드 발급 날짜만 다르고 다른 모든 데이터가 같은(동일인물) index발견**
>7. 계획 새로운 컬럼으로 한 사람의 고유 ID를 식별할 수 있게 몇개의 컬럼을 조합해서 한 사람의 카드 발급 날짜를 하나의 시계열 데이터로 사용할 수도 있지 않을까 생각함.
## 6일차, 데이터 분석 및 모델링의 성능 테스트의 반복
- 화상 회의 결과
>1. 일단 내가 만든 xgbcBoost, RandomFrest GridsearchCV 및 logLoss 성능 테스트 함수를 공유하고, 각자의 데이터 분석을 기반으로 성능을 테스트 후 가장 좋은 성능을 기반으로 다시 파생적으로 분석하기로 결정
>2. 이후 clusterig 작업을 plus해서 log_loss값을 낮추기 위한 작업 예정
>3. 분석 과정에서 연속형 자료의 등급별 구간 도수분표를 통한 머신러닝 모델링의 성능은 모두 좋지 못한 결과를 얻어서 연속형 자료를 그 자체로 쓰거나, 스케일값만 바꾸기로 결정.
- 진행 상황
>1. 일단 나는 라벨값 credit 0, 1, 2 3개를 구분해서 각 라벨값에 따른 시각화를 기반으로 label를 구분할 수 있는 feature을 파악
>2. 하지만 중복값을 제거하고 시각화를 하고, 중복값을 허용해서 시각화를 해도 큰 의미있는 컬럼을 발견하지 못함
>3. 그래서 일단 개인의 고유 번호로 사용할 수 있게 카드 발급 날짜를 제외한 다른 컬럼을 합친 하나의 컬럼을 생성
>4. 이후 각 값들을 LabelEncorder를 진행하고, "income_total", "DAYS_BIRTH", "DAYS_EMPLOYED", "begin_month", "occyp_type","SSN" 6개의 컬럼으로 xgbcboost와 RandomForest로 성능 테스트 시작
>5. 또한, log_loss 성능 테스트 함수에 lgbmboost를 추가해서 3개의 머신러닝 모델 중 가장 최적화된 모델을 찾아 성능 테스트 실행
>6. 데이터를 관찰하면서 가장 많이 생각한 것은 "어떤 데이터를 줘야 컴퓨터가 잘 학습해서 새로운 데이터를 넣었을 때 잘 예측할까?"를 중점으로 생각했을 때
>7. 중복된 데이터에서 한 사람이 여러 카드를 발급받을 경우 카드 발급 개월을 제외한 다른 데이터는 동일하니까 해당 컬럼을 제외하고 고유ID에 해당하는 데이터 컬럼을 주면 컴퓨터가 잘 학습할 수 있지 않을까? 생각해서 SSN(개인 고유 번호) 컬럼을 생성하고 Encoder작업 후 성능 테스트 시작
```
def log_loss_gridCv(x_train, y_train, x_test, y_test):
    rf_clf = RandomForestClassifier()
    xg = XGBClassifier()
    lgbm = LGBMClassifier()


    
    # RandomFroest
    params = {
    "n_estimators" : [1000, 1500, 2000],
    "random_state" : [13]}
    clf = GridSearchCV(rf_clf, param_grid=params, cv = 5, scoring="neg_log_loss")
    clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_test)
    print("RandomForest : ",clf.best_estimator_)
    print("RandomForest_result : ",log_loss(y_test, pred))
    
    pred = clf.predict(x_test)
    print('RandomForest : ', clf.best_estimator_)
    print("RandomForest_result : ", accuracy_score(y_test, pred))
    
    print("=" * 50)
    
    # xgbcBoost
    xg = XGBClassifier()
    params = {
        "n_estimators" : [1000, 1500, 2000],
        "max_depth" : [5, 7, 9],
        "random_state" : [13]
    }
    clf = GridSearchCV(xg, param_grid=params, cv=2, scoring="neg_log_loss")
    clf.fit(x_train, y_train)


    pred = clf.predict_proba(x_test)
    print("xgbm_best : ",clf.best_estimator_)
    print("xgbm_result : ",log_loss(y_test, pred))
    pred = clf.predict(x_test)
    print('xgbm_best : ', clf.best_estimator_)
    print("xgbm_result : ", accuracy_score(y_test, pred))
    
    
    print("=" * 50)
    
    #lgbm
    params = {
    "n_estimators" : [1000, 1500, 2000],
    "random_state" : [13]
    }
    clf = GridSearchCV(lgbm, param_grid=params, cv=2, scoring="neg_log_loss")
    clf.fit(x_train, y_train)


    pred = clf.predict_proba(x_test)
    print("lgbm_best : ",clf.best_estimator_)
    print("lgbm_result : ",log_loss(y_test, pred))
    
    pred = clf.predict(x_test)
    print('lgbm_best : ', clf.best_estimator_)
    print("lgbm_result : ", accuracy_score(y_test, pred))
```
- 진행 결과
>1. 회의해서 principal 기법을 통해 차원을 축소해서 분석할 경우 가장 낮은 log_loss값이 나왔으며, 해당 방법을 통해 정확도를 높일 수 있었다.
>2. 그리고 고유번호 이외에 다른 접근 방법을 고민하는 과정에서 feature특징을 살펴봤다.
>3. occupy_type이 NaN값인 사람의 income_type을 살펴보니 연금 수령자가 대체로 많았지만, 공무원, 자영업자 등 다양한 Nan값인 income_type의 사람이 있었다.
>4. 그래서 두 가지 컬럼을 합쳐서 하나의 수입과 직업의 유형을 살펴보면 NaN값을 하나의 데이터로 사용할 수 있지 않을까 생각했다. 그러니까 NaN 직업이 없을 수도 있지만 해당 직업의 유형이 정확하지 않아 Nan값으로 처리한 것일지도 모른다는 가정을 시작으로 두 가지 컬럼을 합쳤다.
>5. 다른 컬럼으로 일한 개월 수와 카드 발급 개월 수를 빼서 카드 발급 전 일한 개월 수를 구했다. 음수가 나올 경우 카드를 받고 난 후 직장을 얻었기 때문에 신용도가 좋지 못하지 않을까? 라는 직관적 판단을 토대로 가정을 세웠다.
>6. 위 컬럼을 feature로 돌린 결과 최소 log_loss는 0.74가 나왔으며, feature의 중요도는 각 모델링마다 달랐지만, 카드 발급 날짜와 연속형 자료가 대체로 높게 나왔다.
## 7일차, credit_data 마무리와 새로운 데이터 서치
- 화상 회의 결과
>1. 고유 ID값 컬럼을 넣은 결과와 중복값을 제거한 데이터셋 등 두 가지 방법으로 분석을 진행하고, 마무리 후 다른 데이터 유형 머신러닝 진행 예정
>2. 다음 데이터는 머신러능을 통한 이상치 데이터 탐색 혹은 회귀분석을 통한 주가 예측 두 가지 유형의 데이터 중 진행 예정
- 진행 결과
>1. ID값을 통해 예측한 모델의 성능 테스트 결과
```
xgbm_result_loss:  0.8062094451587699
xgbm_result_ac :  0.712880650652544

lgbm_result_loss :  0.7933091701301722
lgbm_result_ac :  0.7062606393039531

log_loss :  0.6684856538442059
ac_score :  0.7359561187819179
```
>2. 중복값을 제거한 후 성능 테스트 결과
```
xgbm_result_loss:  0.7335782866206809
xgbm_result_ac :  0.7662835249042146

lgbm_result_loss :  0.734862324234023
lgbm_result_ac :  0.7701149425287356

log_loss :  0.6175400500281912
ac_score :  0.7659352142110762
```
>3. 두 방법 모두 괜찮은 log_loss의 값을 출력했으며, 중복값을 제거하고 catBoost를 통해 돌린 결과가 0.61값으로 가장 좋은 성능을 나타냈다.
