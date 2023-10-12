| 코드 | 내용 | 링크  |
|------|-----|---|
|   Crime_story   |  기후 및 범죄 데이터를 바탕으로 범죄유형 예측 AI 모델 구축을 위한 전처리 및 분석   | [링크](https://github.com/DataResolvere/Project/blob/main/Dacon_Crime/Crime_story.ipynb)  |
|   Crime_finished   |   분석 및 전처리를 기반으로 최종 AI 모델 결과물  |  [링크](https://github.com/DataResolvere/Project/blob/main/Dacon_Crime/Crime_finished.ipynb) |
# 대회 목표
- 범죄 발생 관련 및 기후 데이터를 통해 3개의(절도, 상해, 강도)범죄 유형을 예측하는 AI모델 구현
# 주요 분석 및 기법
1. Cat_boost 알고리즘을 통해 연속형 자료와 범주형 자료 AI모델 예측 시 범주형 자료의 가중치 높임
2. AI 모델의 Feature별 중요도 분석을 통해 Feature의 군집화를 통해 차원을 축소 및 성능 개선
3. 중복 데이터 및 이상치 데이터 제거
# 대회 결과
- 목표 데이터를 예측한 결과를 제출한 결과 marco f1-score 0.51282성능이 나오며, 대회 1등과 결과와 0.01577869의 차이가 있음
