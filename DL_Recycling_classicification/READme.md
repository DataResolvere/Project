| 코드 | 내용 | 링크  |
|------|-----|---|
|   Project_DataPreprocessing   |  37만장의 사진을 OpenCV를 통해 전처리 및 Pickle타입으로 공유   | [링크](https://github.com/DataResolvere/Project/blob/main/DL_Recycling_classicification/Project_DataPreprocessing.ipynb)  |
|   Project_Model_train_and_Validation   |   Resnet 모델 구축 및 34개 모델 중 앙상블 기법으로 성능 테스트  |  [링크](https://github.com/DataResolvere/Project/blob/main/DL_Recycling_classicification/Project_Model_train_and_Validation.ipynb) |
# 프로젝트 목표
- ResNet모델을 통해 재활용 쓰레기를 9개의 범주별로 분류하는 **AI모델 구현 및 전이학습을 통한 성능 개선 검증**
# 프로젝트 내용
1. 약 100GB 이미지 데이터를 Resize 후 Pickle파일로 변환해서 팀원간 데이터 공유
2. 37만장의 사진을 9:1로 Train, Test로 구분 후 학습 및 검증
3. 메모리 한계로 인해 ResNet모델 구축 후 34만장의 Train사진을 10,000장씩 전이학습 후 각 장수마다 성능 검증
4. 일상 생활의 재활용 쓰레기 데이터 수집 후 AI모델의 성능 테스트
# 프로젝트 결론
- Test_Data에 대한 전이학습 모델 테스트 결과 완벽하지는 않으나 어느정도 선형적인 성능 개선 보임
- 최고 성능 모델을 분석 및 학습 단계에서 구축한 TOP4 모델 4개를 통해 앙상블 시도
- 최종저긍로 약 3만 7천장의 이미지 데이터를 대상으로 80% 성능 결과
