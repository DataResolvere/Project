| 코드 | 내용 | 링크 |
|------|-----|------|
|   DataPreprocessing   |  1. 서울시 공공도서관의 도서관 번호 출력  2. 서울시 공공 도서관의 대출 기록과 도서 정보를 SQL에서 Join   |   [링크](https://github.com/DataResolvere/Project/blob/main/Culture_book/DataPreprocessing.ipynb)   |
|   DataPreprocessing_nltk   |  도서 제목에 대한 유사도 분석을 위해 특수문자, 외국어 전처리 작업   |   [링크](https://github.com/DataResolvere/Project/blob/main/Culture_book/DataPreprocessing_nltk.ipynb)   |
|   nltk_test   |  스마트도서관 도서 제목을 기준으로 공공 도서관의 도서 제목과 자연어 유사도 분석 및 유사도 평균 결과  |   [링크](https://github.com/DataResolvere/Project/blob/main/Culture_book/nltk_test.ipynb)   |
# 프로젝트 목표
- 비치 도서 추천 시스템을 통한 스마트도서관의 커스텀마이징 및 활성도 증가
# 프로젝트 내용
- 스마트도서관 대출기록, 서울시 332개의 공공도서관 리스트, 332개의 공공도서관의 2022년 대출기록 데이터 등 3개의 데이터 수집
- 공공도서관의 대출기록에서 sql를 통한 데이터 전처리 작업을 통해 KDC 구분, 책제목별 대출횟수 등 추출
- Convetorized, KoNLTK 및 코사인 유사도 분석을 통해 책제목의 유사도 분석
# 프로젝트 결론
- 스마트도서관의 대출기록 데이터 중 양천구 기준 연평균 유사도 성능이 0.67나옴
- 책제목의 경우 유사도가 너무 높을 경우 같은 책이기에 60~70%로 적당하다는 결론 도출
