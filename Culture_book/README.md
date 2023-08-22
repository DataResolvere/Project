# 프로젝트 목표
- 비치 도서 추천 시스템을 통한 스마트도서관의 커스텀마이징 및 활성도 증가
# 프로젝트 내용
- 스마트도서관 대출기록, 서울시 332개의 공공도서관 리스트, 332개의 공공도서관의 2022년 대출기록 데이터 등 3개의 데이터 수집
- 공공도서관의 대출기록에서 sql를 통한 데이터 전처리 작업을 통해 KDC 구분, 책제목별 대출횟수 등 추출
- Convetorized, KoNLTK 및 코사인 유사도 분석을 통해 책제목의 유사도 분석
# 프로젝트 결론
- 스마트도서관의 대출기록 데이터 중 양천구 기준 연평균 유사도 성능이 0.67나옴
- 책제목의 경우 유사도가 너무 높을 경우 같은 책이기에 60~70%로 적당하다는 결론 도출