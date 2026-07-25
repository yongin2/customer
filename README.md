# 고객 관리 AI 프로그램 (Streamlit)

Orange3에서 학습한 VIP 여부 / 선호 카테고리 / 할인 민감성 모델을 활용한 고객 분석 웹앱입니다.

## 배포 방법 (Streamlit Community Cloud, 완전 무료)

### 1단계: GitHub에 코드 올리기
1. https://github.com/new 에서 새 저장소 생성 (Public 또는 Private 모두 가능)
2. 이 폴더 안의 파일 전부를 저장소에 업로드
   - `app.py`
   - `logic.py`
   - `requirements.txt`
   - `packages.txt`
   - `models/vip_model_rf.pkcls`
   - `models/interest_model_nb.pkcls`
   - `models/discount_model_lr.pkcls`

   웹에서 직접 업로드하려면: 저장소 페이지 → **Add file → Upload files** → 이 폴더의 파일/폴더를 통째로 드래그

### 2단계: Streamlit Community Cloud에서 배포
1. https://share.streamlit.io 접속 → GitHub 계정으로 로그인
2. **"New app"** 클릭
3. Repository: 방금 만든 저장소 선택
4. Branch: `main` (기본값)
5. Main file path: `app.py`
6. **"Deploy"** 클릭

### 3단계: 완성
몇 분 빌드 후 아래와 같은 형태의 링크가 생성됩니다:
```
https://<앱이름>-<임의문자열>.streamlit.app
```
이 링크를 공유하면 누구나 브라우저에서 바로 접속해서 사용할 수 있어요.

## 로컬에서 먼저 테스트하려면
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 파일 구성
- `app.py` : Streamlit UI (탭 3개: 고객 분석 / 여러 고객 비교 / 재유치 대상)
- `logic.py` : 모델 로드 + 예측 + 추천/우선순위 로직 (UI와 분리된 핵심 로직)
- `models/` : 오렌지에서 학습한 pkcls 모델 3개
