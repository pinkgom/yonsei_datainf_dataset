# DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs

본 프로젝트는 **"DATAINF: EFFICIENTLY ESTIMATING DATA INFLUENCE IN LORA-TUNED LLMS AND DIFFUSION MODELS"** 논문을 기반으로 한 데이터 영향력 추론 실험 및 실제 적용 효과 검증을 위한 **노이즈 주입 데이터셋 생성 도구**입니다.

## 📋 프로젝트 개요

DataInf는 LoRA(Low-Rank Adaptation) 기법을 적용한 대규모 언어 모델에서 데이터의 영향력을 효율적으로 추정하는 방법론을 연구합니다. 본 도구는 **체계적인 노이즈 주입**을 통해 실험용 데이터셋을 생성하고, **6가지 분석 도구**로 데이터 품질을 종합적으로 평가합니다.

### 🎯 핵심 기능
- **3가지 노이즈 전략**: balanced, grammar_heavy, semantic_heavy
- **확실한 노이즈 적용**: 모든 노이즈가 실제로 적용되도록 보장
- **계층적 샘플링**: instruction 길이 기반 균등 샘플링
- **6가지 품질 분석**: 기본 정보부터 종합 분석까지

## 🏗️ 프로젝트 구조

```
datainf-project/
├── src/
│   ├── __init__.py                 # 파이썬 패키지 설정
│   ├── data_loader.py              # Alpaca 데이터셋 로딩 및 캐시 관리
│   ├── noise_injection.py          # 체계적 노이즈 주입 시스템
│   └── analysis.py                 # 6가지 품질 분석 도구
├── data/                           # 데이터셋 저장소 (자동 생성)
│   ├── alpaca_full_dataset.json    # 전체 Alpaca 데이터셋 (52K)
│   ├── alpaca_original_*.json      # 원본 데이터셋
│   ├── alpaca_demo_*.json          # 데모용 노이즈 데이터셋 (500개)
│   ├── alpaca_full_*.json          # 전체 노이즈 데이터셋 (52K)
│   └── experiment_metadata_*.json  # 실험 메타데이터
├── main.py                         # 메인 실행 파일
├── requirements.txt                # 의존성 목록
└── README.md                       # 프로젝트 문서
```

## 🔧 설치 및 환경 설정

### 1. 저장소 클론 및 환경 설정
```bash
git clone https://github.com/pinkgom/yonsei_datainf_dataset.git
cd datainf-project
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

## 🎯 노이즈 주입 방법론

### 📊 3가지 노이즈 전략

#### 1. **Balanced (균형형)** - 추천
```
문법 오류: 40% | 의미적 노이즈: 35% | 품질 저하: 25%
```
- **목적**: 전반적인 데이터 품질 저하 시뮬레이션
- **사용 시기**: 일반적인 DataInf 실험
- **예시**: 
```bash
python main.py --full --noise-ratio 0.2 --strategy balanced
```

#### 2. **Grammar Heavy (문법 중심)**
```
문법 오류: 60% | 의미적 노이즈: 25% | 품질 저하: 15%
```
- **목적**: 문법 오류가 모델에 미치는 영향 분석
- **사용 시기**: 문법 처리 능력 평가
- **예시**:
```bash
python main.py --full --noise-ratio 0.15 --strategy grammar_heavy
```

#### 3. **Semantic Heavy (의미 중심)**
```
문법 오류: 20% | 의미적 노이즈: 60% | 품질 저하: 20%
```
- **목적**: 의미적 일관성이 모델에 미치는 영향 분석
- **사용 시기**: 의미 이해 능력 평가
- **예시**:
```bash
python main.py --full --noise-ratio 0.25 --strategy semantic_heavy
```

### 🔬 노이즈 유형 상세 설명

#### 📝 **문법 노이즈 (Grammar Noise)**

**1. 철자 오류 (Typos)**
```
원본: "The weather is beautiful today."
노이즈: "Teh waether is beatiful todya."
```
- 일반적인 오타 패턴 (the → teh, and → adn)
- 키보드 인접 키 오류 (q → w, p → o)
- 문자 순서 바뀜 (form → from)

**2. 단어 순서 섞기**
```
원본: "Machine learning requires large datasets."
노이즈: "Learning machine requires datasets large."
```

**3. 구두점 오류**
```
원본: "Hello, how are you?"
노이즈: "Hello how are you!!"
```

**4. 문법 일치 오류**
```
원본: "She has three cats."
노이즈: "She have three cats."
```

#### 🧠 **의미적 노이즈 (Semantic Noise)**

**1. 잘못된 맥락**
```
원본: "To make coffee, grind the beans and add hot water."
노이즈: "To make coffee, preheat the oven to 350°F and bake for 30 minutes."
```

**2. 무관한 내용 추가**
```
원본: "Python is a programming language."
노이즈: "Python is a programming language. By the way, did you know cats sleep 16 hours a day?"
```

**3. 주제 이탈**
```
원본: "The capital of France is Paris, located in the north-central part."
노이즈: "The capital of France is Paris. Speaking of pizza, I love pineapple on it."
```

#### 📉 **품질 저하 노이즈 (Quality Noise)**

**1. 불완전한 답변**
```
원본: "Photosynthesis is the process by which plants convert sunlight into energy..."
노이즈: "Photosynthesis is the process..."
```

**2. 중복 내용**
```
원본: "Machine learning is a subset of AI."
노이즈: "Machine learning is a subset of AI. Machine learning is a subset of AI."
```

**3. 무의미한 답변**
```
원본: "The fastest way to sort an array is using quicksort algorithm..."
노이즈: "I don't know."
```

## 🚀 실행 옵션 상세 가이드

### 🎮 **데모 모드** - 첫 사용 추천
```bash
python main.py --demo
```
- **목적**: 빠른 프로토타입 및 기능 테스트
- **샘플 크기**: 500개
- **예상 시간**: 1-2분
- **생성 파일**: 
  - `alpaca_original_demo_500.json` (원본)
  - `alpaca_demo_20percent_balanced_500.json` (균형형 20% 노이즈)
  - `alpaca_demo_20percent_grammar_heavy_500.json` (문법 중심 20% 노이즈)

### 🏭 **전체 데이터 모드** - 실제 실험용

#### 기본 실험
```bash
python main.py --full --noise-ratio 0.2 --strategy balanced
```
- **샘플 크기**: 52,002개 (전체 Alpaca)
- **예상 시간**: 15-20분
- **노이즈 비율**: 20%
- **전략**: 균형형

#### 고급 실험 옵션

**1. 여러 노이즈 비율 한번에**
```bash
python main.py --full --noise-ratios 0.1,0.15,0.2,0.25 --strategy balanced
```
- 4가지 노이즈 비율 (10%, 15%, 20%, 25%)로 각각 데이터셋 생성
- 예상 시간: 60-80분

**2. 모든 전략 테스트**
```bash
python main.py --full --noise-ratio 0.2 --strategy all
```
- balanced, grammar_heavy, semantic_heavy 3가지 전략 모두 적용
- 같은 노이즈 비율로 3가지 다른 데이터셋 생성

**3. 복합 실험**
```bash
python main.py --full --noise-ratios 0.15,0.2 --strategy balanced,grammar_heavy
```
- 2가지 비율 × 2가지 전략 = 4개 데이터셋 생성
- 체계적인 비교 실험 가능

**4. 자동 실행 (확인 건너뛰기)**
```bash
python main.py --full --noise-ratio 0.2 --strategy balanced --yes
```
- `-y` 또는 `--yes` 옵션으로 확인 절차 생략
- 스크립트 자동화에 유용

### 📊 **품질 분석 모드** - 6가지 분석 도구

```bash
python main.py --analysis
```

#### 분석 옵션 상세

**1. 파일 기본 정보**
```
파일명                                            타입         샘플수   노이즈%  전략            크기(MB)
alpaca_original_demo_500.json                   original     500      N/A      N/A             2.1
alpaca_demo_20percent_balanced_500.json         demo         500      20%      balanced        2.2
alpaca_full_20percent_52002.json               experiment   52002    20%      balanced        220.5
```

**2. 데모 파일 상세 분석**
```
🔍 분석 중: alpaca_demo_20percent_balanced_500.json
    📈 분석 결과:
      - 전체 샘플: 500개
      - 노이즈 대상: 100개
      - 실제 변경: 98개 (19.6%)
      - 평균 길이 변화: +15.3 문자
      - 변경 유형:
        • grammar: 45개 (45.9%)
        • semantic: 30개 (30.6%)
        • quality: 23개 (23.5%)
```

**3. 전체 파일 상세 분석** (시간 소요)
- 대용량 파일을 1000개 샘플링으로 빠르게 분석
- 실제 노이즈 적용률 추정

**4. 파일 간 비교 분석**
- 두 파일 직접 비교
- 샘플 수, 평균 길이, 차이점 분석

**5. 노이즈 전략별 효과 분석**
```
전략             변경 샘플    변경 비율    평균 길이 변화
balanced         98          19.6%        +15.3
grammar_heavy    102         20.4%        +8.7
semantic_heavy   95          19.0%        +22.1
```

**6. 전체 종합 분석**
- 모든 분석을 순차적으로 실행
- **DataInf 실험 준비도 자동 체크** ⭐
```
🎯 DataInf 실험 준비도:
  - 원본 데이터: ✅
  - 노이즈 데이터: ✅  
  - 다양한 노이즈 비율: ✅
  📈 준비도: 완벽 (3/3) - DataInf 실험 진행 가능!
```

### 🗂️ **캐시 관리 모드**
```bash
python main.py --cache
```
- 캐시 정보 확인
- 캐시 삭제 (재다운로드 필요시)
- 강제 재다운로드

## 📊 DataInf 실험 시나리오

### 🎯 **표준 실험 절차**

#### 1단계: 데모로 기능 확인
```bash
# 1. 데모 데이터 생성
python main.py --demo

# 2. 분석으로 결과 확인
python main.py --analysis
# → 옵션 2 선택 (데모 파일 상세 분석)
```

#### 2단계: 실험용 데이터셋 생성
```bash
# 기본 실험 (20% 노이즈)
python main.py --full --noise-ratio 0.2 --strategy balanced --yes

# 비교 실험 (15% 노이즈)
python main.py --full --noise-ratio 0.15 --strategy balanced --yes
```

#### 3단계: 품질 검증
```bash
python main.py --analysis
# → 옵션 6 선택 (전체 종합 분석)
```

### 🔬 **고급 실험 시나리오**

#### 시나리오 1: 노이즈 비율 효과 연구
```bash
# 다양한 노이즈 비율로 데이터셋 생성
python main.py --full --noise-ratios 0.05,0.1,0.15,0.2,0.25,0.3 --strategy balanced --yes

# 결과 분석
python main.py --analysis  # → 옵션 3 (전체 파일 상세 분석)
```

#### 시나리오 2: 노이즈 전략 비교 연구
```bash
# 같은 비율, 다른 전략으로 데이터셋 생성
python main.py --full --noise-ratio 0.2 --strategy all --yes

# 전략별 효과 분석
python main.py --analysis  # → 옵션 5 (전략별 효과 분석)
```

#### 시나리오 3: 복합 비교 연구
```bash
# 2×3 실험 설계 (2가지 비율, 3가지 전략)
python main.py --full --noise-ratios 0.15,0.25 --strategy all --yes

# 종합 분석
python main.py --analysis  # → 옵션 6 (전체 종합 분석)
```

## 📚 참고 자료

### 📖 **관련 논문**
- [DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs](https://openreview.net/pdf?id=9m02jb92Wz)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### 🔗 **관련 리소스**
- [Stanford Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [DataInf GitHub Repository](https://github.com/ykwon0407/DataInf)

### 📊 **실험 레퍼런스**
- [Stanford CS224N Projects](https://web.stanford.edu/class/cs224n/project.html)
- [CS231n Reports](http://cs231n.stanford.edu/2017/reports.html)

## 📄 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다. 연구 및 교육 목적으로 자유롭게 사용하실 수 있습니다.

