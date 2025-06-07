# DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs

본 프로젝트는 **"DATAINF: EFFICIENTLY ESTIMATING DATA INFLUENCE IN LORA-TUNED LLMS AND DIFFUSION MODELS"** 논문을 기반으로 한 데이터 영향력 추론 실험 및 실제 적용 효과 검증을 위한 **다중 데이터셋 노이즈 주입 도구**입니다.

## 📋 프로젝트 개요

DataInf는 LoRA(Low-Rank Adaptation) 기법을 적용한 대규모 언어 모델에서 데이터의 영향력을 효율적으로 추정하는 방법론을 연구합니다. 본 도구는 **다양한 NLP 태스크 데이터셋**에 **체계적인 노이즈 주입**을 통해 실험용 데이터셋을 생성하고, **6가지 분석 도구**로 데이터 품질을 종합적으로 평가합니다.

### 🎯 핵심 기능
- **4개 데이터셋 지원**: Alpaca (instruction-following), GSM8K (math), SST-2 (sentiment), MRPC (paraphrase)
- **2가지 노이즈 타입**: 텍스트 노이즈 vs 라벨 플리핑 (DataInf 논문 표준)
- **라벨 보존 노이즈**: 분류 태스크에서 정답 라벨은 보존하면서 입력 텍스트만 노이즈 주입
- **라벨 플리핑 모드**: SST-2, MRPC에서 0↔1 라벨 플리핑 지원
- **3가지 노이즈 전략**: balanced, grammar_heavy, semantic_heavy
- **확실한 노이즈 적용**: 모든 노이즈가 실제로 적용되도록 보장
- **데이터셋별 계층적 샘플링**: instruction 길이 또는 라벨 균형 기반 샘플링
- **6가지 품질 분석**: 기본 정보부터 종합 분석까지

## 🏗️ 프로젝트 구조

```
datainf-project/
├── src/
│   ├── __init__.py                 # 파이썬 패키지 설정
│   ├── data_loader.py              # 다중 데이터셋 로딩 및 캐시 관리
│   ├── noise_injection.py          # 라벨 보존 노이즈 주입 시스템
│   └── analysis.py                 # 6가지 품질 분석 도구
├── data/                           # 데이터셋 저장소 (자동 생성)
│   ├── alpaca_full_dataset.json    # 전체 Alpaca 데이터셋 (52K)
│   ├── gsm8k_full_dataset.json     # 전체 GSM8K 데이터셋 (7.5K)
│   ├── sst2_full_dataset.json      # 전체 SST-2 데이터셋 (67K)
│   ├── mrpc_full_dataset.json      # 전체 MRPC 데이터셋 (3.7K)
│   ├── *_original_*.json           # 원본 데이터셋들
│   ├── *_demo_*.json               # 데모용 노이즈 데이터셋 (500개)
│   ├── *_full_*.json               # 전체 노이즈 데이터셋
│   └── experiment_metadata_*.json  # 실험 메타데이터
├── docs/                           # 논문 및 문서 (gitignore)
│   ├── 6868_DataInf_Efficiently_Estim.pdf
│   └── DataInf 기반 효율적 영향력 추론 실험 및 실제 적용 효과 검증.pdf
├── main.py                         # 메인 실행 파일
├── requirements.txt                # 의존성 목록
└── README.md                       # 프로젝트 문서
```

## 🗂️ 지원 데이터셋

| 데이터셋 | 태스크 | 크기 | 텍스트 컬럼 | 라벨 컬럼 | 라벨 보존 | 라벨 플리핑 | 설명 |
|---------|-------|------|-----------|-----------|-----------|-------------|------|
| **Alpaca** | Instruction-following | ~52K | instruction, output | - | ❌ | ❌ | Stanford에서 제작한 instruction-following 데이터셋 |
| **GSM8K** | Math reasoning | ~7.5K | question | answer | ✅ | ❌ | 초등학교 수준의 수학 문제 해결 데이터셋 |
| **SST-2** | Sentiment classification | ~67K | sentence | label | ✅ | ✅ | 영화 리뷰 감정 분류 데이터셋 (긍정/부정) |
| **MRPC** | Paraphrase detection | ~3.7K | sentence1, sentence2 | label | ✅ | ✅ | 문장 쌍의 의미적 유사성 판정 데이터셋 |

### 📋 **데이터셋 상세 정보**

#### 🎓 **Stanford Alpaca**
- **목적**: Instruction-following 능력 학습 및 평가
- **특징**: Self-instruct 방법으로 생성된 고품질 instruction 데이터
- **용도**: 범용 언어모델의 지시사항 수행 능력 연구
- **노이즈 적용**: instruction과 output 모두에 텍스트 노이즈 적용

#### 🧮 **GSM8K (Grade School Math 8K)**
- **목적**: 수학적 추론 능력 평가
- **특징**: 초등학교 수준의 다단계 추론이 필요한 수학 문제
- **용도**: 모델의 논리적 사고 및 계산 능력 측정
- **노이즈 적용**: question에만 텍스트 노이즈, answer는 정답 보존

#### 😊 **SST-2 (Stanford Sentiment Treebank)**
- **목적**: 영화 리뷰 감정 분류 (긍정/부정)
- **특징**: 세밀한 감정 어노테이션이 포함된 문장 단위 분류
- **용도**: 감정 분석 모델의 성능 평가
- **노이즈 적용**: sentence 텍스트 노이즈 또는 label 플리핑 (0↔1)

#### 🔄 **MRPC (Microsoft Research Paraphrase Corpus)**
- **목적**: 문장 쌍의 의미적 동등성 판정
- **특징**: 뉴스 기사에서 추출한 문장 쌍의 패러프레이즈 여부 판단
- **용도**: 문장 간 의미 유사성 이해 능력 평가
- **노이즈 적용**: sentence1, sentence2 텍스트 노이즈 또는 label 플리핑 (0↔1)

## 🔧 설치 및 환경 설정

### 1. 저장소 클론 및 환경 설정
```bash
git clone https://github.com/pinkgom/yonsei_datainf_dataset.git
cd datainf-project
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. 의존성 설치"
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

### 🔄 라벨 플리핑 (Label Flipping)

**DataInf 논문 표준** 기법으로, 분류 태스크에서 정답 라벨을 의도적으로 뒤바꿔서 모델에 미치는 영향을 분석합니다.

#### 🎯 **지원 데이터셋**
- **SST-2**: 감정 분류 (0: Negative ↔ 1: Positive)
- **MRPC**: 유사도 판정 (0: Not Paraphrase ↔ 1: Paraphrase)

#### 📊 **라벨 플리핑 예시**

**SST-2 (감정 분류)**
```
원본: "This movie is amazing!" → Label: 1 (Positive)
플리핑: "This movie is amazing!" → Label: 0 (Negative)
```

**MRPC (유사도 판정)**
```
원본: 
  Sentence1: "The cat is sleeping."
  Sentence2: "A cat is taking a nap."
  Label: 1 (Paraphrase)
플리핑:
  Sentence1: "The cat is sleeping."
  Sentence2: "A cat is taking a nap."
  Label: 0 (Not Paraphrase)
```

#### ⚡ **사용법**
```bash
# SST-2에서 20% 라벨 플리핑
python main.py --full --dataset sst2 --noise-ratio 0.2 --flip-labels

# MRPC에서 15% 라벨 플리핑
python main.py --full --dataset mrpc --noise-ratio 0.15 --flip-labels
```

## 🚀 실행 옵션 상세 가이드

### 🎮 **데모 모드** - 첫 사용 추천
```bash
# 기본 데모 (Alpaca)
python main.py --demo --dataset alpaca

# 다른 데이터셋으로 데모
python main.py --demo --dataset gsm8k
python main.py --demo --dataset sst2
python main.py --demo --dataset mrpc
```
- **목적**: 빠른 프로토타입 및 기능 테스트
- **샘플 크기**: 500개 (모든 데이터셋)
- **예상 시간**: 1-2분
- **생성 파일**: 
  - `{dataset}_original_demo_500.json` (원본)
  - `{dataset}_demo_20percent_balanced_500.json` (균형형 20% 노이즈)
  - `{dataset}_demo_20percent_grammar_heavy_500.json` (문법 중심 20% 노이즈)

### 🏭 **전체 데이터 모드** - 실제 실험용

#### 기본 실험
```bash
# Alpaca 전체 데이터 실험
python main.py --full --dataset alpaca --noise-ratio 0.2 --strategy balanced

# GSM8K 수학 문제 실험
python main.py --full --dataset gsm8k --noise-ratio 0.15 --strategy balanced

# SST-2 감정 분류 실험 (라벨 보존)
python main.py --full --dataset sst2 --noise-ratio 0.2 --strategy semantic_heavy

# MRPC 유사도 판정 실험 (라벨 보존)
python main.py --full --dataset mrpc --noise-ratio 0.25 --strategy grammar_heavy

# SST-2 라벨 플리핑 실험 (0↔1 플리핑)
python main.py --full --dataset sst2 --noise-ratio 0.2 --flip-labels

# MRPC 라벨 플리핑 실험 (0↔1 플리핑)
python main.py --full --dataset mrpc --noise-ratio 0.15 --flip-labels
```
- **샘플 크기**: 데이터셋별 다름 (Alpaca: 52K, GSM8K: 7.5K, SST-2: 67K, MRPC: 3.7K)
- **예상 시간**: 5-30분 (데이터셋 크기에 따라)
- **라벨 보존**: GSM8K, SST-2, MRPC는 정답 라벨 자동 보존

#### 고급 실험 옵션

**1. 여러 노이즈 비율 한번에**
```bash
# 다양한 비율로 Alpaca 실험
python main.py --full --dataset alpaca --noise-ratios 0.1,0.15,0.2,0.25 --strategy balanced

# GSM8K에서 수학 추론 능력 비교
python main.py --full --dataset gsm8k --noise-ratios 0.1,0.2,0.3 --strategy all
```
- 4가지 노이즈 비율 (10%, 15%, 20%, 25%)로 각각 데이터셋 생성
- 분류 태스크에서는 라벨 자동 보존
- 예상 시간: 데이터셋과 실험 수에 따라 다름

**2. 모든 전략 테스트**
```bash
# 단일 데이터셋, 모든 전략
python main.py --full --dataset sst2 --noise-ratio 0.2 --strategy all

# 여러 데이터셋에서 전략 비교
python main.py --full --dataset gsm8k --noise-ratio 0.15 --strategy balanced,semantic_heavy
```
- balanced, grammar_heavy, semantic_heavy 3가지 전략 모두 적용
- 같은 노이즈 비율로 3가지 다른 데이터셋 생성
- 분류 태스크: 입력만 노이즈, 라벨은 보존

**3. 복합 실험**
```bash
# 복합 실험 (비율 × 전략 × 데이터셋)
python main.py --full --dataset alpaca --noise-ratios 0.15,0.2 --strategy balanced,grammar_heavy

# 다중 데이터셋 비교 실험
python main.py --full --dataset sst2 --noise-ratios 0.1,0.2 --strategy all
python main.py --full --dataset mrpc --noise-ratios 0.1,0.2 --strategy all
```
- 2가지 비율 × 2가지 전략 = 4개 데이터셋 생성
- 다양한 NLP 태스크에서 체계적인 비교 실험 가능
- 라벨 보존 여부는 데이터셋에 따라 자동 결정

**4. 자동 실행 (확인 건너뛰기)**
```bash
# 자동 실행 모드
python main.py --full --dataset gsm8k --noise-ratio 0.2 --strategy balanced --yes

# 배치 실험용 자동 실행
python main.py --full --dataset sst2 --noise-ratios 0.1,0.15,0.2 --strategy all --yes
```
- `-y` 또는 `--yes` 옵션으로 확인 절차 생략
- 스크립트 자동화에 유용
- 모든 데이터셋에서 사용 가능

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
# Alpaca에서 다양한 노이즈 비율로 데이터셋 생성
python main.py --full --dataset alpaca --noise-ratios 0.05,0.1,0.15,0.2,0.25,0.3 --strategy balanced --yes

# 분류 태스크에서 라벨 보존 노이즈 데이터셋 생성
python main.py --full --dataset sst2 --noise-ratios 0.1,0.2,0.3 --strategy all --yes

# 결과 분석
python main.py --analysis  # → 옵션 3 (전체 파일 상세 분석)
```

#### 시나리오 2: 태스크별 노이즈 전략 비교 연구
```bash
# 수학 추론 태스크 데이터셋 생성 (answer 라벨 보존)
python main.py --full --dataset gsm8k --noise-ratio 0.2 --strategy all --yes

# 감정 분류 데이터셋 생성 (label 보존)
python main.py --full --dataset sst2 --noise-ratio 0.15 --strategy all --yes

# 유사도 판정 데이터셋 생성 (label 보존)
python main.py --full --dataset mrpc --noise-ratio 0.2 --strategy all --yes

# 전략별 효과 분석
python main.py --analysis  # → 옵션 5 (전략별 효과 분석)
```

#### 시나리오 3: 다중 데이터셋 비교 연구
```bash
# 모든 데이터셋에서 동일 조건 실험
python main.py --full --dataset alpaca --noise-ratios 0.15,0.25 --strategy balanced,semantic_heavy --yes
python main.py --full --dataset gsm8k --noise-ratios 0.15,0.25 --strategy balanced,semantic_heavy --yes
python main.py --full --dataset sst2 --noise-ratios 0.15,0.25 --strategy balanced,semantic_heavy --yes
python main.py --full --dataset mrpc --noise-ratios 0.15,0.25 --strategy balanced,semantic_heavy --yes

# 태스크별 라벨 보존 효과 종합 분석
python main.py --analysis  # → 옵션 6 (전체 종합 분석)
```

## 📚 참고 자료

### 📖 **관련 논문**
- [DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs](https://openreview.net/pdf?id=9m02jb92Wz)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### 🔗 **관련 리소스**

#### 📊 **데이터셋 원본 링크**
- [Stanford Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca) - Instruction-following 데이터셋
- [GSM8K Dataset](https://github.com/openai/grade-school-math) - OpenAI의 수학 문제 데이터셋
- [SST-2 (GLUE)](https://gluebenchmark.com/tasks) - Stanford Sentiment Treebank v2
- [MRPC (GLUE)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) - Microsoft Research Paraphrase Corpus

#### 🤗 **HuggingFace 데이터셋 페이지**
- [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) - Alpaca 데이터셋
- [gsm8k](https://huggingface.co/datasets/gsm8k) - GSM8K 데이터셋
- [glue/sst2](https://huggingface.co/datasets/glue) - SST-2 데이터셋
- [glue/mrpc](https://huggingface.co/datasets/glue) - MRPC 데이터셋

#### 🛠️ **도구 및 프레임워크**
- [Hugging Face Datasets](https://huggingface.co/datasets) - 데이터셋 라이브러리
- [DataInf GitHub Repository](https://github.com/ykwon0407/DataInf) - 원본 DataInf 구현

### 📊 **실험 레퍼런스**
- [Stanford CS224N Projects](https://web.stanford.edu/class/cs224n/project.html)
- [CS231n Reports](http://cs231n.stanford.edu/2017/reports.html)

## 📄 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다. 연구 및 교육 목적으로 자유롭게 사용하실 수 있습니다.

