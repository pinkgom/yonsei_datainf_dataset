from datasets import load_dataset
import pandas as pd
import json
import os


class MultiDatasetLoader:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # 지원하는 데이터셋 정의
        self.supported_datasets = {
            'alpaca': {
                'loader': self._load_alpaca,
                'cache_file': 'alpaca_full_dataset.json',
                'columns': ['instruction', 'input', 'output'],
                'text_columns': ['instruction', 'output'],  # 노이즈 적용 대상
                'label_columns': []  # 라벨 없음
            },
            'gsm8k': {
                'loader': self._load_gsm8k,
                'cache_file': 'gsm8k_full_dataset.json',
                'columns': ['question', 'answer'],
                'text_columns': ['question'],  # question만 노이즈 적용
                'label_columns': ['answer']  # answer는 보존
            },
            'sst2': {
                'loader': self._load_sst2,
                'cache_file': 'sst2_full_dataset.json',
                'columns': ['sentence', 'label'],
                'text_columns': ['sentence'],  # sentence만 노이즈 적용
                'label_columns': ['label']  # label은 보존
            },
            'mrpc': {
                'loader': self._load_mrpc,
                'cache_file': 'mrpc_full_dataset.json',
                'columns': ['sentence1', 'sentence2', 'label'],
                'text_columns': ['sentence1', 'sentence2'],  # 두 문장 모두 노이즈 가능
                'label_columns': ['label']
            }
        }

    def load_dataset(self, dataset_name='alpaca', subset_size=None, force_download=False):
        """
        다양한 데이터셋 로드 (통합 인터페이스)

        Args:
            dataset_name: 'alpaca', 'gsm8k', 'sst2', 'mrpc' 등
            subset_size: 테스트용으로 일부만 로드하려면 숫자 입력
            force_download: True시 강제로 재다운로드
        """
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")

        dataset_info = self.supported_datasets[dataset_name]
        cache_file = dataset_info['cache_file']
        cache_path = os.path.join(self.data_dir, cache_file)

        # 1. 캐시된 데이터 확인
        if os.path.exists(cache_path) and not force_download:
            print(f"저장된 {dataset_name} 데이터셋을 로딩중...")
            try:
                df = pd.read_json(cache_path)
                print(f"캐시된 데이터 로드 성공! 전체 데이터 개수: {len(df)}")
            except Exception as e:
                print(f"캐시된 데이터 로드 실패: {e}")
                print("새로 다운로드를 시도합니다...")
                df = self._download_and_save_dataset(dataset_name)
        else:
            print(f"{dataset_name} 데이터셋을 새로 다운로드합니다...")
            df = self._download_and_save_dataset(dataset_name)

        if df is None:
            return None

        # 2. subset_size 적용
        if subset_size and subset_size < len(df):
            print(f"전체 데이터에서 {subset_size}개 샘플을 추출합니다.")
            df = df.head(subset_size).reset_index(drop=True)

        # 3. 데이터 정보 출력
        self._print_dataset_info(df, dataset_name)

        return df

    def _download_and_save_dataset(self, dataset_name):
        """데이터셋별 다운로드 및 저장"""
        try:
            dataset_info = self.supported_datasets[dataset_name]
            loader_func = dataset_info['loader']
            cache_file = dataset_info['cache_file']

            print(f"인터넷에서 {dataset_name} 데이터셋을 다운로드중...")

            # 데이터셋별 로더 호출
            df = loader_func()

            if df is not None:
                # 캐시 저장
                cache_path = os.path.join(self.data_dir, cache_file)
                df.to_json(cache_path, orient='records', force_ascii=False, indent=2)
                print(f"데이터셋 저장 완료: {cache_path}")

            return df

        except Exception as e:
            print(f"{dataset_name} 다운로드 중 오류 발생: {e}")
            return None

    def _load_alpaca(self):
        """Alpaca 데이터셋 로드"""
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        df = pd.DataFrame(dataset)
        print(f"Alpaca 다운로드 완료! 전체 데이터 개수: {len(df)}")
        return df

    def _load_gsm8k(self):
        """GSM8K 데이터셋 로드"""
        dataset = load_dataset("gsm8k", "main", split="train")
        df = pd.DataFrame(dataset)
        print(f"GSM8K 다운로드 완료! 전체 데이터 개수: {len(df)}")
        return df

    def _load_sst2(self):
        """Stanford Sentiment Treebank (SST-2) 로드"""
        dataset = load_dataset("glue", "sst2", split="train")
        df = pd.DataFrame(dataset)
        print(f"SST-2 다운로드 완료! 전체 데이터 개수: {len(df)}")
        return df

    def _load_mrpc(self):
        """Microsoft Research Paraphrase Corpus (MRPC) 로드"""
        dataset = load_dataset("glue", "mrpc", split="train")
        df = pd.DataFrame(dataset)
        print(f"MRPC 다운로드 완료! 전체 데이터 개수: {len(df)}")
        return df

    def _print_dataset_info(self, df, dataset_name):
        """데이터셋별 정보 출력"""
        dataset_info = self.supported_datasets[dataset_name]

        print(f"\n=== {dataset_name.upper()} 데이터셋 정보 ===")
        print(f"컬럼명: {df.columns.tolist()}")
        print(f"데이터 크기: {df.shape}")
        print(f"텍스트 컬럼: {dataset_info['text_columns']}")
        print(f"라벨 컬럼: {dataset_info['label_columns']}")

        # 데이터셋별 통계
        if dataset_name == 'alpaca':
            self._print_alpaca_stats(df)
        elif dataset_name == 'gsm8k':
            self._print_gsm8k_stats(df)
        elif dataset_name == 'sst2':
            self._print_sst2_stats(df)
        elif dataset_name == 'mrpc':
            self._print_mrpc_stats(df)

        # 샘플 데이터 출력
        self._print_samples(df, dataset_name)

    def _print_alpaca_stats(self, df):
        """Alpaca 통계"""
        inst_lengths = df['instruction'].str.len()
        out_lengths = df['output'].str.len()
        print(f"Instruction 길이 - 평균: {inst_lengths.mean():.1f}, 최대: {inst_lengths.max()}, 최소: {inst_lengths.min()}")
        print(f"Output 길이 - 평균: {out_lengths.mean():.1f}, 최대: {out_lengths.max()}, 최소: {out_lengths.min()}")

    def _print_gsm8k_stats(self, df):
        """GSM8K 통계"""
        q_lengths = df['question'].str.len()
        a_lengths = df['answer'].str.len()
        print(f"Question 길이 - 평균: {q_lengths.mean():.1f}, 최대: {q_lengths.max()}, 최소: {q_lengths.min()}")
        print(f"Answer 길이 - 평균: {a_lengths.mean():.1f}, 최대: {a_lengths.max()}, 최소: {a_lengths.min()}")

    def _print_sst2_stats(self, df):
        """SST-2 통계"""
        sent_lengths = df['sentence'].str.len()
        label_dist = df['label'].value_counts()
        print(f"Sentence 길이 - 평균: {sent_lengths.mean():.1f}, 최대: {sent_lengths.max()}, 최소: {sent_lengths.min()}")
        print(f"라벨 분포: {dict(label_dist)}")

    def _print_mrpc_stats(self, df):
        """MRPC 통계"""
        s1_lengths = df['sentence1'].str.len()
        s2_lengths = df['sentence2'].str.len()
        label_dist = df['label'].value_counts()
        print(f"Sentence1 길이 - 평균: {s1_lengths.mean():.1f}")
        print(f"Sentence2 길이 - 평균: {s2_lengths.mean():.1f}")
        print(f"라벨 분포: {dict(label_dist)}")

    def _print_samples(self, df, dataset_name):
        """샘플 데이터 출력"""
        print(f"\n=== 샘플 데이터 (상위 2개) ===")

        for i in range(min(2, len(df))):
            print(f"\n[샘플 {i + 1}]")

            if dataset_name == 'alpaca':
                print(f"Instruction: {df.iloc[i]['instruction']}")
                if pd.notna(df.iloc[i]['input']) and df.iloc[i]['input'].strip():
                    print(f"Input: {df.iloc[i]['input']}")
                print(f"Output: {df.iloc[i]['output']}")

            elif dataset_name == 'gsm8k':
                print(f"Question: {df.iloc[i]['question']}")
                print(f"Answer: {df.iloc[i]['answer']}")

            elif dataset_name == 'sst2':
                print(f"Sentence: {df.iloc[i]['sentence']}")
                print(f"Label: {df.iloc[i]['label']} ({'Positive' if df.iloc[i]['label'] == 1 else 'Negative'})")

            elif dataset_name == 'mrpc':
                print(f"Sentence1: {df.iloc[i]['sentence1']}")
                print(f"Sentence2: {df.iloc[i]['sentence2']}")
                print(
                    f"Label: {df.iloc[i]['label']} ({'Paraphrase' if df.iloc[i]['label'] == 1 else 'Not paraphrase'})")

            print("-" * 50)

    def get_dataset_info(self, dataset_name):
        """데이터셋 정보 반환"""
        return self.supported_datasets.get(dataset_name, None)

    def save_dataset(self, df, filename):
        """데이터셋을 JSON 파일로 저장"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_json(filepath, orient='records', force_ascii=False, indent=2)
        print(f"데이터셋이 저장되었습니다: {filepath}")
        return filepath

    def load_saved_dataset(self, filename):
        """저장된 데이터셋 로드"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_json(filepath)
                print(f"저장된 데이터셋을 로드했습니다: {filepath}")
                return df
            except Exception as e:
                print(f"파일 로드 실패: {e}")
                return None
        else:
            print(f"파일을 찾을 수 없습니다: {filepath}")
            return None


# 하위 호환성을 위한 AlpacaDataLoader 유지
class AlpacaDataLoader(MultiDatasetLoader):
    def __init__(self, data_dir="./data"):
        super().__init__(data_dir)

    def load_alpaca_dataset(self, subset_size=None, force_download=False):
        """기존 Alpaca 전용 메서드 (하위 호환성)"""
        return self.load_dataset('alpaca', subset_size, force_download)


if __name__ == "__main__":
    # 테스트 실행
    loader = MultiDatasetLoader()

    print("=== MultiDatasetLoader 테스트 ===")

    # 각 데이터셋 테스트 (작은 샘플로)
    for dataset_name in ['alpaca', 'gsm8k', 'sst2']:
        print(f"\n>>> {dataset_name} 테스트")
        df = loader.load_dataset(dataset_name, subset_size=100)
        if df is not None:
            print(f"{dataset_name} 테스트 성공!")
        else:
            print(f"{dataset_name} 테스트 실패")