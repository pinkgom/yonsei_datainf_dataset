from datasets import load_dataset
import pandas as pd
import json
import os


class AlpacaDataLoader:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.full_dataset_file = "alpaca_full_dataset.json"

    def load_alpaca_dataset(self, subset_size=None, force_download=False):
        """
        Alpaca 데이터셋 로드 (캐싱 기능 포함)

        Args:
            subset_size: 테스트용으로 일부만 로드하려면 숫자 입력 (예: 1000)
            force_download: True시 강제로 재다운로드
        """
        full_dataset_path = os.path.join(self.data_dir, self.full_dataset_file)

        # 1. 이미 저장된 파일이 있는지 확인
        if os.path.exists(full_dataset_path) and not force_download:
            print("저장된 Alpaca 데이터셋을 로딩중...")
            try:
                df = pd.read_json(full_dataset_path)
                print(f"캐시된 데이터 로드 성공! 전체 데이터 개수: {len(df)}")
            except Exception as e:
                print(f"캐시된 데이터 로드 실패: {e}")
                print("새로 다운로드를 시도합니다...")
                df = self._download_and_save_dataset()
        else:
            # 2. 파일이 없거나 강제 다운로드인 경우
            if force_download:
                print("강제 재다운로드를 시작합니다...")
            else:
                print("저장된 데이터셋이 없습니다. 새로 다운로드합니다...")

            df = self._download_and_save_dataset()

        if df is None:
            return None

        # 3. subset_size가 지정된 경우 일부만 추출
        if subset_size and subset_size < len(df):
            print(f"전체 데이터에서 {subset_size}개 샘플을 추출합니다.")
            df = df.head(subset_size).reset_index(drop=True)
            print(f"{len(df)}개 샘플 준비 완료")

        # 4. 데이터 구조 정보 출력
        self._print_dataset_info(df)

        return df

    def _download_and_save_dataset(self):
        """Alpaca 데이터셋 다운로드 및 저장"""
        try:
            print("인터넷에서 Alpaca 데이터셋을 다운로드중...")
            print("처음 다운로드시 시간이 걸릴 수 있습니다...")

            # Stanford Alpaca 52K 데이터셋 로드
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
            print(f"다운로드 완료! 전체 데이터 개수: {len(dataset)}")

            # pandas DataFrame으로 변환
            df = pd.DataFrame(dataset)

            # 전체 데이터셋 저장
            full_dataset_path = os.path.join(self.data_dir, self.full_dataset_file)
            df.to_json(full_dataset_path, orient='records', force_ascii=False, indent=2)
            print(f"전체 데이터셋 저장 완료: {full_dataset_path}")

            return df

        except Exception as e:
            print(f"데이터 다운로드 중 오류 발생: {e}")
            print("인터넷 연결을 확인하거나 잠시 후 다시 시도해주세요.")
            return None

    def _print_dataset_info(self, df):
        """데이터셋 정보 출력"""
        print(f"\n=== 데이터셋 정보 ===")
        print(f"컬럼명: {df.columns.tolist()}")
        print(f"데이터 크기: {df.shape}")
        print(f"데이터 타입:\n{df.dtypes}")

        # 기본 통계
        if 'instruction' in df.columns:
            inst_lengths = df['instruction'].str.len()
            print(
                f"Instruction 길이 - 평균: {inst_lengths.mean():.1f}, 최대: {inst_lengths.max()}, 최소: {inst_lengths.min()}")

        if 'output' in df.columns:
            out_lengths = df['output'].str.len()
            print(f"Output 길이 - 평균: {out_lengths.mean():.1f}, 최대: {out_lengths.max()}, 최소: {out_lengths.min()}")

        # 샘플 데이터 출력
        print(f"\n=== 샘플 데이터 (상위 2개) ===")
        for i in range(min(2, len(df))):
            print(f"\n[샘플 {i + 1}]")
            print(f"Instruction: {df.iloc[i]['instruction']}")
            if 'input' in df.columns and pd.notna(df.iloc[i]['input']) and df.iloc[i]['input'].strip():
                print(f"Input: {df.iloc[i]['input']}")
            print(f"Output: {df.iloc[i]['output']}")
            print("-" * 50)

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

    def get_cache_info(self):
        """캐시 정보 확인"""
        full_dataset_path = os.path.join(self.data_dir, self.full_dataset_file)

        if os.path.exists(full_dataset_path):
            file_size = os.path.getsize(full_dataset_path) / (1024 * 1024)  # MB
            import time
            mod_time = time.ctime(os.path.getmtime(full_dataset_path))
            print(f"캐시 파일 정보:")
            print(f"   - 경로: {full_dataset_path}")
            print(f"   - 크기: {file_size:.2f} MB")
            print(f"   - 수정일: {mod_time}")
            return True
        else:
            print("캐시된 데이터셋이 없습니다.")
            return False

    def clear_cache(self):
        """캐시 파일 삭제"""
        full_dataset_path = os.path.join(self.data_dir, self.full_dataset_file)
        if os.path.exists(full_dataset_path):
            os.remove(full_dataset_path)
            print(f"캐시 파일이 삭제되었습니다: {full_dataset_path}")
            return True
        else:
            print("삭제할 캐시 파일이 없습니다.")
            return False


if __name__ == "__main__":
    # 테스트 실행
    loader = AlpacaDataLoader()

    print("=== AlpacaDataLoader 테스트 ===")

    # 캐시 정보 확인
    loader.get_cache_info()

    # 작은 샘플로 테스트
    df = loader.load_alpaca_dataset(subset_size=100)

    if df is not None:
        print("테스트 성공!")

        # 샘플 저장 테스트
        loader.save_dataset(df, "alpaca_test_sample.json")
    else:
        print("테스트 실패")