import os
import sys
from src.data_loader import AlpacaDataLoader
from src.noise_injection import NoiseInjector, compare_samples


def print_separator(title="", char="=", length=60):
    """구분선 출력 함수"""
    if title:
        title_line = f" {title} "
        padding = (length - len(title_line)) // 2
        line = char * padding + title_line + char * padding
        if len(line) < length:
            line += char
    else:
        line = char * length
    print(line)


def main():
    print_separator("DataInf 노이즈 주입 프로젝트", "=", 60)
    print("🎯 목표: Alpaca 데이터셋에 다양한 노이즈를 주입하여")
    print("       DataInf 알고리즘의 데이터 정제 효과를 검증")
    print()

    # 프로젝트 설정
    print("⚙️  프로젝트 설정 확인...")
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 데이터 디렉토리 생성: {data_dir}")

    # ========================================
    # 1단계: 데이터 로딩
    # ========================================
    print_separator("1단계: 데이터 로딩", "-")

    loader = AlpacaDataLoader()

    # 캐시 정보 확인
    print("💾 캐시 상태 확인:")
    cache_exists = loader.get_cache_info()

    # 데이터 로딩 (처음엔 작은 샘플로 테스트)
    print(f"\n📊 데이터 로딩 중...")

    # 개발/테스트 단계에서는 작은 샘플 사용
    SAMPLE_SIZE = 500  # 테스트용 샘플 크기
    df = loader.load_alpaca_dataset(subset_size=SAMPLE_SIZE)

    if df is None:
        print("❌ 데이터 로딩 실패. 프로그램을 종료합니다.")
        return False

    print("✅ 데이터 로딩 성공!")

    # 원본 데이터 저장
    original_filename = f"alpaca_original_{SAMPLE_SIZE}.json"
    loader.save_dataset(df, original_filename)

    # ========================================
    # 2단계: 노이즈 주입
    # ========================================
    print_separator("2단계: 노이즈 주입", "-")

    injector = NoiseInjector(random_seed=42)

    # 다양한 노이즈 비율로 실험
    noise_experiments = [
        {"ratio": 0.1, "name": "10%"},
        {"ratio": 0.2, "name": "20%"},
        {"ratio": 0.3, "name": "30%"}
    ]

    experiment_results = {}

    for exp in noise_experiments:
        ratio = exp["ratio"]
        name = exp["name"]

        print(f"\n🧪 노이즈 비율 {name} 실험 진행...")

        # 노이즈 주입
        noisy_df, noisy_indices = injector.inject_noise(df.copy(), noise_ratio=ratio)

        # 결과 저장
        noisy_filename = f"alpaca_noisy_{name.replace('%', 'percent')}_{SAMPLE_SIZE}.json"
        saved_path = loader.save_dataset(noisy_df, noisy_filename)

        # 실험 결과 기록
        experiment_results[name] = {
            "original_count": len(df),
            "noisy_count": len(noisy_indices),
            "noisy_ratio": ratio,
            "filename": noisy_filename,
            "filepath": saved_path
        }

        print(f"   ✅ {name} 노이즈 주입 완료 ({len(noisy_indices)}개 샘플)")

    # ========================================
    # 3단계: 결과 분석 및 비교
    # ========================================
    print_separator("3단계: 결과 분석", "-")

    # 첫 번째 실험 결과로 샘플 비교 표시
    first_exp = noise_experiments[0]
    ratio = first_exp["ratio"]
    name = first_exp["name"]

    print(f"📋 노이즈 주입 전후 비교 (노이즈 비율: {name})")

    # 다시 노이즈 주입 (비교용)
    noisy_df_sample, noisy_indices_sample = injector.inject_noise(df.copy(), noise_ratio=ratio)

    # 샘플 비교 표시
    compare_samples(df, noisy_df_sample, noisy_indices_sample, num_examples=3)

    # ========================================
    # 4단계: 실험 결과 요약
    # ========================================
    print_separator("실험 결과 요약", "=")

    print("📊 생성된 데이터셋 파일들:")
    print(f"   🔹 원본 데이터: {original_filename}")

    for name, result in experiment_results.items():
        print(f"   🔸 노이즈 {name}: {result['filename']}")
        print(f"      - 노이즈 주입 샘플: {result['noisy_count']}개")
        print(f"      - 노이즈 비율: {result['noisy_ratio'] * 100:.1f}%")

    print(f"\n📁 모든 파일이 '{data_dir}' 폴더에 저장되었습니다.")

    # ========================================
    # 5단계: 다음 단계 안내
    # ========================================
    print_separator("다음 단계 안내", "=")
    print("🎯 완료된 작업:")
    print("   ✅ Alpaca 데이터셋 로딩 및 캐싱")
    print("   ✅ 다양한 비율의 노이즈 주입")
    print("   ✅ 노이즈 유형별 테스트 (문법, 의미, 품질)")
    print("   ✅ 결과 데이터셋 저장")

    print("\n🔜 다음 작업:")
    print("   📌 DataInf 알고리즘 구현")
    print("   📌 영향력 점수 계산")
    print("   📌 데이터 정제 효과 검증")
    print("   📌 성능 비교 및 분석")

    print(f"\n✨ 노이즈 주입 단계가 성공적으로 완료되었습니다!")

    return True


def run_cache_management():
    """캐시 관리 함수 (옵션)"""
    loader = AlpacaDataLoader()

    print("=== 캐시 관리 메뉴 ===")
    print("1. 캐시 정보 확인")
    print("2. 캐시 삭제")
    print("3. 강제 재다운로드")

    choice = input("선택하세요 (1-3): ").strip()

    if choice == "1":
        loader.get_cache_info()
    elif choice == "2":
        loader.clear_cache()
    elif choice == "3":
        print("강제 재다운로드를 시작합니다...")
        df = loader.load_alpaca_dataset(subset_size=100, force_download=True)
        if df is not None:
            print("✅ 재다운로드 완료!")
    else:
        print("올바르지 않은 선택입니다.")


if __name__ == "__main__":
    # 명령행 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--cache":
        run_cache_management()
    else:
        success = main()

        if success:
            print("\n" + "=" * 60)
            print("🎉 프로그램이 성공적으로 완료되었습니다!")
            print("💡 캐시 관리를 원하시면: python main.py --cache")
        else:
            print("\n" + "=" * 60)
            print("❌ 프로그램 실행 중 오류가 발생했습니다.")