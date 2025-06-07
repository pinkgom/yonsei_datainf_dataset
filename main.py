import os
import sys
import argparse
import time
from datetime import datetime
from src.data_loader import MultiDatasetLoader, AlpacaDataLoader  # 하위 호환성
from src.noise_injection import MultiDatasetNoiseInjector, NoiseInjector, compare_samples, analyze_noise_distribution
from src.analysis import run_quality_analysis


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


def run_demo_mode(args):
    """데모 모드 (500개 샘플로 빠른 테스트)"""
    dataset_name = args.dataset

    print_separator(f"DataInf {dataset_name.upper()} 데모 모드", "=", 70)
    print(f"데이터셋: {dataset_name}")
    print("목적: 빠른 프로토타입 및 기능 테스트")
    print("샘플 크기: 500개")
    print("예상 시간: 1-2분")
    print()

    # 데이터 로딩
    loader = MultiDatasetLoader()
    SAMPLE_SIZE = 500

    print(f"{dataset_name} 데이터 로딩 중...")
    df = loader.load_dataset(dataset_name, subset_size=SAMPLE_SIZE)

    if df is None:
        print("데이터 로딩 실패")
        return False

    # 원본 데이터 저장
    original_filename = f"{dataset_name}_original_demo_{SAMPLE_SIZE}.json"
    loader.save_dataset(df, original_filename)

    # 데모용 노이즈 실험
    injector = MultiDatasetNoiseInjector(random_seed=42)

    demo_experiments = [
        {"ratio": 0.2, "strategy": "balanced"},
        {"ratio": 0.2, "strategy": "grammar_heavy"}
    ]

    print_separator("노이즈 주입 데모", "-")

    for exp in demo_experiments:
        ratio = exp["ratio"]
        strategy = exp["strategy"]

        print(f"\n{strategy} 전략 ({ratio * 100:.0f}% 노이즈) 테스트...")

        noisy_df, noisy_indices = injector.inject_noise(
            df.copy(),
            dataset_name=dataset_name,
            noise_ratio=ratio,
            noise_strategy=strategy
        )

        # 파일 저장
        filename = f"{dataset_name}_demo_{ratio * 100:.0f}percent_{strategy}_{SAMPLE_SIZE}.json"
        loader.save_dataset(noisy_df, filename)

        # 분석
        analysis = analyze_noise_distribution(noisy_df, df, noisy_indices, dataset_name)
        print(f"   완료: 실제 변경 {analysis['actual_changes']}/{len(noisy_indices)}개")

    # 샘플 비교
    print_separator("결과 비교", "-")
    final_noisy_df, final_noisy_indices = injector.inject_noise(
        df, dataset_name=dataset_name, noise_ratio=0.2, noise_strategy="balanced"
    )
    compare_samples(df, final_noisy_df, final_noisy_indices, num_examples=2, dataset_name=dataset_name)

    print(f"\n{dataset_name} 데모 완료! 생성된 파일들을 data/ 폴더에서 확인하세요.")
    return True


def run_full_mode(args):
    """전체 데이터 모드 (실제 실험용)"""
    dataset_name = args.dataset

    print_separator(f"DataInf {dataset_name.upper()} 전체 데이터 모드", "=", 70)
    print(f"데이터셋: {dataset_name}")
    print("목적: 실제 DataInf 실험용 데이터셋 생성")

    # 데이터셋별 예상 크기 정보
    dataset_sizes = {
        'alpaca': '52,002개',
        'gsm8k': '7,473개',
        'sst2': '67,349개',
        'mrpc': '3,668개'
    }
    print(f"예상 샘플 크기: {dataset_sizes.get(dataset_name, '알 수 없음')}")
    print("예상 시간: 5-30분 (데이터셋과 옵션에 따라)")
    print()

    # 인자 파싱
    noise_ratios = parse_noise_ratios(args.noise_ratios) if args.noise_ratios else [args.noise_ratio]
    strategies = parse_strategies(args.strategy)

    print(f"설정:")
    print(f"   - 노이즈 비율: {[f'{r * 100:.0f}%' for r in noise_ratios]}")
    print(f"   - 노이즈 전략: {strategies}")
    print()

    # 사용자 확인
    total_experiments = len(noise_ratios) * len(strategies)
    estimated_time = total_experiments * 3  # 실험당 약 3분 추정

    print(f"총 {total_experiments}개 실험이 실행됩니다 (예상 시간: {estimated_time}분)")

    if not args.yes:
        confirm = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
        if confirm != 'y':
            print("실행이 취소되었습니다.")
            return False

    # 데이터 로딩
    loader = MultiDatasetLoader()

    print(f"{dataset_name} 전체 데이터 로딩 중...")
    start_time = time.time()
    df = loader.load_dataset(dataset_name)  # 전체 데이터
    load_time = time.time() - start_time

    if df is None:
        print("데이터 로딩 실패")
        return False

    print(f"데이터 로딩 완료 ({len(df):,}개, {load_time:.1f}초)")

    # 원본 데이터 저장
    original_filename = f"{dataset_name}_original_full_{len(df)}.json"
    print(f"원본 데이터 저장 중...")
    loader.save_dataset(df, original_filename)

    # 실험 실행
    injector = MultiDatasetNoiseInjector(random_seed=42)
    results = {}

    print_separator("노이즈 주입 실험 시작", "=")

    experiment_count = 0
    for ratio in noise_ratios:
        for strategy in strategies:
            experiment_count += 1

            print(f"\n실험 {experiment_count}/{total_experiments}: {strategy} 전략, {ratio * 100:.0f}% 노이즈")
            print(f"   예상 노이즈 샘플 수: {int(len(df) * ratio):,}개")

            exp_start_time = time.time()

            # 노이즈 주입 (라벨 플리핑 옵션 포함)
            flip_labels = hasattr(args, 'flip_labels') and args.flip_labels
            noisy_df, noisy_indices = injector.inject_noise(
                df.copy(),
                dataset_name=dataset_name,
                noise_ratio=ratio,
                noise_strategy=strategy,
                flip_labels=flip_labels
            )

            # 파일 저장
            if strategy == "balanced":
                filename = f"{dataset_name}_full_{ratio * 100:.0f}percent_{len(df)}.json"
            else:
                strategy_short = strategy.replace('_heavy', '').replace('_', '')
                filename = f"{dataset_name}_full_{ratio * 100:.0f}percent_{strategy_short}_{len(df)}.json"

            print(f"   파일 저장 중: {filename}")
            saved_path = loader.save_dataset(noisy_df, filename)

            # 분석
            analysis = analyze_noise_distribution(noisy_df, df, noisy_indices, dataset_name)

            exp_time = time.time() - exp_start_time
            print(f"   완료 ({exp_time:.1f}초)")
            print(f"      - 실제 변경: {analysis['actual_changes']:,}/{len(noisy_indices):,}개")
            print(f"      - 평균 길이 변화: {analysis['avg_length_change']:.1f} 문자")

            # 라벨 보존 확인 (classification 데이터셋의 경우)
            config = injector.dataset_configs.get(dataset_name, {})
            if config.get('preserve_labels', False):
                label_columns = config.get('label_columns', [])
                for label_col in label_columns:
                    label_changes = analysis['field_changes'].get(label_col, 0)
                    if label_changes > 0:
                        print(f"      - ⚠️  WARNING: {label_col} 라벨이 {label_changes}개 변경됨!")
                    else:
                        print(f"      - ✅ {label_col} 라벨 보존 성공")

            # 결과 저장
            results[f"{ratio * 100:.0f}%_{strategy}"] = {
                'filename': filename,
                'filepath': saved_path,
                'noise_ratio': ratio,
                'strategy': strategy,
                'analysis': analysis,
                'processing_time': exp_time
            }

    # 결과 요약
    print_separator("실험 결과 요약", "=")

    total_time = time.time() - start_time
    print(f"{dataset_name} 모든 실험 완료! (총 소요시간: {total_time / 60:.1f}분)")
    print()

    print("생성된 파일:")
    print(f"   원본: {original_filename}")

    for exp_name, result in results.items():
        analysis = result['analysis']
        print(f"   {exp_name}: {result['filename']}")
        print(f"      -> 실제 노이즈: {analysis['actual_changes']:,}개 ({analysis['actual_noise_ratio'] * 100:.1f}%)")

    # 메타데이터 저장
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset_name': dataset_name,
        'total_samples': len(df),
        'experiments': len(results),
        'processing_time_minutes': total_time / 60,
        'results': {k: {
            'filename': v['filename'],
            'noise_ratio': v['noise_ratio'],
            'strategy': v['strategy'],
            'actual_changes': v['analysis']['actual_changes'],
            'processing_time': v['processing_time']
        } for k, v in results.items()}
    }

    import json
    metadata_file = f"experiment_metadata_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join("./data", metadata_file), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n실험 메타데이터 저장: {metadata_file}")

    print_separator(f"{dataset_name.upper()} DataInf 실험 준비 완료", "=")

    return True


def parse_noise_ratios(ratios_str):
    """노이즈 비율 문자열 파싱"""
    try:
        ratios = [float(r.strip()) for r in ratios_str.split(',')]
        for r in ratios:
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"노이즈 비율은 0.0~1.0 사이여야 합니다: {r}")
        return ratios
    except ValueError as e:
        print(f"노이즈 비율 파싱 오류: {e}")
        print("   예시: --noise-ratios 0.1,0.2,0.3")
        sys.exit(1)


def parse_strategies(strategy_str):
    """노이즈 전략 문자열 파싱"""
    available_strategies = ['balanced', 'grammar_heavy', 'semantic_heavy']

    if strategy_str == 'all':
        return available_strategies

    strategies = [s.strip() for s in strategy_str.split(',')]

    for strategy in strategies:
        if strategy not in available_strategies:
            print(f"알 수 없는 전략: {strategy}")
            print(f"   사용 가능한 전략: {', '.join(available_strategies)}, all")
            print(f"   예시: balanced,grammar_heavy 또는 all")
            sys.exit(1)

    return strategies


def run_cache_management():
    """캐시 관리 모드"""
    loader = MultiDatasetLoader()

    print("=== 캐시 관리 메뉴 ===")
    print("지원하는 데이터셋: alpaca, gsm8k, sst2, mrpc")
    print("1. 모든 캐시 정보 확인")
    print("2. 특정 데이터셋 캐시 삭제")
    print("3. 모든 캐시 삭제")
    print("4. 특정 데이터셋 강제 재다운로드")

    choice = input("선택하세요 (1-4): ").strip()

    if choice == "1":
        # 모든 캐시 정보 확인
        for dataset_name in ['alpaca', 'gsm8k', 'sst2', 'mrpc']:
            cache_file = loader.supported_datasets[dataset_name]['cache_file']
            cache_path = os.path.join(loader.data_dir, cache_file)
            print(f"\n{dataset_name.upper()}:")
            if os.path.exists(cache_path):
                file_size = os.path.getsize(cache_path) / (1024 * 1024)
                import time
                mod_time = time.ctime(os.path.getmtime(cache_path))
                print(f"   - 상태: 존재")
                print(f"   - 크기: {file_size:.2f} MB")
                print(f"   - 수정일: {mod_time}")
            else:
                print(f"   - 상태: 없음")

    elif choice == "2":
        dataset_name = input("삭제할 데이터셋 (alpaca/gsm8k/sst2/mrpc): ").strip()
        if dataset_name in loader.supported_datasets:
            cache_file = loader.supported_datasets[dataset_name]['cache_file']
            cache_path = os.path.join(loader.data_dir, cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"{dataset_name} 캐시 파일이 삭제되었습니다.")
            else:
                print(f"{dataset_name} 캐시 파일이 없습니다.")
        else:
            print("올바르지 않은 데이터셋 이름입니다.")

    elif choice == "3":
        for dataset_name in ['alpaca', 'gsm8k', 'sst2', 'mrpc']:
            cache_file = loader.supported_datasets[dataset_name]['cache_file']
            cache_path = os.path.join(loader.data_dir, cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"{dataset_name} 캐시 삭제됨")
        print("모든 캐시가 삭제되었습니다.")

    elif choice == "4":
        dataset_name = input("재다운로드할 데이터셋 (alpaca/gsm8k/sst2/mrpc): ").strip()
        if dataset_name in loader.supported_datasets:
            print(f"{dataset_name} 강제 재다운로드를 시작합니다...")
            df = loader.load_dataset(dataset_name, subset_size=100, force_download=True)
            if df is not None:
                print("재다운로드 완료!")
        else:
            print("올바르지 않은 데이터셋 이름입니다.")
    else:
        print("올바르지 않은 선택입니다.")


def main():
    parser = argparse.ArgumentParser(
        description='DataInf 다중 데이터셋 노이즈 주입 실험 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 데모 모드 (빠른 테스트)
  python main.py --demo --dataset alpaca
  python main.py --demo --dataset gsm8k
  python main.py --demo --dataset sst2

  # 전체 데이터로 특정 실험
  python main.py --full --dataset gsm8k --noise-ratio 0.2 --strategy balanced
  python main.py --full --dataset sst2 --noise-ratio 0.15 --strategy grammar_heavy

  # 여러 비율/전략 한번에
  python main.py --full --dataset alpaca --noise-ratios 0.1,0.2,0.3 --strategy all
  python main.py --full --dataset gsm8k --noise-ratio 0.2 --strategy balanced,grammar_heavy

  # 기타 옵션
  python main.py --cache      # 캐시 관리
  python main.py --analysis   # 강화된 품질 분석

지원 데이터셋:
  - alpaca: Stanford Alpaca (instruction-following, ~52K)
  - gsm8k: Grade School Math 8K (math problems, ~7.5K)  
  - sst2: Stanford Sentiment Treebank (sentiment classification, ~67K)
  - mrpc: Microsoft Research Paraphrase Corpus (paraphrase detection, ~3.7K)
        """
    )

    # 메인 모드 선택
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--demo', action='store_true',
                            help='데모 모드 (500개 샘플로 빠른 테스트)')
    mode_group.add_argument('--full', action='store_true',
                            help='전체 데이터 모드 (실제 실험용)')
    mode_group.add_argument('--cache', action='store_true',
                            help='캐시 관리')
    mode_group.add_argument('--analysis', action='store_true',
                            help='강화된 품질 분석 (6가지 분석 옵션)')

    # 데이터셋 선택
    parser.add_argument('--dataset', type=str, default='alpaca',
                        choices=['alpaca', 'gsm8k', 'sst2', 'mrpc'],
                        help='데이터셋 선택 (기본값: alpaca)')

    # 전체 모드용 옵션들
    parser.add_argument('--noise-ratio', type=float, default=0.2,
                        help='노이즈 비율 (0.0~1.0, 기본값: 0.2)')
    parser.add_argument('--noise-ratios', type=str,
                        help='여러 노이즈 비율 (쉼표 구분, 예: 0.1,0.2,0.3)')
    parser.add_argument('--strategy', type=str, default='balanced',
                        help='노이즈 전략 (balanced, grammar_heavy, semantic_heavy, all 또는 콤마로 구분, 기본값: balanced)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='확인 없이 바로 실행')
    parser.add_argument('--flip-labels', action='store_true',
                        help='라벨 플리핑 모드 (SST-2, MRPC에서만 지원)')

    args = parser.parse_args()

    # 데이터 디렉토리 확인
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 모드별 실행
    try:
        if args.demo:
            success = run_demo_mode(args)
        elif args.full:
            success = run_full_mode(args)
        elif args.cache:
            run_cache_management()
            return
        elif args.analysis:
            run_quality_analysis()
            return

        if success:
            print("\n실행이 성공적으로 완료되었습니다!")
        else:
            print("\n실행 중 오류가 발생했습니다.")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()