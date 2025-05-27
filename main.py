import os
import sys
import argparse
import time
from datetime import datetime
from src.data_loader import AlpacaDataLoader
from src.noise_injection import NoiseInjector, compare_samples, analyze_noise_distribution
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
    print_separator("DataInf 데모 모드", "=", 70)
    print("목적: 빠른 프로토타입 및 기능 테스트")
    print("샘플 크기: 500개")
    print("예상 시간: 1-2분")
    print()

    # 데이터 로딩
    loader = AlpacaDataLoader()
    SAMPLE_SIZE = 500

    print("데이터 로딩 중...")
    df = loader.load_alpaca_dataset(subset_size=SAMPLE_SIZE)

    if df is None:
        print("데이터 로딩 실패")
        return False

    # 원본 데이터 저장
    original_filename = f"alpaca_original_demo_{SAMPLE_SIZE}.json"
    loader.save_dataset(df, original_filename)

    # 데모용 노이즈 실험
    injector = NoiseInjector(random_seed=42)

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
            noise_ratio=ratio,
            noise_strategy=strategy
        )

        # 파일 저장
        filename = f"alpaca_demo_{ratio * 100:.0f}percent_{strategy}_{SAMPLE_SIZE}.json"
        loader.save_dataset(noisy_df, filename)

        # 분석
        analysis = analyze_noise_distribution(noisy_df, df, noisy_indices)
        print(f"   완료: 실제 변경 {analysis['actual_changes']}/{len(noisy_indices)}개")

    # 샘플 비교
    print_separator("결과 비교", "-")
    final_noisy_df, final_noisy_indices = injector.inject_noise(df, noise_ratio=0.2, noise_strategy="balanced")
    compare_samples(df, final_noisy_df, final_noisy_indices, num_examples=2)

    print(f"\n데모 완료! 생성된 파일들을 data/ 폴더에서 확인하세요.")
    return True


def run_full_mode(args):
    """전체 데이터 모드 (실제 실험용)"""
    print_separator("DataInf 전체 데이터 모드", "=", 70)
    print("목적: 실제 DataInf 실험용 데이터셋 생성")
    print("샘플 크기: 전체 52,002개")
    print("예상 시간: 10-30분 (옵션에 따라)")
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
    loader = AlpacaDataLoader()

    print("전체 데이터 로딩 중...")
    start_time = time.time()
    df = loader.load_alpaca_dataset()  # 전체 데이터
    load_time = time.time() - start_time

    if df is None:
        print("데이터 로딩 실패")
        return False

    print(f"데이터 로딩 완료 ({len(df):,}개, {load_time:.1f}초)")

    # 원본 데이터 저장
    original_filename = f"alpaca_original_full_{len(df)}.json"
    print(f"원본 데이터 저장 중...")
    loader.save_dataset(df, original_filename)

    # 실험 실행
    injector = NoiseInjector(random_seed=42)
    results = {}

    print_separator("노이즈 주입 실험 시작", "=")

    experiment_count = 0
    for ratio in noise_ratios:
        for strategy in strategies:
            experiment_count += 1

            print(f"\n실험 {experiment_count}/{total_experiments}: {strategy} 전략, {ratio * 100:.0f}% 노이즈")
            print(f"   예상 노이즈 샘플 수: {int(len(df) * ratio):,}개")

            exp_start_time = time.time()

            # 노이즈 주입
            noisy_df, noisy_indices = injector.inject_noise(
                df.copy(),
                noise_ratio=ratio,
                noise_strategy=strategy
            )

            # 파일 저장
            if strategy == "balanced":
                filename = f"alpaca_full_{ratio * 100:.0f}percent_{len(df)}.json"
            else:
                strategy_short = strategy.replace('_heavy', '').replace('_', '')
                filename = f"alpaca_full_{ratio * 100:.0f}percent_{strategy_short}_{len(df)}.json"

            print(f"   파일 저장 중: {filename}")
            saved_path = loader.save_dataset(noisy_df, filename)

            # 분석
            analysis = analyze_noise_distribution(noisy_df, df, noisy_indices)

            exp_time = time.time() - exp_start_time
            print(f"   완료 ({exp_time:.1f}초)")
            print(f"      - 실제 변경: {analysis['actual_changes']:,}/{len(noisy_indices):,}개")
            print(f"      - 평균 길이 변화: {analysis['avg_length_change']:.1f} 문자")

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
    print(f"모든 실험 완료! (총 소요시간: {total_time / 60:.1f}분)")
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
    metadata_file = f"experiment_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join("./data", metadata_file), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n실험 메타데이터 저장: {metadata_file}")

    print_separator("DataInf 실험 준비 완료", "=")

    return True


def parse_noise_ratios(ratios_str):
    """노이즈 비율 문자열 파싱"""
    try:
        ratios = [float(r.strip()) for r in ratios_str.split(',')]
        # 0.0 ~ 1.0 범위 체크
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

    # 콤마로 구분된 전략들 파싱
    strategies = [s.strip() for s in strategy_str.split(',')]

    # 각 전략 유효성 검사
    for strategy in strategies:
        if strategy not in available_strategies:
            print(f"알 수 없는 전략: {strategy}")
            print(f"   사용 가능한 전략: {', '.join(available_strategies)}, all")
            print(f"   예시: balanced,grammar_heavy 또는 all")
            sys.exit(1)

    return strategies


def run_cache_management():
    """캐시 관리 모드"""
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
            print("재다운로드 완료!")
    else:
        print("올바르지 않은 선택입니다.")


def main():
    parser = argparse.ArgumentParser(
        description='DataInf 노이즈 주입 실험 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 데모 모드 (빠른 테스트)
  python main.py --demo

  # 전체 데이터로 특정 실험
  python main.py --full --noise-ratio 0.2 --strategy balanced
  python main.py --full --noise-ratio 0.15 --strategy grammar_heavy

  # 여러 비율/전략 한번에
  python main.py --full --noise-ratios 0.1,0.2,0.3 --strategy all
  python main.py --full --noise-ratio 0.2 --strategy balanced,grammar_heavy
  python main.py --full --noise-ratios 0.1,0.2 --strategy grammar_heavy,semantic_heavy

  # 기타 옵션
  python main.py --cache      # 캐시 관리
  python main.py --analysis   # 강화된 품질 분석
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

    # 전체 모드용 옵션들
    parser.add_argument('--noise-ratio', type=float, default=0.2,
                        help='노이즈 비율 (0.0~1.0, 기본값: 0.2)')
    parser.add_argument('--noise-ratios', type=str,
                        help='여러 노이즈 비율 (쉼표 구분, 예: 0.1,0.2,0.3)')
    parser.add_argument('--strategy', type=str, default='balanced',
                        help='노이즈 전략 (balanced, grammar_heavy, semantic_heavy, all 또는 콤마로 구분, 기본값: balanced)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='확인 없이 바로 실행')

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