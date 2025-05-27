"""
DataInf 데이터셋 품질 분석 모듈

이 모듈은 생성된 데이터셋들의 품질을 종합적으로 분석하는 기능을 제공합니다.
"""

import os
import time
import random
import re
from .noise_injection import analyze_noise_distribution, compare_samples


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


def run_quality_analysis():
    """강화된 품질 분석 모드"""
    print_separator("데이터셋 품질 분석", "=", 70)

    from .data_loader import AlpacaDataLoader
    loader = AlpacaDataLoader()

    # 기존 파일들 찾기
    data_files = [f for f in os.listdir("./data") if f.startswith("alpaca_") and f.endswith(".json")]

    if not data_files:
        print("분석할 데이터 파일이 없습니다.")
        return

    # 파일 분류 및 기본 정보
    original_files = [f for f in data_files if "original" in f]
    demo_files = [f for f in data_files if "demo" in f and "original" not in f]
    full_files = [f for f in data_files if "full" in f and "original" not in f]

    print(f"발견된 데이터 파일: {len(data_files)}개")
    print(f"   - 원본 파일: {len(original_files)}개")
    print(f"   - 데모 파일: {len(demo_files)}개")
    print(f"   - 전체 실험 파일: {len(full_files)}개")

    # 분석 옵션 선택
    print(f"\n분석 옵션:")
    print("1. 파일 기본 정보")
    print("2. 데모 파일 상세 분석")
    print("3. 전체 파일 상세 분석 (시간 소요)")
    print("4. 파일 간 비교 분석")
    print("5. 노이즈 전략별 효과 분석")
    print("6. 전체 종합 분석")

    choice = input("\n선택하세요 (1-6, 또는 Enter로 기본 정보): ").strip()

    if choice == "1" or choice == "":
        analyze_file_basic_info(data_files)
    elif choice == "2":
        analyze_demo_files_detailed(demo_files, original_files, loader)
    elif choice == "3":
        analyze_full_files_detailed(full_files, original_files, loader)
    elif choice == "4":
        analyze_file_comparison(data_files, loader)
    elif choice == "5":
        analyze_strategy_effects(data_files, loader)
    elif choice == "6":
        run_comprehensive_analysis(data_files, loader)
    else:
        print("올바르지 않은 선택입니다.")


def analyze_file_basic_info(data_files):
    """파일 기본 정보 분석"""
    print_separator("파일 기본 정보", "-")

    file_info = []
    for file in sorted(data_files):
        filepath = os.path.join("./data", file)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB

        # 파일명에서 정보 추출
        info = parse_filename_info(file)
        info['filename'] = file
        info['size_mb'] = file_size
        file_info.append(info)

    # 표 형태로 출력
    print(f"{'파일명':<50} {'타입':<12} {'샘플수':<8} {'노이즈%':<8} {'전략':<15} {'크기(MB)':<10}")
    print("=" * 110)

    for info in file_info:
        print(f"{info['filename']:<50} {info['type']:<12} {info['samples']:<8} "
              f"{info['noise_percent']:<8} {info['strategy']:<15} {info['size_mb']:<10.1f}")


def parse_filename_info(filename):
    """파일명에서 정보 추출"""
    info = {
        'type': 'unknown',
        'samples': 'N/A',
        'noise_percent': 'N/A',
        'strategy': 'N/A'
    }

    # 타입 판별
    if 'original' in filename:
        info['type'] = 'original'
    elif 'demo' in filename:
        info['type'] = 'demo'
    elif 'full' in filename:
        info['type'] = 'experiment'

    # 샘플 수 추출
    sample_match = re.search(r'_(\d+)\.json$', filename)
    if sample_match:
        info['samples'] = sample_match.group(1)

    # 노이즈 비율 추출
    noise_match = re.search(r'_(\d+)percent_', filename)
    if noise_match:
        info['noise_percent'] = f"{noise_match.group(1)}%"

    # 전략 추출
    if 'balanced' in filename:
        info['strategy'] = 'balanced'
    elif 'grammar' in filename:
        info['strategy'] = 'grammar_heavy'
    elif 'semantic' in filename:
        info['strategy'] = 'semantic_heavy'

    return info


def analyze_demo_files_detailed(demo_files, original_files, loader):
    """데모 파일 상세 분석"""
    print_separator("데모 파일 상세 분석", "-")

    if not demo_files:
        print("분석할 데모 파일이 없습니다.")
        return

    # 원본 데모 파일 찾기
    original_demo = [f for f in original_files if 'demo' in f]
    if not original_demo:
        print("원본 데모 파일을 찾을 수 없습니다.")
        return

    print(f"원본 파일: {original_demo[0]}")

    # 원본 데이터 로드
    original_df = loader.load_saved_dataset(original_demo[0])
    if original_df is None:
        return

    print(f"원본 데이터 로드: {len(original_df)}개 샘플")

    # 각 노이즈 파일 분석
    for demo_file in sorted(demo_files):
        print(f"\n분석 중: {demo_file}")

        # 노이즈 파일 로드
        noisy_df = loader.load_saved_dataset(demo_file)
        if noisy_df is None:
            continue

        # 노이즈 인덱스 추정
        noisy_indices = []
        for i in range(len(original_df)):
            if (original_df.iloc[i]['instruction'] != noisy_df.iloc[i]['instruction'] or
                    original_df.iloc[i]['output'] != noisy_df.iloc[i]['output']):
                noisy_indices.append(i)

        # 분석 실행
        analysis = analyze_noise_distribution(noisy_df, original_df, noisy_indices)

        # 결과 출력
        print(f"    분석 결과:")
        print(f"      - 전체 샘플: {analysis['total_samples']}개")
        print(f"      - 노이즈 대상: {analysis['noisy_samples']}개")
        print(f"      - 실제 변경: {analysis['actual_changes']}개 ({analysis['actual_noise_ratio'] * 100:.1f}%)")
        print(f"      - 평균 길이 변화: {analysis['avg_length_change']:.1f} 문자")
        print(
            f"      - 필드별 변경: Instruction {analysis['field_changes']['instruction']}개, Output {analysis['field_changes']['output']}개")

        # 변경 유형 분석
        change_types = analysis['change_types']
        total_changes = sum(change_types.values())
        if total_changes > 0:
            print(f"      - 변경 유형:")
            for change_type, count in change_types.items():
                percentage = (count / total_changes) * 100
                print(f"        * {change_type}: {count}개 ({percentage:.1f}%)")

        # 샘플 비교
        if len(noisy_indices) >= 2:
            print(f"    샘플 비교 (2개):")
            compare_samples(original_df, noisy_df, noisy_indices[:2], num_examples=2)


def analyze_full_files_detailed(full_files, original_files, loader):
    """전체 파일 상세 분석 (시간 소요)"""
    print_separator("전체 파일 상세 분석", "-")

    if not full_files:
        print("분석할 전체 파일이 없습니다.")
        return

    print("전체 파일 분석은 시간이 오래 걸릴 수 있습니다.")
    confirm = input("계속 진행하시겠습니까? (y/N): ").strip().lower()

    if confirm != 'y':
        print("분석이 취소되었습니다.")
        return

    # 원본 전체 파일 찾기
    original_full = [f for f in original_files if 'full' in f]
    if not original_full:
        print("원본 전체 파일을 찾을 수 없습니다.")
        return

    print(f"원본 파일: {original_full[0]} (로딩 중...)")

    # 원본 데이터 로드
    start_time = time.time()
    original_df = loader.load_saved_dataset(original_full[0])
    load_time = time.time() - start_time

    if original_df is None:
        return

    print(f"원본 데이터 로드: {len(original_df)}개 샘플 ({load_time:.1f}초)")

    # 각 실험 파일 분석
    analysis_results = {}

    for full_file in sorted(full_files):
        print(f"\n분석 중: {full_file}")
        file_start_time = time.time()

        # 노이즈 파일 로드
        noisy_df = loader.load_saved_dataset(full_file)
        if noisy_df is None:
            continue

        # 변경된 인덱스 찾기 (샘플링으로 빠르게)
        sample_size = min(1000, len(original_df))  # 1000개 샘플로 추정
        sample_indices = random.sample(range(len(original_df)), sample_size)

        noisy_indices_sample = []
        for i in sample_indices:
            if (original_df.iloc[i]['instruction'] != noisy_df.iloc[i]['instruction'] or
                    original_df.iloc[i]['output'] != noisy_df.iloc[i]['output']):
                noisy_indices_sample.append(i)

        # 전체 비율로 추정
        estimated_noise_ratio = len(noisy_indices_sample) / len(sample_indices)
        estimated_total_noisy = int(len(original_df) * estimated_noise_ratio)

        file_analysis_time = time.time() - file_start_time

        # 결과 저장
        file_info = parse_filename_info(full_file)
        analysis_results[full_file] = {
            'file_info': file_info,
            'estimated_noise_ratio': estimated_noise_ratio,
            'estimated_total_noisy': estimated_total_noisy,
            'sample_size': sample_size,
            'analysis_time': file_analysis_time
        }

        # 결과 출력
        print(f"    추정 결과 (샘플 {sample_size}개 기준):")
        print(f"      - 추정 노이즈 비율: {estimated_noise_ratio * 100:.1f}%")
        print(f"      - 추정 변경 샘플: {estimated_total_noisy:,}개")
        print(f"      - 분석 시간: {file_analysis_time:.1f}초")

    # 요약 출력
    print_separator("전체 파일 분석 요약", "-")
    print(f"{'파일명':<50} {'설정 노이즈%':<12} {'실제 노이즈%':<12} {'변경 샘플수':<12}")
    print("=" * 90)

    for filename, result in analysis_results.items():
        file_info = result['file_info']
        print(f"{filename:<50} {file_info['noise_percent']:<12} "
              f"{result['estimated_noise_ratio'] * 100:<11.1f}% {result['estimated_total_noisy']:<12,}")


def analyze_file_comparison(data_files, loader):
    """파일 간 비교 분석"""
    print_separator("파일 간 비교 분석", "-")

    # 사용자가 비교할 파일 선택
    print("비교 가능한 파일:")
    for i, file in enumerate(sorted(data_files), 1):
        print(f"{i}. {file}")

    try:
        choice1 = int(input("첫 번째 파일 번호: ")) - 1
        choice2 = int(input("두 번째 파일 번호: ")) - 1

        file1 = sorted(data_files)[choice1]
        file2 = sorted(data_files)[choice2]

        print(f"\n비교 대상:")
        print(f"  파일 1: {file1}")
        print(f"  파일 2: {file2}")

        # 데이터 로드 및 기본 비교
        df1 = loader.load_saved_dataset(file1)
        df2 = loader.load_saved_dataset(file2)

        if df1 is None or df2 is None:
            print("파일 로드 실패")
            return

        # 기본 통계 비교
        print(f"\n기본 통계 비교:")
        print(f"{'구분':<20} {'파일 1':<15} {'파일 2':<15} {'차이':<15}")
        print("-" * 70)

        # 샘플 수
        print(f"{'샘플 수':<20} {len(df1):<15,} {len(df2):<15,} {len(df2) - len(df1):<15,}")

        # 평균 길이
        avg_len1 = df1['output'].str.len().mean()
        avg_len2 = df2['output'].str.len().mean()
        print(f"{'평균 Output 길이':<20} {avg_len1:<15.1f} {avg_len2:<15.1f} {avg_len2 - avg_len1:<15.1f}")

        # 고유값 비교
        if len(df1) == len(df2):
            differences = 0
            for i in range(len(df1)):
                if (df1.iloc[i]['instruction'] != df2.iloc[i]['instruction'] or
                        df1.iloc[i]['output'] != df2.iloc[i]['output']):
                    differences += 1

            print(f"{'다른 샘플 수':<20} {'-':<15} {'-':<15} {differences:<15,}")
            print(f"{'차이 비율':<20} {'-':<15} {'-':<15} {differences / len(df1) * 100:<15.1f}%")

    except (ValueError, IndexError):
        print("잘못된 선택입니다.")


def analyze_strategy_effects(data_files, loader):
    """노이즈 전략별 효과 분석"""
    print_separator("노이즈 전략별 효과 분석", "-")

    # 전략별 파일 그룹핑
    strategies = {'balanced': [], 'grammar_heavy': [], 'semantic_heavy': []}

    for file in data_files:
        if 'original' in file:
            continue
        for strategy in strategies.keys():
            if strategy.replace('_', '') in file:
                strategies[strategy].append(file)
                break

    # 결과 출력
    print("전략별 파일 분포:")
    for strategy, files in strategies.items():
        print(f"  {strategy}: {len(files)}개 파일")
        for file in files:
            file_info = parse_filename_info(file)
            print(f"    - {file} (노이즈: {file_info['noise_percent']}, 샘플: {file_info['samples']})")

    # 전략별 효과 비교 (데모 파일 기준)
    demo_strategies = {k: [f for f in v if 'demo' in f] for k, v in strategies.items()}
    demo_strategies = {k: v for k, v in demo_strategies.items() if v}

    if len(demo_strategies) > 1:
        print(f"\n데모 파일 기준 전략 효과 비교:")

        # 원본 데모 파일 찾기
        original_demo = [f for f in data_files if 'original' in f and 'demo' in f]
        if original_demo:
            original_df = loader.load_saved_dataset(original_demo[0])
            if original_df is not None:
                strategy_results = {}

                for strategy, files in demo_strategies.items():
                    if not files:
                        continue

                    file = files[0]  # 첫 번째 파일 사용
                    noisy_df = loader.load_saved_dataset(file)
                    if noisy_df is None:
                        continue

                    # 변경 분석
                    changes = 0
                    total_length_change = 0

                    for i in range(len(original_df)):
                        if (original_df.iloc[i]['output'] != noisy_df.iloc[i]['output']):
                            changes += 1
                            total_length_change += len(noisy_df.iloc[i]['output']) - len(original_df.iloc[i]['output'])

                    strategy_results[strategy] = {
                        'changes': changes,
                        'change_ratio': changes / len(original_df),
                        'avg_length_change': total_length_change / changes if changes > 0 else 0
                    }

                # 결과 출력
                print(f"{'전략':<15} {'변경 샘플':<12} {'변경 비율':<12} {'평균 길이 변화':<15}")
                print("-" * 60)

                for strategy, result in strategy_results.items():
                    print(f"{strategy:<15} {result['changes']:<12} "
                          f"{result['change_ratio'] * 100:<11.1f}% {result['avg_length_change']:<15.1f}")


def run_comprehensive_analysis(data_files, loader):
    """전체 종합 분석"""
    print_separator("전체 종합 분석", "=", 70)

    print("종합 분석을 시작합니다...")
    print("   이 분석은 모든 데이터를 종합적으로 분석하며 시간이 걸릴 수 있습니다.")

    confirm = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
    if confirm != 'y':
        return

    # 1. 파일 기본 정보
    print("\n" + "=" * 50)
    print("1. 파일 기본 정보")
    print("=" * 50)
    analyze_file_basic_info(data_files)

    # 2. 데모 파일 분석
    demo_files = [f for f in data_files if "demo" in f and "original" not in f]
    original_files = [f for f in data_files if "original" in f]

    if demo_files:
        print("\n" + "=" * 50)
        print("2. 데모 파일 상세 분석")
        print("=" * 50)
        analyze_demo_files_detailed(demo_files, original_files, loader)

    # 3. 전략별 효과 분석
    print("\n" + "=" * 50)
    print("3. 전략별 효과 분석")
    print("=" * 50)
    analyze_strategy_effects(data_files, loader)

    # 4. 최종 요약
    print("\n" + "=" * 50)
    print("4. 최종 요약")
    print("=" * 50)

    total_files = len(data_files)
    original_count = len([f for f in data_files if 'original' in f])
    demo_count = len([f for f in data_files if 'demo' in f and 'original' not in f])
    full_count = len([f for f in data_files if 'full' in f and 'original' not in f])

    print(f"전체 현황:")
    print(f"  - 총 파일 수: {total_files}개")
    print(f"  - 원본 파일: {original_count}개")
    print(f"  - 데모 실험: {demo_count}개")
    print(f"  - 전체 실험: {full_count}개")

    # 사용 가능한 전략 요약
    strategies_found = set()
    for file in data_files:
        if 'balanced' in file:
            strategies_found.add('balanced')
        elif 'grammar' in file:
            strategies_found.add('grammar_heavy')
        elif 'semantic' in file:
            strategies_found.add('semantic_heavy')

    print(f"  - 테스트된 전략: {', '.join(strategies_found)}")

    # DataInf 실험 준비도 체크
    print(f"\nDataInf 실험 준비도:")

    has_original = original_count > 0
    has_noisy = (demo_count + full_count) > 0
    has_multiple_ratios = len(set(parse_filename_info(f)['noise_percent']
                                  for f in data_files if parse_filename_info(f)['noise_percent'] != 'N/A')) > 1

    print(f"  - 원본 데이터: {'O' if has_original else 'X'}")
    print(f"  - 노이즈 데이터: {'O' if has_noisy else 'X'}")
    print(f"  - 다양한 노이즈 비율: {'O' if has_multiple_ratios else 'X'}")

    readiness = sum([has_original, has_noisy, has_multiple_ratios])
    if readiness == 3:
        print(f"  준비도: 완벽 (3/3) - DataInf 실험 진행 가능!")
    elif readiness == 2:
        print(f"  준비도: 양호 (2/3) - 추가 데이터 생성 권장")
    else:
        print(f"  준비도: 부족 ({readiness}/3) - 데이터 생성 필요")

    print(f"\n종합 분석이 완료되었습니다!")


if __name__ == "__main__":
    # 테스트 실행 (직접 실행시)
    print("=== Analysis 모듈 테스트 ===")
    run_quality_analysis()