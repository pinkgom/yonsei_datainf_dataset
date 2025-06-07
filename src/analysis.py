"""
DataInf 다중 데이터셋 품질 분석 모듈

이 모듈은 다양한 데이터셋(alpaca, gsm8k, sst2, mrpc)으로 생성된
노이즈 데이터셋들의 품질을 종합적으로 분석하는 기능을 제공합니다.
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
    """다중 데이터셋 품질 분석 모드"""
    print_separator("다중 데이터셋 품질 분석", "=", 70)

    from .data_loader import MultiDatasetLoader
    loader = MultiDatasetLoader()

    # 기존 파일들 찾기 및 데이터셋별 분류
    data_files = [f for f in os.listdir("./data") if f.endswith(".json")]

    if not data_files:
        print("분석할 데이터 파일이 없습니다.")
        return

    # 데이터셋별로 파일 분류
    datasets = {
        'alpaca': [],
        'gsm8k': [],
        'sst2': [],
        'mrpc': []
    }

    other_files = []

    for file in data_files:
        classified = False
        for dataset_name in datasets.keys():
            if file.startswith(dataset_name):
                datasets[dataset_name].append(file)
                classified = True
                break
        if not classified:
            other_files.append(file)

    # 전체 파일 현황
    total_files = sum(len(files) for files in datasets.values()) + len(other_files)
    print(f"발견된 데이터 파일: {total_files}개")

    for dataset_name, files in datasets.items():
        if files:
            original_count = len([f for f in files if "original" in f])
            experiment_count = len(files) - original_count
            print(f"   - {dataset_name.upper()}: {len(files)}개 (원본: {original_count}, 실험: {experiment_count})")

    if other_files:
        print(f"   - 기타: {len(other_files)}개")

    # 분석 옵션 선택
    print(f"\n분석 옵션:")
    print("1. 전체 파일 기본 정보")
    print("2. 데이터셋별 상세 분석")
    print("3. 데이터셋간 노이즈 효과 비교")
    print("4. 특정 데이터셋 집중 분석")
    print("5. 라벨 보존 검증 (Classification 데이터셋)")
    print("6. 전체 종합 분석")

    choice = input("\n선택하세요 (1-6, 또는 Enter로 기본 정보): ").strip()

    if choice == "1" or choice == "":
        analyze_all_files_basic_info(datasets, other_files)
    elif choice == "2":
        analyze_by_dataset_detailed(datasets, loader)
    elif choice == "3":
        analyze_cross_dataset_comparison(datasets, loader)
    elif choice == "4":
        analyze_specific_dataset(datasets, loader)
    elif choice == "5":
        analyze_label_preservation(datasets, loader)
    elif choice == "6":
        run_comprehensive_multi_dataset_analysis(datasets, loader)
    else:
        print("올바르지 않은 선택입니다.")


def analyze_all_files_basic_info(datasets, other_files):
    """전체 파일 기본 정보 분석"""
    print_separator("전체 파일 기본 정보", "-")

    # 테이블 헤더
    print(f"{'파일명':<60} {'데이터셋':<10} {'타입':<12} {'샘플수':<8} {'노이즈%':<8} {'전략':<15} {'크기(MB)':<10}")
    print("=" * 130)

    # 데이터셋별 파일 정보
    for dataset_name, files in datasets.items():
        if not files:
            continue

        for file in sorted(files):
            info = parse_filename_info(file, dataset_name)
            filepath = os.path.join("./data", file)
            file_size = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0

            print(f"{file:<60} {dataset_name:<10} {info['type']:<12} {info['samples']:<8} "
                  f"{info['noise_percent']:<8} {info['strategy']:<15} {file_size:<10.1f}")

    # 기타 파일들
    if other_files:
        print("\n기타 파일들:")
        for file in sorted(other_files):
            filepath = os.path.join("./data", file)
            file_size = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
            print(f"{file:<60} {'unknown':<10} {'unknown':<12} {'N/A':<8} "
                  f"{'N/A':<8} {'N/A':<15} {file_size:<10.1f}")


def analyze_by_dataset_detailed(datasets, loader):
    """데이터셋별 상세 분석"""
    print_separator("데이터셋별 상세 분석", "-")

    for dataset_name, files in datasets.items():
        if not files:
            continue

        print(f"\n🔍 {dataset_name.upper()} 데이터셋 분석")
        print("=" * 50)

        # 원본 파일과 노이즈 파일 분리
        original_files = [f for f in files if "original" in f]
        noise_files = [f for f in files if "original" not in f]

        print(f"원본 파일: {len(original_files)}개")
        print(f"노이즈 파일: {len(noise_files)}개")

        if not original_files:
            print("   ⚠️  원본 파일이 없어 분석을 건너뜁니다.")
            continue

        # 대표 원본 파일 (가장 큰 것)
        main_original = max(original_files, key=lambda f: os.path.getsize(os.path.join("./data", f)))
        print(f"분석 기준 원본: {main_original}")

        # 원본 데이터 로드
        original_df = loader.load_saved_dataset(main_original)
        if original_df is None:
            print("   ⚠️  원본 파일 로드 실패")
            continue

        print(f"원본 데이터: {len(original_df):,}개 샘플")

        # 데이터셋별 특성 분석
        analyze_dataset_characteristics(original_df, dataset_name)

        # 노이즈 파일들 분석
        if noise_files:
            print(f"\n📊 노이즈 파일 분석:")

            for noise_file in sorted(noise_files)[:3]:  # 최대 3개만
                print(f"\n   분석 중: {noise_file}")

                noisy_df = loader.load_saved_dataset(noise_file)
                if noisy_df is None:
                    continue

                # 노이즈 인덱스 추정
                noisy_indices = estimate_noisy_indices(original_df, noisy_df, dataset_name)

                # 분석 실행
                analysis = analyze_noise_distribution(noisy_df, original_df, noisy_indices, dataset_name)

                # 결과 출력
                print(f"      - 전체 샘플: {analysis['total_samples']:,}개")
                print(f"      - 추정 노이즈: {len(noisy_indices):,}개 ({len(noisy_indices)/len(original_df)*100:.1f}%)")
                print(f"      - 실제 변경: {analysis['actual_changes']:,}개")
                print(f"      - 평균 길이 변화: {analysis['avg_length_change']:.1f} 문자")


def analyze_cross_dataset_comparison(datasets, loader):
    """데이터셋간 노이즈 효과 비교"""
    print_separator("데이터셋간 노이즈 효과 비교", "-")

    # 각 데이터셋의 20% balanced 노이즈 파일 찾기
    comparison_files = {}

    for dataset_name, files in datasets.items():
        if not files:
            continue

        # 20% balanced 파일 찾기
        target_files = [f for f in files if "20percent" in f and "balanced" in f and "original" not in f]
        if target_files:
            comparison_files[dataset_name] = target_files[0]

    if len(comparison_files) < 2:
        print("비교할 데이터셋이 부족합니다. (각 데이터셋에 20% balanced 노이즈 파일 필요)")
        return

    print(f"비교 대상: {list(comparison_files.keys())}")
    print(f"분석 조건: 20% balanced 노이즈")

    # 비교 결과 수집
    comparison_results = {}

    for dataset_name, noise_file in comparison_files.items():
        print(f"\n📊 {dataset_name.upper()} 분석 중...")

        # 원본 파일 찾기
        original_files = [f for f in datasets[dataset_name] if "original" in f]
        if not original_files:
            continue

        original_file = original_files[0]

        # 데이터 로드
        original_df = loader.load_saved_dataset(original_file)
        noisy_df = loader.load_saved_dataset(noise_file)

        if original_df is None or noisy_df is None:
            continue

        # 노이즈 인덱스 추정 및 분석
        noisy_indices = estimate_noisy_indices(original_df, noisy_df, dataset_name)
        analysis = analyze_noise_distribution(noisy_df, original_df, noisy_indices, dataset_name)

        comparison_results[dataset_name] = {
            'total_samples': len(original_df),
            'target_noise_samples': len(noisy_indices),
            'actual_changes': analysis['actual_changes'],
            'actual_noise_ratio': analysis['actual_noise_ratio'],
            'avg_length_change': analysis['avg_length_change'],
            'field_changes': analysis['field_changes']
        }

    # 비교 결과 출력
    print_separator("데이터셋간 비교 결과", "-")

    print(f"{'데이터셋':<10} {'총 샘플':<10} {'노이즈 대상':<12} {'실제 변경':<12} {'실제 비율':<10} {'길이 변화':<10}")
    print("-" * 70)

    for dataset_name, result in comparison_results.items():
        print(f"{dataset_name:<10} {result['total_samples']:<10,} {result['target_noise_samples']:<12,} "
              f"{result['actual_changes']:<12,} {result['actual_noise_ratio']*100:<9.1f}% {result['avg_length_change']:<10.1f}")

    # 노이즈 효과성 분석
    print(f"\n📈 노이즈 효과성 순위:")
    sorted_results = sorted(comparison_results.items(),
                          key=lambda x: x[1]['actual_noise_ratio'], reverse=True)

    for i, (dataset_name, result) in enumerate(sorted_results, 1):
        effectiveness = result['actual_changes'] / result['target_noise_samples'] * 100
        print(f"{i}. {dataset_name.upper()}: {effectiveness:.1f}% 효과성 "
              f"(목표 {result['target_noise_samples']:,}개 → 실제 {result['actual_changes']:,}개)")


def analyze_specific_dataset(datasets, loader):
    """특정 데이터셋 집중 분석"""
    print_separator("특정 데이터셋 집중 분석", "-")

    # 데이터셋 선택
    available_datasets = [name for name, files in datasets.items() if files]

    if not available_datasets:
        print("분석할 데이터셋이 없습니다.")
        return

    print("분석 가능한 데이터셋:")
    for i, dataset_name in enumerate(available_datasets, 1):
        file_count = len(datasets[dataset_name])
        print(f"{i}. {dataset_name.upper()} ({file_count}개 파일)")

    try:
        choice = int(input("선택하세요: ")) - 1
        selected_dataset = available_datasets[choice]
    except (ValueError, IndexError):
        print("잘못된 선택입니다.")
        return

    print(f"\n🎯 {selected_dataset.upper()} 집중 분석")
    print("=" * 50)

    files = datasets[selected_dataset]

    # 파일 분류
    original_files = [f for f in files if "original" in f]
    demo_files = [f for f in files if "demo" in f and "original" not in f]
    full_files = [f for f in files if "full" in f and "original" not in f]

    print(f"원본 파일: {len(original_files)}개")
    print(f"데모 파일: {len(demo_files)}개")
    print(f"전체 파일: {len(full_files)}개")

    # 원본 데이터 특성
    if original_files:
        main_original = original_files[0]
        original_df = loader.load_saved_dataset(main_original)
        if original_df is not None:
            print(f"\n📋 데이터셋 특성:")
            analyze_dataset_characteristics(original_df, selected_dataset)

    # 노이즈 전략별 분석
    if demo_files or full_files:
        print(f"\n📊 노이즈 전략별 효과:")

        target_files = demo_files if demo_files else full_files
        strategy_results = {}

        for file in target_files:
            info = parse_filename_info(file, selected_dataset)
            strategy = info['strategy']
            noise_percent = info['noise_percent']

            if strategy not in strategy_results:
                strategy_results[strategy] = []

            strategy_results[strategy].append({
                'file': file,
                'noise_percent': noise_percent
            })

        for strategy, file_info_list in strategy_results.items():
            print(f"\n   {strategy} 전략:")
            for file_info in file_info_list:
                print(f"      - {file_info['file']} (노이즈: {file_info['noise_percent']})")


def analyze_label_preservation(datasets, loader):
    """라벨 보존 검증 (Classification 데이터셋)"""
    print_separator("라벨 보존 검증", "-")

    # Classification 데이터셋만 선택
    classification_datasets = ['gsm8k', 'sst2', 'mrpc']
    available_classification = {name: files for name, files in datasets.items()
                              if name in classification_datasets and files}

    if not available_classification:
        print("라벨 보존 검증이 가능한 Classification 데이터셋이 없습니다.")
        print("필요한 데이터셋: gsm8k, sst2, mrpc")
        return

    print(f"검증 대상 데이터셋: {list(available_classification.keys())}")

    from .noise_injection import MultiDatasetNoiseInjector
    injector = MultiDatasetNoiseInjector()

    for dataset_name, files in available_classification.items():
        print(f"\n🔍 {dataset_name.upper()} 라벨 보존 검증")
        print("-" * 40)

        config = injector.dataset_configs.get(dataset_name, {})
        label_columns = config.get('label_columns', [])

        if not label_columns:
            print("   라벨 컬럼이 정의되지 않음")
            continue

        print(f"   검증 라벨: {label_columns}")

        # 원본 파일과 노이즈 파일 찾기
        original_files = [f for f in files if "original" in f]
        noise_files = [f for f in files if "original" not in f]

        if not original_files or not noise_files:
            print("   원본 또는 노이즈 파일이 없음")
            continue

        # 대표 파일들로 검증
        original_file = original_files[0]

        print(f"   원본 파일: {original_file}")

        original_df = loader.load_saved_dataset(original_file)
        if original_df is None:
            continue

        label_preservation_results = {}

        for noise_file in noise_files[:3]:  # 최대 3개 파일만 검증
            print(f"\n   검증 중: {noise_file}")

            noisy_df = loader.load_saved_dataset(noise_file)
            if noisy_df is None:
                continue

            # 라벨 변경 여부 확인
            label_changes = {}
            total_samples = min(len(original_df), len(noisy_df))

            for label_col in label_columns:
                if label_col in original_df.columns and label_col in noisy_df.columns:
                    changes = 0
                    for i in range(total_samples):
                        if original_df.iloc[i][label_col] != noisy_df.iloc[i][label_col]:
                            changes += 1
                    label_changes[label_col] = changes
                else:
                    label_changes[label_col] = "컬럼 없음"

            label_preservation_results[noise_file] = label_changes

            # 결과 출력
            for label_col, changes in label_changes.items():
                if isinstance(changes, int):
                    if changes == 0:
                        print(f"      ✅ {label_col}: 완벽 보존 (변경 0개)")
                    else:
                        print(f"      ❌ {label_col}: {changes}개 변경됨 ({changes/total_samples*100:.2f}%)")
                else:
                    print(f"      ⚠️  {label_col}: {changes}")

        # 요약
        if label_preservation_results:
            print(f"\n   📊 {dataset_name.upper()} 라벨 보존 요약:")
            perfect_preservation = 0
            total_files = len(label_preservation_results)

            for noise_file, label_changes in label_preservation_results.items():
                all_preserved = all(changes == 0 for changes in label_changes.values()
                                  if isinstance(changes, int))
                if all_preserved:
                    perfect_preservation += 1

            print(f"      완벽 보존 파일: {perfect_preservation}/{total_files}개")
            if perfect_preservation == total_files:
                print(f"      ✅ 모든 파일에서 라벨이 완벽하게 보존됨!")
            else:
                print(f"      ⚠️  일부 파일에서 라벨 변경 발생")


def run_comprehensive_multi_dataset_analysis(datasets, loader):
    """전체 종합 분석 (다중 데이터셋)"""
    print_separator("다중 데이터셋 종합 분석", "=", 70)

    print("종합 분석을 시작합니다...")
    print("   이 분석은 모든 데이터셋을 종합적으로 분석하며 시간이 걸릴 수 있습니다.")

    confirm = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
    if confirm != 'y':
        return

    # 1. 전체 현황
    print("\n" + "=" * 50)
    print("1. 전체 현황")
    print("=" * 50)
    analyze_all_files_basic_info(datasets, [])

    # 2. 데이터셋별 요약
    print("\n" + "=" * 50)
    print("2. 데이터셋별 요약")
    print("=" * 50)

    total_datasets = 0
    total_files = 0

    for dataset_name, files in datasets.items():
        if files:
            total_datasets += 1
            total_files += len(files)

            original_count = len([f for f in files if "original" in f])
            noise_count = len(files) - original_count

            print(f"{dataset_name.upper()}:")
            print(f"   - 총 파일: {len(files)}개")
            print(f"   - 원본: {original_count}개")
            print(f"   - 노이즈: {noise_count}개")

            # 노이즈 전략 분포
            strategies = set()
            noise_ratios = set()

            for file in files:
                if "original" not in file:
                    info = parse_filename_info(file, dataset_name)
                    if info['strategy'] != 'N/A':
                        strategies.add(info['strategy'])
                    if info['noise_percent'] != 'N/A':
                        noise_ratios.add(info['noise_percent'])

            if strategies:
                print(f"   - 테스트된 전략: {', '.join(strategies)}")
            if noise_ratios:
                print(f"   - 테스트된 노이즈 비율: {', '.join(sorted(noise_ratios))}")
            print()

    # 3. 데이터셋간 비교
    if total_datasets > 1:
        print("\n" + "=" * 50)
        print("3. 데이터셋간 비교")
        print("=" * 50)
        analyze_cross_dataset_comparison(datasets, loader)

    # 4. 라벨 보존 검증
    print("\n" + "=" * 50)
    print("4. 라벨 보존 검증")
    print("=" * 50)
    analyze_label_preservation(datasets, loader)

    # 5. 최종 요약
    print("\n" + "=" * 50)
    print("5. 최종 요약")
    print("=" * 50)

    print(f"전체 현황:")
    print(f"  - 활성 데이터셋: {total_datasets}개")
    print(f"  - 총 파일 수: {total_files}개")

    # DataInf 실험 준비도 체크
    print(f"\nDataInf 실험 준비도:")

    ready_datasets = []

    for dataset_name, files in datasets.items():
        if not files:
            continue

        has_original = any("original" in f for f in files)
        has_noise = any("original" not in f for f in files)
        has_multiple_ratios = len(set(parse_filename_info(f, dataset_name)['noise_percent']
                                    for f in files if parse_filename_info(f, dataset_name)['noise_percent'] != 'N/A')) > 1

        dataset_readiness = sum([has_original, has_noise, has_multiple_ratios])

        print(f"  {dataset_name.upper()}:")
        print(f"    - 원본 데이터: {'✅' if has_original else '❌'}")
        print(f"    - 노이즈 데이터: {'✅' if has_noise else '❌'}")
        print(f"    - 다양한 비율: {'✅' if has_multiple_ratios else '❌'}")

        if dataset_readiness == 3:
            print(f"    📊 준비도: 완벽 (3/3) - 실험 가능!")
            ready_datasets.append(dataset_name)
        elif dataset_readiness == 2:
            print(f"    📊 준비도: 양호 (2/3) - 추가 데이터 권장")
        else:
            print(f"    📊 준비도: 부족 ({dataset_readiness}/3) - 데이터 생성 필요")

    if ready_datasets:
        print(f"\n🎉 실험 준비 완료된 데이터셋: {', '.join(ready_datasets)}")
        print(f"   이제 LoRA 학습 담당자에게 데이터를 전달할 수 있습니다!")
    else:
        print(f"\n⚠️  아직 실험 준비가 완료된 데이터셋이 없습니다.")

    print(f"\n다중 데이터셋 종합 분석이 완료되었습니다!")


# 헬퍼 함수들

def parse_filename_info(filename, dataset_name):
    """데이터셋별 파일명 정보 추출"""
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


def estimate_noisy_indices(original_df, noisy_df, dataset_name):
    """데이터셋별 노이즈 인덱스 추정"""
    from .noise_injection import MultiDatasetNoiseInjector

    injector = MultiDatasetNoiseInjector()
    config = injector.dataset_configs.get(dataset_name, {})
    text_columns = config.get('text_columns', [])

    if not text_columns:
        # fallback: 모든 문자열 컬럼 비교
        text_columns = [col for col in original_df.columns if original_df[col].dtype == 'object']

    noisy_indices = []
    max_samples = min(len(original_df), len(noisy_df))

    for i in range(max_samples):
        for col in text_columns:
            if col in original_df.columns and col in noisy_df.columns:
                if str(original_df.iloc[i][col]) != str(noisy_df.iloc[i][col]):
                    noisy_indices.append(i)
                    break

    return noisy_indices


def analyze_dataset_characteristics(df, dataset_name):
    """데이터셋별 특성 분석"""
    print(f"   총 샘플 수: {len(df):,}개")
    print(f"   컬럼: {list(df.columns)}")

    if dataset_name == 'alpaca':
        if 'instruction' in df.columns:
            inst_lengths = df['instruction'].str.len()
            print(f"   Instruction 길이 - 평균: {inst_lengths.mean():.1f}, 범위: {inst_lengths.min()}-{inst_lengths.max()}")
        if 'output' in df.columns:
            out_lengths = df['output'].str.len()
            print(f"   Output 길이 - 평균: {out_lengths.mean():.1f}, 범위: {out_lengths.min()}-{out_lengths.max()}")

    elif dataset_name == 'gsm8k':
        if 'question' in df.columns:
            q_lengths = df['question'].str.len()
            print(f"   Question 길이 - 평균: {q_lengths.mean():.1f}, 범위: {q_lengths.min()}-{q_lengths.max()}")
        if 'answer' in df.columns:
            a_lengths = df['answer'].str.len()
            print(f"   Answer 길이 - 평균: {a_lengths.mean():.1f}, 범위: {a_lengths.min()}-{a_lengths.max()}")

    elif dataset_name == 'sst2':
        if 'sentence' in df.columns:
            sent_lengths = df['sentence'].str.len()
            print(f"   Sentence 길이 - 평균: {sent_lengths.mean():.1f}, 범위: {sent_lengths.min()}-{sent_lengths.max()}")
        if 'label' in df.columns:
            label_dist = df['label'].value_counts()
            print(f"   라벨 분포: {dict(label_dist)}")

    elif dataset_name == 'mrpc':
        if 'sentence1' in df.columns and 'sentence2' in df.columns:
            s1_lengths = df['sentence1'].str.len()
            s2_lengths = df['sentence2'].str.len()
            print(f"   Sentence1 길이 - 평균: {s1_lengths.mean():.1f}")
            print(f"   Sentence2 길이 - 평균: {s2_lengths.mean():.1f}")
        if 'label' in df.columns:
            label_dist = df['label'].value_counts()
            print(f"   라벨 분포: {dict(label_dist)}")


if __name__ == "__main__":
    # 테스트 실행 (직접 실행시)
    print("=== Multi-Dataset Analysis 모듈 테스트 ===")
    run_quality_analysis()