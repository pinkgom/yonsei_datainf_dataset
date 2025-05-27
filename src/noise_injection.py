import random
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple
import copy


class NoiseInjector:
    def __init__(self, random_seed=42):
        """
        노이즈 주입기 초기화
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 일반적인 철자 오류 패턴들
        self.common_typos = {
            'the': ['teh', 'hte', 'te'],
            'and': ['adn', 'nad', 'an'],
            'you': ['yuo', 'yu', 'oyu'],
            'for': ['fro', 'ofr', 'fo'],
            'are': ['aer', 'rae', 'ar'],
            'with': ['wiht', 'whit', 'wit'],
            'this': ['thsi', 'tihs', 'ths']
        }

    def inject_noise(self, df: pd.DataFrame, noise_ratio: float = 0.2) -> Tuple[pd.DataFrame, List[int]]:
        """
        데이터프레임에 노이즈 주입

        Args:
            df: 원본 데이터프레임
            noise_ratio: 노이즈를 주입할 비율 (0.0 ~ 1.0)

        Returns:
            noisy_df: 노이즈가 주입된 데이터프레임
            noisy_indices: 노이즈가 주입된 인덱스 리스트
        """
        print(f"\n=== 노이즈 주입 시작 (비율: {noise_ratio * 100}%) ===")

        noisy_df = df.copy()
        n_samples = len(df)
        n_noisy = int(n_samples * noise_ratio)

        # 노이즈를 주입할 샘플 인덱스 랜덤 선택
        noisy_indices = random.sample(range(n_samples), n_noisy)

        print(f"전체 샘플 수: {n_samples}")
        print(f"노이즈 주입 대상: {n_noisy}개")

        # 노이즈 유형별 카운터
        noise_type_counts = {"grammar": 0, "semantic": 0, "quality": 0}

        for idx in noisy_indices:
            # 노이즈 유형을 랜덤하게 선택
            noise_type = random.choice(['grammar', 'semantic', 'quality'])
            noise_type_counts[noise_type] += 1

            # 선택된 노이즈 유형에 따라 노이즈 주입
            if noise_type == 'grammar':
                noisy_df.iloc[idx] = self._apply_grammar_noise(df.iloc[idx])
            elif noise_type == 'semantic':
                noisy_df.iloc[idx] = self._apply_semantic_noise(df.iloc[idx], df)
            elif noise_type == 'quality':
                noisy_df.iloc[idx] = self._apply_quality_noise(df.iloc[idx])

        print("노이즈 유형별 적용 개수:")
        for noise_type, count in noise_type_counts.items():
            print(f"  - {noise_type}: {count}개")

        return noisy_df, noisy_indices

    def _apply_grammar_noise(self, sample: pd.Series) -> pd.Series:
        """문법/철자 오류 노이즈 주입"""
        noisy_sample = sample.copy()

        # 50% 확률로 instruction 또는 output에 노이즈 주입
        target_field = random.choice(['instruction', 'output'])
        text = str(noisy_sample[target_field])

        # 노이즈 유형 선택
        grammar_noise_type = random.choice(['typo', 'word_order', 'punctuation'])

        if grammar_noise_type == 'typo':
            # 철자 오류 주입
            noisy_text = self._introduce_typos(text)
        elif grammar_noise_type == 'word_order':
            # 단어 순서 섞기
            noisy_text = self._shuffle_words(text)
        else:  # punctuation
            # 구두점 오류
            noisy_text = self._mess_punctuation(text)

        noisy_sample[target_field] = noisy_text
        return noisy_sample

    def _apply_semantic_noise(self, sample: pd.Series, full_df: pd.DataFrame) -> pd.Series:
        """의미적 노이즈 주입"""
        noisy_sample = sample.copy()

        semantic_noise_type = random.choice(['wrong_output', 'irrelevant_content'])

        if semantic_noise_type == 'wrong_output':
            # 다른 샘플의 output과 섞기
            random_idx = random.randint(0, len(full_df) - 1)
            noisy_sample['output'] = full_df.iloc[random_idx]['output']
        else:  # irrelevant_content
            # 무관한 내용 추가
            irrelevant_phrases = [
                "By the way, did you know that cats sleep 12-16 hours a day?",
                "Speaking of pizza, I love pineapple on it.",
                "Random fact: The sky is blue because of Rayleigh scattering.",
                "Unrelated: My favorite color is purple.",
                "Fun fact: Bananas are berries but strawberries aren't."
            ]
            irrelevant_text = random.choice(irrelevant_phrases)
            noisy_sample['output'] = str(noisy_sample['output']) + " " + irrelevant_text

        return noisy_sample

    def _apply_quality_noise(self, sample: pd.Series) -> pd.Series:
        """품질 저하 노이즈 주입"""
        noisy_sample = sample.copy()

        quality_noise_type = random.choice(['truncate', 'duplicate', 'empty'])

        if quality_noise_type == 'truncate':
            # 응답을 중간에 자르기
            output = str(noisy_sample['output'])
            cut_point = random.randint(len(output) // 3, 2 * len(output) // 3)
            noisy_sample['output'] = output[:cut_point] + "..."
        elif quality_noise_type == 'duplicate':
            # 내용 중복
            output = str(noisy_sample['output'])
            sentences = output.split('. ')
            if len(sentences) > 1:
                dup_sentence = random.choice(sentences)
                noisy_sample['output'] = output + " " + dup_sentence
        else:  # empty
            # 빈 응답 또는 매우 짧은 응답
            short_responses = ["I don't know.", "Yes.", "No.", "Maybe.", ""]
            noisy_sample['output'] = random.choice(short_responses)

        return noisy_sample

    def _introduce_typos(self, text: str) -> str:
        """텍스트에 철자 오류 주입"""
        words = text.split()
        if not words:
            return text

        # 20% 확률로 각 단어에 오타 적용
        for i, word in enumerate(words):
            if random.random() < 0.2:
                if word.lower() in self.common_typos:
                    # 미리 정의된 오타 사용
                    typo_word = random.choice(self.common_typos[word.lower()])
                    # 원래 단어의 대소문자 패턴 유지
                    if word.isupper():
                        words[i] = typo_word.upper()
                    elif word.istitle():
                        words[i] = typo_word.capitalize()
                    else:
                        words[i] = typo_word
                else:
                    # 랜덤 문자 변경
                    if len(word) > 2:
                        char_list = list(word)
                        rand_idx = random.randint(1, len(char_list) - 2)
                        char_list[rand_idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                        words[i] = ''.join(char_list)

        return ' '.join(words)

    def _shuffle_words(self, text: str) -> str:
        """문장 내 단어 순서 섞기"""
        sentences = text.split('. ')
        shuffled_sentences = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) > 3:  # 단어가 3개 이상인 경우만 섞기
                # 처음과 마지막 단어는 유지하고 중간 단어들만 섞기
                if len(words) > 4:
                    middle_words = words[1:-1]
                    random.shuffle(middle_words)
                    shuffled_sentence = [words[0]] + middle_words + [words[-1]]
                else:
                    shuffled_sentence = words[:]
                    random.shuffle(shuffled_sentence)
                shuffled_sentences.append(' '.join(shuffled_sentence))
            else:
                shuffled_sentences.append(sentence)

        return '. '.join(shuffled_sentences)

    def _mess_punctuation(self, text: str) -> str:
        """구두점 오류 주입"""
        # 구두점 제거 또는 잘못된 구두점 추가
        noise_operations = [
            lambda x: x.replace('.', ''),  # 마침표 제거
            lambda x: x.replace(',', ''),  # 쉼표 제거
            lambda x: x.replace('.', '!'),  # 마침표를 느낌표로
            lambda x: x.replace('?', '.'),  # 물음표를 마침표로
            lambda x: x + '???',  # 물음표 과다 사용
        ]

        operation = random.choice(noise_operations)
        return operation(text)


# 노이즈 분석 유틸리티 함수들
def compare_samples(original_df: pd.DataFrame, noisy_df: pd.DataFrame,
                    noisy_indices: List[int], num_examples: int = 5):
    """노이즈 주입 전후 샘플 비교"""
    print(f"\n=== 노이즈 주입 전후 비교 ({num_examples}개 샘플) ===")

    sample_indices = random.sample(noisy_indices, min(num_examples, len(noisy_indices)))

    for i, idx in enumerate(sample_indices):
        print(f"\n[예시 {i + 1}] 인덱스: {idx}")
        print("=" * 60)

        print("🔵 원본:")
        print(f"Instruction: {original_df.iloc[idx]['instruction']}")
        print(f"Output: {original_df.iloc[idx]['output']}")

        print("\n🔴 노이즈 주입 후:")
        print(f"Instruction: {noisy_df.iloc[idx]['instruction']}")
        print(f"Output: {noisy_df.iloc[idx]['output']}")
        print("-" * 60)


if __name__ == "__main__":
    # 테스트 실행
    from data_loader import AlpacaDataLoader

    print("=== 노이즈 주입 테스트 ===")

    # 데이터 로드
    loader = AlpacaDataLoader()
    df = loader.load_alpaca_dataset(subset_size=50)  # 작은 샘플로 테스트

    if df is not None:
        # 노이즈 주입기 생성
        injector = NoiseInjector()

        # 노이즈 주입 (20% 비율)
        noisy_df, noisy_indices = injector.inject_noise(df, noise_ratio=0.2)

        # 결과 비교
        compare_samples(df, noisy_df, noisy_indices, num_examples=3)

        # 저장
        loader.save_dataset(noisy_df, "alpaca_noisy_test.json")

        print(f"\n✅ 노이즈 주입 완료!")
        print(f"노이즈가 주입된 샘플 수: {len(noisy_indices)}")