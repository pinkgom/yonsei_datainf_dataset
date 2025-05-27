import random
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import copy


class NoiseInjector:
    def __init__(self, random_seed=42):
        """
        노이즈 주입기 초기화 (확실한 노이즈 적용 버전)
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 확장된 철자 오류 패턴들
        self.common_typos = {
            'the': ['teh', 'hte', 'te', 'th'],
            'and': ['adn', 'nad', 'an', 'nd'],
            'you': ['yuo', 'yu', 'oyu', 'yo'],
            'for': ['fro', 'ofr', 'fo', 'fr'],
            'are': ['aer', 'rae', 'ar', 're'],
            'with': ['wiht', 'whit', 'wit', 'wth'],
            'this': ['thsi', 'tihs', 'ths', 'tis'],
            'that': ['taht', 'tath', 'tat', 'htat'],
            'have': ['hav', 'ahve', 'hvae', 'hve'],
            'they': ['tehy', 'thye', 'tey', 'thy'],
            'from': ['form', 'fomr', 'frm', 'rom'],
            'what': ['waht', 'whta', 'wht', 'wat'],
            'would': ['woudl', 'wolud', 'wuold', 'wouldl'],
            'was': ['wsa', 'aws', 'wa', 'wass'],
            'were': ['wre', 'ewer', 'werre', 'wer'],
            'been': ['ben', 'beeen', 'bene', 'bean'],
            'said': ['siad', 'sayd', 'sed', 'saide'],
            'people': ['poeple', 'peopel', 'peple', 'ppl'],
            'time': ['tiem', 'timee', 'tme', 'tym'],
            'group': ['grupe', 'gropu', 'grup', 'grop']
        }

        # 키보드 인접 키 매핑 (QWERTY)
        self.keyboard_neighbors = {
            'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'rfg',
            'y': 'tgh', 'u': 'yhj', 'i': 'ujk', 'o': 'ikl', 'p': 'ol',
            'a': 'qws', 's': 'awde', 'd': 'serf', 'f': 'drtg', 'g': 'ftyh',
            'h': 'gyuj', 'j': 'huik', 'k': 'jiol', 'l': 'kop',
            'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
            'n': 'bhjm', 'm': 'njk'
        }

        # 무관한 문장들 (의미적 노이즈용)
        self.irrelevant_sentences = [
            "By the way, did you know that cats sleep 12-16 hours a day?",
            "Speaking of pizza, I love pineapple on it.",
            "Random fact: The sky is blue because of Rayleigh scattering.",
            "Unrelated: My favorite color is purple.",
            "Fun fact: Bananas are berries but strawberries aren't.",
            "The weather is quite nice today, isn't it?",
            "I heard that coffee was first discovered in Ethiopia.",
            "Did you watch the latest movie that came out?",
            "Python is one of the most popular programming languages.",
            "The Pacific Ocean covers about 46% of the water surface.",
            "Chocolate was once used as currency by the Aztecs.",
            "A group of flamingos is called a 'flamboyance'."
        ]

    def inject_noise(self, df: pd.DataFrame, noise_ratio: float = 0.2,
                     noise_strategy: str = 'balanced') -> Tuple[pd.DataFrame, List[int]]:
        """
        데이터프레임에 노이즈 주입 (확실한 노이즈 적용)
        """
        print(f"\n=== 노이즈 주입 시작 (비율: {noise_ratio * 100:.1f}%, 전략: {noise_strategy}) ===")

        noisy_df = df.copy()
        n_samples = len(df)
        n_noisy = int(n_samples * noise_ratio)

        # 전략에 따른 노이즈 유형 가중치 설정
        noise_weights = self._get_noise_weights(noise_strategy)

        # 개선된 샘플링 (instruction 길이 기반 계층적 샘플링)
        noisy_indices = self._stratified_sampling(df, n_noisy)

        print(f"전체 샘플 수: {n_samples}")
        print(f"노이즈 주입 대상: {n_noisy}개")
        print(f"샘플링 전략: {noise_strategy}")

        # 노이즈 유형별 카운터
        noise_type_counts = {"grammar": 0, "semantic": 0, "quality": 0}

        for idx in noisy_indices:
            # 가중치 기반 노이즈 유형 선택
            noise_type = np.random.choice(
                list(noise_weights.keys()),
                p=list(noise_weights.values())
            )
            noise_type_counts[noise_type] += 1

            # 선택된 노이즈 유형에 따라 노이즈 주입 (확실하게!)
            original_sample = df.iloc[idx].copy()

            if noise_type == 'grammar':
                noisy_df.iloc[idx] = self._apply_guaranteed_grammar_noise(original_sample)
            elif noise_type == 'semantic':
                noisy_df.iloc[idx] = self._apply_guaranteed_semantic_noise(original_sample, df)
            elif noise_type == 'quality':
                noisy_df.iloc[idx] = self._apply_guaranteed_quality_noise(original_sample)

            # 노이즈가 제대로 적용되었는지 확인
            changed = (original_sample['instruction'] != noisy_df.iloc[idx]['instruction'] or
                       original_sample['output'] != noisy_df.iloc[idx]['output'])

            if not changed:
                # 강제로 노이즈 적용
                noisy_df.iloc[idx] = self._force_apply_noise(original_sample)

        print("노이즈 유형별 적용 개수:")
        for noise_type, count in noise_type_counts.items():
            percentage = (count / n_noisy) * 100 if n_noisy > 0 else 0
            print(f"  - {noise_type}: {count}개 ({percentage:.1f}%)")

        return noisy_df, noisy_indices

    def _force_apply_noise(self, sample: pd.Series) -> pd.Series:
        """강제로 노이즈 적용 (최후의 수단)"""
        noisy_sample = sample.copy()

        # 간단하고 확실한 변경
        output = str(noisy_sample['output'])

        # 마지막 단어에 오타 강제 적용
        words = output.split()
        if len(words) > 0:
            last_word = words[-1]
            if len(last_word) > 3:
                # 마지막에서 두 번째 문자를 'x'로 변경
                char_list = list(last_word)
                char_list[-2] = 'x'
                words[-1] = ''.join(char_list)
                noisy_sample['output'] = ' '.join(words)
            else:
                # 짧은 단어면 끝에 'x' 추가
                words[-1] = last_word + 'x'
                noisy_sample['output'] = ' '.join(words)

        return noisy_sample

    def _get_noise_weights(self, strategy: str) -> Dict[str, float]:
        """전략별 노이즈 가중치 반환"""
        strategies = {
            'balanced': {'grammar': 0.4, 'semantic': 0.35, 'quality': 0.25},
            'grammar_heavy': {'grammar': 0.6, 'semantic': 0.25, 'quality': 0.15},
            'semantic_heavy': {'grammar': 0.2, 'semantic': 0.6, 'quality': 0.2}
        }
        return strategies.get(strategy, strategies['balanced'])

    def _stratified_sampling(self, df: pd.DataFrame, n_samples: int) -> List[int]:
        """instruction 길이 기반 계층적 샘플링"""
        if n_samples >= len(df):
            return list(range(len(df)))

        # instruction 길이 계산
        df_with_length = df.copy()
        df_with_length['inst_length'] = df['instruction'].str.len()

        # 분위수 기반 그룹 분할
        quartiles = df_with_length['inst_length'].quantile([0.25, 0.5, 0.75]).values

        def get_length_group(length):
            if length <= quartiles[0]:
                return 'short'
            elif length <= quartiles[1]:
                return 'medium'
            elif length <= quartiles[2]:
                return 'long'
            else:
                return 'very_long'

        df_with_length['length_group'] = df_with_length['inst_length'].apply(get_length_group)

        # 각 그룹에서 균등하게 샘플링
        selected_indices = []
        groups = df_with_length.groupby('length_group')
        samples_per_group = n_samples // len(groups)

        for group_name, group_df in groups:
            group_indices = group_df.index.tolist()
            n_group_samples = min(samples_per_group, len(group_indices))
            selected_indices.extend(random.sample(group_indices, n_group_samples))

        # 부족한 샘플 추가
        remaining = n_samples - len(selected_indices)
        if remaining > 0:
            all_indices = set(df.index.tolist())
            available_indices = list(all_indices - set(selected_indices))
            if available_indices:
                additional = random.sample(available_indices, min(remaining, len(available_indices)))
                selected_indices.extend(additional)

        return selected_indices[:n_samples]

    def _apply_guaranteed_grammar_noise(self, sample: pd.Series) -> pd.Series:
        """확실한 문법 노이즈 적용"""
        noisy_sample = sample.copy()

        # output에 노이즈 적용 (더 높은 확률)
        target_field = random.choices(['instruction', 'output'], weights=[0.2, 0.8])[0]
        text = str(noisy_sample[target_field])

        # 반드시 하나 이상의 노이즈 적용
        grammar_operations = [
            self._guaranteed_typos,
            self._guaranteed_word_shuffle,
            self._guaranteed_punctuation_mess,
            self._guaranteed_grammar_errors
        ]

        # 50% 확률로 복수 오류 적용
        if random.random() < 0.5:
            n_operations = random.randint(1, 2)
            selected_ops = random.sample(grammar_operations, n_operations)
        else:
            selected_ops = [random.choice(grammar_operations)]

        noisy_text = text
        for op in selected_ops:
            noisy_text = op(noisy_text)

        noisy_sample[target_field] = noisy_text
        return noisy_sample

    def _guaranteed_typos(self, text: str) -> str:
        """확실한 오타 적용 (최소 1개 이상)"""
        words = text.split()
        if not words:
            return text + " (typo)"

        # 최소 1개, 최대 3개 단어에 오타 적용
        n_typos = min(random.randint(1, 3), len(words))
        target_indices = random.sample(range(len(words)), n_typos)

        for idx in target_indices:
            words[idx] = self._create_guaranteed_typo(words[idx])

        return ' '.join(words)

    def _create_guaranteed_typo(self, word: str) -> str:
        """확실한 오타 생성"""
        if len(word) <= 2:
            return word + 'x'  # 짧은 단어는 끝에 x 추가

        word_lower = word.lower()

        # 1. 미리 정의된 오타 사용 (50% 확률)
        if word_lower in self.common_typos and random.random() < 0.5:
            typo_word = random.choice(self.common_typos[word_lower])
            return self._preserve_case(word, typo_word)

        # 2. 강제 오타 패턴 적용
        typo_patterns = [
            self._transpose_chars,  # 문자 순서 바꾸기
            self._substitute_char,  # 문자 치환
            self._keyboard_typo  # 키보드 인접 키
        ]

        typo_func = random.choice(typo_patterns)
        result = typo_func(word)

        # 변경되지 않았으면 강제로 변경
        if result == word:
            if len(word) > 3:
                char_list = list(word)
                char_list[1] = 'x'  # 두 번째 문자를 x로
                result = ''.join(char_list)
            else:
                result = word + 'x'

        return result

    def _guaranteed_word_shuffle(self, text: str) -> str:
        """확실한 단어 순서 섞기"""
        sentences = text.split('. ')
        if len(sentences) == 0:
            return text

        # 적어도 하나의 문장에서 단어 순서 섞기
        target_sentence_idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[target_sentence_idx]

        words = sentence.split()
        if len(words) > 2:
            # 첫 번째와 마지막 단어는 유지하고 중간 섞기
            if len(words) > 4:
                middle_words = words[1:-1]
                random.shuffle(middle_words)
                sentences[target_sentence_idx] = ' '.join([words[0]] + middle_words + [words[-1]])
            else:
                # 짧은 문장은 전체 섞기
                random.shuffle(words)
                sentences[target_sentence_idx] = ' '.join(words)
        elif len(words) == 2:
            # 두 단어만 있으면 순서 바꾸기
            sentences[target_sentence_idx] = f"{words[1]} {words[0]}"

        return '. '.join(sentences)

    def _guaranteed_punctuation_mess(self, text: str) -> str:
        """확실한 구두점 오류"""
        # 반드시 변경되도록 하는 구두점 오류
        operations = [
            lambda x: x.replace('.', '!', 1),  # 첫 번째 마침표를 느낌표로
            lambda x: x.replace(',', '', 1),  # 첫 번째 쉼표 제거
            lambda x: x.replace('?', '.', 1),  # 첫 번째 물음표를 마침표로
            lambda x: x + '???',  # 끝에 물음표 추가
            lambda x: x.replace(' ', '  ', 1),  # 첫 번째 공백을 두 개로
            lambda x: x.replace('.', '. .', 1),  # 첫 번째 마침표를 두 개로
        ]

        operation = random.choice(operations)
        return operation(text)

    def _guaranteed_grammar_errors(self, text: str) -> str:
        """확실한 문법 오류"""
        # 반드시 하나는 적용되도록
        grammar_patterns = [
            (r'\bis\b', 'are', 1),
            (r'\bare\b', 'is', 1),
            (r'\bwas\b', 'were', 1),
            (r'\bwere\b', 'was', 1),
            (r'\bhas\b', 'have', 1),
            (r'\bhave\b', 'has', 1),
            (r'\ba\b', 'an', 1),
            (r'\ban\b', 'a', 1),
        ]

        # 패턴이 존재하는지 확인하고 적용
        for pattern, replacement, count in grammar_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return re.sub(pattern, replacement, text, count=count, flags=re.IGNORECASE)

        # 어떤 패턴도 매치되지 않으면 강제로 오류 추가
        words = text.split()
        if len(words) > 1:
            # 두 번째 단어 뒤에 "is are" 추가 (명백한 문법 오류)
            words.insert(1, "is")
            words.insert(2, "are")
            return ' '.join(words)

        return text + " is are"

    def _apply_guaranteed_semantic_noise(self, sample: pd.Series, full_df: pd.DataFrame) -> pd.Series:
        """확실한 의미적 노이즈 적용"""
        noisy_sample = sample.copy()

        semantic_operations = ['wrong_context', 'irrelevant_addition', 'topic_drift']
        operation = random.choice(semantic_operations)

        if operation == 'wrong_context':
            # 다른 샘플의 output으로 완전 교체
            other_indices = [i for i in range(len(full_df)) if i != sample.name]
            if other_indices:
                random_idx = random.choice(other_indices)
                noisy_sample['output'] = full_df.iloc[random_idx]['output']

        elif operation == 'irrelevant_addition':
            # 무관한 내용 확실히 추가
            irrelevant = random.choice(self.irrelevant_sentences)
            output = str(noisy_sample['output'])

            position = random.choice(['start', 'end'])
            if position == 'start':
                noisy_sample['output'] = f"{irrelevant} {output}"
            else:
                noisy_sample['output'] = f"{output} {irrelevant}"

        elif operation == 'topic_drift':
            # 답변의 일부를 무관한 내용으로 교체
            output = str(noisy_sample['output'])
            sentences = output.split('. ')

            if len(sentences) > 1:
                # 마지막 문장을 무관한 내용으로 교체
                irrelevant = random.choice(self.irrelevant_sentences).rstrip('.')
                sentences[-1] = irrelevant
                noisy_sample['output'] = '. '.join(sentences)
            else:
                # 문장이 하나뿐이면 끝에 무관한 내용 추가
                irrelevant = random.choice(self.irrelevant_sentences)
                noisy_sample['output'] = f"{output} {irrelevant}"

        return noisy_sample

    def _apply_guaranteed_quality_noise(self, sample: pd.Series) -> pd.Series:
        """확실한 품질 저하 노이즈"""
        noisy_sample = sample.copy()

        quality_operations = ['truncate', 'duplicate', 'empty', 'rambling']
        operation = random.choice(quality_operations)

        if operation == 'truncate':
            # 응답을 확실히 자르기
            output = str(noisy_sample['output'])
            if len(output) > 20:
                cut_point = len(output) // 2
                noisy_sample['output'] = output[:cut_point] + "..."
            else:
                noisy_sample['output'] = "Incomplete answer..."

        elif operation == 'duplicate':
            # 내용 확실히 중복
            output = str(noisy_sample['output'])
            sentences = output.split('. ')

            if len(sentences) > 1:
                # 첫 번째 문장 중복
                duplicated = sentences[0] + '. ' + sentences[0] + '. ' + '. '.join(sentences[1:])
                noisy_sample['output'] = duplicated
            else:
                # 전체 텍스트 중복
                noisy_sample['output'] = output + " " + output

        elif operation == 'empty':
            # 짧고 무의미한 답변
            short_responses = ["I don't know.", "No.", "Maybe.", "Yes.", "Not sure."]
            noisy_sample['output'] = random.choice(short_responses)

        elif operation == 'rambling':
            # 횡설수설 추가
            rambling = "Well, um, you know, it's like, how do I say this..."
            original = str(noisy_sample['output'])
            noisy_sample['output'] = f"{rambling} {original}"

        return noisy_sample

    # 헬퍼 함수들
    def _preserve_case(self, original: str, typo: str) -> str:
        """원래 단어의 대소문자 패턴 유지"""
        if original.isupper():
            return typo.upper()
        elif original.istitle():
            return typo.capitalize()
        else:
            return typo

    def _keyboard_typo(self, word: str) -> str:
        """키보드 인접 키 오타"""
        if len(word) <= 2:
            return word + 'x'

        char_list = list(word.lower())
        # 중간 문자 선택
        rand_idx = random.randint(1, len(char_list) - 2)
        original_char = char_list[rand_idx]

        if original_char in self.keyboard_neighbors:
            neighbors = self.keyboard_neighbors[original_char]
            if neighbors:
                char_list[rand_idx] = random.choice(neighbors)

        result = ''.join(char_list)
        return self._preserve_case(word, result)

    def _transpose_chars(self, word: str) -> str:
        """인접한 문자 순서 바꾸기"""
        if len(word) <= 3:
            return word
        char_list = list(word)
        idx = random.randint(0, len(char_list) - 2)
        char_list[idx], char_list[idx + 1] = char_list[idx + 1], char_list[idx]
        return ''.join(char_list)

    def _substitute_char(self, word: str) -> str:
        """문자 치환"""
        if len(word) <= 2:
            return word
        char_list = list(word)
        idx = random.randint(1, len(char_list) - 2)
        char_list[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return ''.join(char_list)

    # 기존 호환성 메서드들
    def _apply_grammar_noise(self, sample: pd.Series) -> pd.Series:
        return self._apply_guaranteed_grammar_noise(sample)

    def _apply_semantic_noise(self, sample: pd.Series, full_df: pd.DataFrame) -> pd.Series:
        return self._apply_guaranteed_semantic_noise(sample, full_df)

    def _apply_quality_noise(self, sample: pd.Series) -> pd.Series:
        return self._apply_guaranteed_quality_noise(sample)

    def _introduce_typos(self, text: str) -> str:
        return self._guaranteed_typos(text)

    def _shuffle_words(self, text: str) -> str:
        return self._guaranteed_word_shuffle(text)

    def _mess_punctuation(self, text: str) -> str:
        return self._guaranteed_punctuation_mess(text)


def compare_samples(original_df: pd.DataFrame, noisy_df: pd.DataFrame,
                    noisy_indices: List[int], num_examples: int = 5):
    """노이즈 주입 전후 샘플 비교"""
    print(f"\n=== 노이즈 주입 전후 비교 ({num_examples}개 샘플) ===")

    sample_indices = random.sample(noisy_indices, min(num_examples, len(noisy_indices)))

    for i, idx in enumerate(sample_indices):
        print(f"\n[예시 {i + 1}] 인덱스: {idx}")
        print("=" * 60)

        print("원본:")
        print(f"Instruction: {original_df.iloc[idx]['instruction']}")
        print(f"Output: {original_df.iloc[idx]['output']}")

        print("\n노이즈 주입 후:")
        print(f"Instruction: {noisy_df.iloc[idx]['instruction']}")
        print(f"Output: {noisy_df.iloc[idx]['output']}")

        # 변경 여부 확인
        inst_changed = original_df.iloc[idx]['instruction'] != noisy_df.iloc[idx]['instruction']
        out_changed = original_df.iloc[idx]['output'] != noisy_df.iloc[idx]['output']

        print(f"변경사항: Instruction {'O' if inst_changed else 'X'}, Output {'O' if out_changed else 'X'}")
        print("-" * 60)


def analyze_noise_distribution(noisy_df: pd.DataFrame, original_df: pd.DataFrame,
                               noisy_indices: List[int]) -> Dict:
    """노이즈 분포 분석 (향상된 버전)"""
    analysis = {
        'total_samples': len(original_df),
        'noisy_samples': len(noisy_indices),
        'noise_ratio': len(noisy_indices) / len(original_df),
        'length_changes': [],
        'field_changes': {'instruction': 0, 'output': 0},
        'actual_changes': 0,  # 실제로 변경된 샘플 수
        'change_types': {'typos': 0, 'grammar': 0, 'semantic': 0, 'quality': 0}
    }

    for idx in noisy_indices:
        # 길이 변화 분석
        orig_len = len(original_df.iloc[idx]['output'])
        noisy_len = len(noisy_df.iloc[idx]['output'])
        analysis['length_changes'].append(noisy_len - orig_len)

        # 어느 필드가 변경되었는지 확인
        inst_changed = original_df.iloc[idx]['instruction'] != noisy_df.iloc[idx]['instruction']
        out_changed = original_df.iloc[idx]['output'] != noisy_df.iloc[idx]['output']

        if inst_changed:
            analysis['field_changes']['instruction'] += 1
        if out_changed:
            analysis['field_changes']['output'] += 1

        # 실제 변경 여부
        if inst_changed or out_changed:
            analysis['actual_changes'] += 1

            # 변경 유형 추측 (간단한 휴리스틱)
            orig_text = original_df.iloc[idx]['output']
            noisy_text = noisy_df.iloc[idx]['output']

            if len(noisy_text) < len(orig_text) * 0.5:
                analysis['change_types']['quality'] += 1
            elif any(phrase in noisy_text for phrase in ['By the way', 'Random fact', 'Fun fact']):
                analysis['change_types']['semantic'] += 1
            elif 'are is' in noisy_text or 'is are' in noisy_text:
                analysis['change_types']['grammar'] += 1
            else:
                analysis['change_types']['typos'] += 1

    analysis['avg_length_change'] = np.mean(analysis['length_changes']) if analysis['length_changes'] else 0
    analysis['actual_noise_ratio'] = analysis['actual_changes'] / len(original_df)

    return analysis