import random
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import copy


class MultiDatasetNoiseInjector:
    def __init__(self, random_seed=42):
        """
        다중 데이터셋용 노이즈 주입기 (라벨 보존 기능 포함)
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 기존 노이즈 패턴들 유지
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

        self.keyboard_neighbors = {
            'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'rfg',
            'y': 'tgh', 'u': 'yhj', 'i': 'ujk', 'o': 'ikl', 'p': 'ol',
            'a': 'qws', 's': 'awde', 'd': 'serf', 'f': 'drtg', 'g': 'ftyh',
            'h': 'gyuj', 'j': 'huik', 'k': 'jiol', 'l': 'kop',
            'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
            'n': 'bhjm', 'm': 'njk'
        }

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

        # 데이터셋별 설정
        self.dataset_configs = {
            'alpaca': {
                'text_columns': ['instruction', 'output'],
                'label_columns': [],
                'preserve_labels': False,
                'flip_labels': False  # 라벨이 없음
            },
            'gsm8k': {
                'text_columns': ['question'],
                'label_columns': ['answer'],
                'preserve_labels': True,  # answer는 보존
                'flip_labels': False  # 수학 답은 플리핑 안함
            },
            'sst2': {
                'text_columns': ['sentence'],
                'label_columns': ['label'],
                'preserve_labels': True,  # label은 보존
                'flip_labels': True,  # 0↔1 플리핑 가능
                'label_mapping': {0: 1, 1: 0}  # negative ↔ positive
            },
            'mrpc': {
                'text_columns': ['sentence1', 'sentence2'],
                'label_columns': ['label'],
                'preserve_labels': True,  # label은 보존
                'flip_labels': True,  # 0↔1 플리핑 가능
                'label_mapping': {0: 1, 1: 0}  # not paraphrase ↔ paraphrase
            }
        }

    def inject_noise(self, df: pd.DataFrame, dataset_name: str, noise_ratio: float = 0.2,
                     noise_strategy: str = 'balanced', flip_labels: bool = False) -> Tuple[pd.DataFrame, List[int]]:
        """
        데이터셋별 노이즈 주입 (라벨 보존 지원)

        Args:
            df: 데이터프레임
            dataset_name: 데이터셋 이름 ('alpaca', 'gsm8k', 'sst2', 'mrpc')
            noise_ratio: 노이즈 비율
            noise_strategy: 노이즈 전략
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")

        config = self.dataset_configs[dataset_name]

        # 라벨 플리핑 옵션 검증
        if flip_labels and not config.get('flip_labels', False):
            print(f"WARNING: {dataset_name}는 라벨 플리핑을 지원하지 않습니다. 텍스트 노이즈만 적용됩니다.")
            flip_labels = False

        mode_desc = "라벨 플리핑" if flip_labels else f"전략: {noise_strategy}"
        print(f"\n=== {dataset_name.upper()} 노이즈 주입 (비율: {noise_ratio * 100:.1f}%, {mode_desc}) ===")
        print(f"텍스트 컬럼: {config['text_columns']}")
        print(f"라벨 컬럼: {config['label_columns']} {'(플리핑)' if flip_labels else '(보존)' if config['preserve_labels'] else '(변경 가능)'}")

        noisy_df = df.copy()
        n_samples = len(df)
        n_noisy = int(n_samples * noise_ratio)

        # 샘플링 전략 (데이터셋별 적용)
        if dataset_name == 'alpaca':
            noisy_indices = self._stratified_sampling_alpaca(df, n_noisy)
        else:
            noisy_indices = self._stratified_sampling_classification(df, n_noisy, config)

        print(f"전체 샘플 수: {n_samples}")
        print(f"노이즈 주입 대상: {n_noisy}개")

        # 라벨 플리핑 모드
        if flip_labels:
            return self._apply_label_flipping(df, noisy_df, noisy_indices, config, dataset_name)

        # 기존 텍스트 노이즈 모드
        # 전략에 따른 노이즈 유형 가중치
        noise_weights = self._get_noise_weights(noise_strategy)

        # 노이즈 유형별 카운터
        noise_type_counts = {"grammar": 0, "semantic": 0, "quality": 0}

        for idx in noisy_indices:
            # 가중치 기반 노이즈 유형 선택
            noise_type = np.random.choice(
                list(noise_weights.keys()),
                p=list(noise_weights.values())
            )
            noise_type_counts[noise_type] += 1

            # 원본 샘플
            original_sample = df.iloc[idx].copy()

            # 데이터셋별 노이즈 적용
            if config['preserve_labels']:
                noisy_df.iloc[idx] = self._apply_noise_preserve_labels(
                    original_sample, noise_type, config, df
                )
            else:
                # 기존 Alpaca 방식
                noisy_df.iloc[idx] = self._apply_noise_alpaca_style(
                    original_sample, noise_type, df
                )

            # 노이즈 적용 확인
            changed = self._check_sample_changed(original_sample, noisy_df.iloc[idx], config)
            if not changed:
                # 강제 노이즈 적용
                noisy_df.iloc[idx] = self._force_apply_noise_by_dataset(
                    original_sample, config
                )

        print("노이즈 유형별 적용 개수:")
        for noise_type, count in noise_type_counts.items():
            percentage = (count / n_noisy) * 100 if n_noisy > 0 else 0
            print(f"  - {noise_type}: {count}개 ({percentage:.1f}%)")

        return noisy_df, noisy_indices

    def _apply_noise_preserve_labels(self, sample: pd.Series, noise_type: str,
                                     config: Dict, full_df: pd.DataFrame) -> pd.Series:
        """라벨 보존 노이즈 적용"""
        noisy_sample = sample.copy()

        # 텍스트 컬럼만 선택해서 노이즈 적용
        text_columns = config['text_columns']
        target_column = random.choice(text_columns)

        original_text = str(noisy_sample[target_column])

        if noise_type == 'grammar':
            noisy_text = self._apply_guaranteed_grammar_noise_text(original_text)
        elif noise_type == 'semantic':
            noisy_text = self._apply_guaranteed_semantic_noise_text(original_text, target_column)
        elif noise_type == 'quality':
            noisy_text = self._apply_guaranteed_quality_noise_text(original_text)

        noisy_sample[target_column] = noisy_text

        # 라벨 컬럼은 원본 그대로 유지 (중요!)
        for label_col in config['label_columns']:
            noisy_sample[label_col] = sample[label_col]

        return noisy_sample

    def _apply_guaranteed_grammar_noise_text(self, text: str) -> str:
        """텍스트에만 문법 노이즈 적용"""
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

        return noisy_text

    def _apply_guaranteed_semantic_noise_text(self, text: str, column_name: str) -> str:
        """텍스트에만 의미적 노이즈 적용"""
        operations = ['irrelevant_addition', 'topic_drift', 'wrong_context']
        operation = random.choice(operations)

        if operation == 'irrelevant_addition':
            irrelevant = random.choice(self.irrelevant_sentences)
            position = random.choice(['start', 'end'])
            if position == 'start':
                return f"{irrelevant} {text}"
            else:
                return f"{text} {irrelevant}"

        elif operation == 'topic_drift':
            sentences = text.split('. ')
            if len(sentences) > 1:
                irrelevant = random.choice(self.irrelevant_sentences).rstrip('.')
                sentences[-1] = irrelevant
                return '. '.join(sentences)
            else:
                irrelevant = random.choice(self.irrelevant_sentences)
                return f"{text} {irrelevant}"

        elif operation == 'wrong_context':
            # 문맥에 맞지 않는 단어들 추가
            wrong_phrases = [
                "about cooking recipes",
                "in space exploration",
                "regarding weather patterns",
                "concerning movie reviews",
                "about sports statistics"
            ]
            wrong_phrase = random.choice(wrong_phrases)
            return f"{text} {wrong_phrase}"

        return text

    def _apply_guaranteed_quality_noise_text(self, text: str) -> str:
        """텍스트에만 품질 저하 노이즈 적용"""
        operations = ['truncate', 'duplicate', 'empty', 'rambling']
        operation = random.choice(operations)

        if operation == 'truncate':
            if len(text) > 20:
                cut_point = len(text) // 2
                return text[:cut_point] + "..."
            else:
                return "Incomplete text..."

        elif operation == 'duplicate':
            sentences = text.split('. ')
            if len(sentences) > 1:
                duplicated = sentences[0] + '. ' + sentences[0] + '. ' + '. '.join(sentences[1:])
                return duplicated
            else:
                return text + " " + text

        elif operation == 'empty':
            short_responses = ["Not sure.", "Maybe.", "Unclear.", "Don't know."]
            return random.choice(short_responses)

        elif operation == 'rambling':
            rambling = "Well, um, you know, it's like, how do I say this..."
            return f"{rambling} {text}"

        return text

    def _stratified_sampling_classification(self, df: pd.DataFrame, n_samples: int, config: Dict) -> List[int]:
        """Classification 데이터용 계층적 샘플링 (라벨 균형 고려)"""
        if n_samples >= len(df):
            return list(range(len(df)))

        # 라벨이 있는 경우 라벨별 균등 샘플링
        if config['label_columns']:
            label_col = config['label_columns'][0]  # 첫 번째 라벨 컬럼 사용

            selected_indices = []
            label_groups = df.groupby(label_col)
            samples_per_label = n_samples // len(label_groups)

            for label_value, group_df in label_groups:
                group_indices = group_df.index.tolist()
                n_group_samples = min(samples_per_label, len(group_indices))
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
        else:
            # 라벨이 없는 경우 랜덤 샘플링
            return random.sample(range(len(df)), n_samples)

    def _stratified_sampling_alpaca(self, df: pd.DataFrame, n_samples: int) -> List[int]:
        """Alpaca용 기존 샘플링 방법"""
        if n_samples >= len(df):
            return list(range(len(df)))

        # instruction 길이 기반 계층적 샘플링
        df_with_length = df.copy()
        df_with_length['inst_length'] = df['instruction'].str.len()

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

    def _check_sample_changed(self, original: pd.Series, noisy: pd.Series, config: Dict) -> bool:
        """샘플이 변경되었는지 확인 (텍스트 컬럼만 체크)"""
        for col in config['text_columns']:
            if original[col] != noisy[col]:
                return True
        return False

    def _force_apply_noise_by_dataset(self, sample: pd.Series, config: Dict) -> pd.Series:
        """데이터셋별 강제 노이즈 적용"""
        noisy_sample = sample.copy()

        # 첫 번째 텍스트 컬럼에 강제 노이즈
        target_column = config['text_columns'][0]
        text = str(noisy_sample[target_column])

        # 간단하고 확실한 변경
        words = text.split()
        if len(words) > 0:
            last_word = words[-1]
            if len(last_word) > 3:
                char_list = list(last_word)
                char_list[-2] = 'x'
                words[-1] = ''.join(char_list)
                noisy_sample[target_column] = ' '.join(words)
            else:
                words[-1] = last_word + 'x'
                noisy_sample[target_column] = ' '.join(words)

        # 라벨 컬럼은 보존
        for label_col in config['label_columns']:
            noisy_sample[label_col] = sample[label_col]

        return noisy_sample

    def _apply_label_flipping(self, original_df: pd.DataFrame, noisy_df: pd.DataFrame, 
                             noisy_indices: List[int], config: Dict, dataset_name: str) -> Tuple[pd.DataFrame, List[int]]:
        """라벨 플리핑 적용"""
        if not config.get('flip_labels', False):
            print("ERROR: 이 데이터셋은 라벨 플리핑을 지원하지 않습니다.")
            return noisy_df, noisy_indices

        label_column = config['label_columns'][0]  # 첫 번째 라벨 컬럼
        label_mapping = config.get('label_mapping', {})
        
        if not label_mapping:
            print("ERROR: 라벨 매핑이 정의되지 않았습니다.")
            return noisy_df, noisy_indices

        # 라벨 플리핑 수행
        flipped_count = 0
        for idx in noisy_indices:
            original_label = original_df.iloc[idx][label_column]
            
            if original_label in label_mapping:
                noisy_df.iloc[idx, noisy_df.columns.get_loc(label_column)] = label_mapping[original_label]
                flipped_count += 1

        print(f"라벨 플리핑 완료:")
        print(f"  - 플리핑된 라벨: {flipped_count}/{len(noisy_indices)}개")
        print(f"  - 매핑: {label_mapping}")
        
        # 플리핑 전후 라벨 분포 출력
        original_dist = original_df[label_column].value_counts().sort_index()
        noisy_dist = noisy_df[label_column].value_counts().sort_index()
        
        print("\n라벨 분포 변화:")
        for label in sorted(original_dist.index):
            orig_count = original_dist.get(label, 0)
            noisy_count = noisy_dist.get(label, 0)
            change = noisy_count - orig_count
            print(f"  라벨 {label}: {orig_count} → {noisy_count} ({change:+d})")

        return noisy_df, noisy_indices

    def _apply_noise_alpaca_style(self, sample: pd.Series, noise_type: str, full_df: pd.DataFrame) -> pd.Series:
        """기존 Alpaca 스타일 노이즈 적용"""
        noisy_sample = sample.copy()

        target_field = random.choices(['instruction', 'output'], weights=[0.2, 0.8])[0]
        text = str(noisy_sample[target_field])

        if noise_type == 'grammar':
            noisy_text = self._apply_guaranteed_grammar_noise_text(text)
        elif noise_type == 'semantic':
            if target_field == 'output':
                # 다른 샘플의 output으로 교체
                other_indices = [i for i in range(len(full_df)) if i != sample.name]
                if other_indices:
                    random_idx = random.choice(other_indices)
                    noisy_text = full_df.iloc[random_idx]['output']
                else:
                    noisy_text = self._apply_guaranteed_semantic_noise_text(text, target_field)
            else:
                noisy_text = self._apply_guaranteed_semantic_noise_text(text, target_field)
        elif noise_type == 'quality':
            noisy_text = self._apply_guaranteed_quality_noise_text(text)

        noisy_sample[target_field] = noisy_text
        return noisy_sample

    def _get_noise_weights(self, strategy: str) -> Dict[str, float]:
        """전략별 노이즈 가중치 반환"""
        strategies = {
            'balanced': {'grammar': 0.4, 'semantic': 0.35, 'quality': 0.25},
            'grammar_heavy': {'grammar': 0.6, 'semantic': 0.25, 'quality': 0.15},
            'semantic_heavy': {'grammar': 0.2, 'semantic': 0.6, 'quality': 0.2}
        }
        return strategies.get(strategy, strategies['balanced'])

    # 기존 노이즈 메서드들 유지 (하위 호환성)
    def _guaranteed_typos(self, text: str) -> str:
        """확실한 오타 적용"""
        words = text.split()
        if not words:
            return text + " (typo)"

        n_typos = min(random.randint(1, 3), len(words))
        target_indices = random.sample(range(len(words)), n_typos)

        for idx in target_indices:
            words[idx] = self._create_guaranteed_typo(words[idx])

        return ' '.join(words)

    def _create_guaranteed_typo(self, word: str) -> str:
        """확실한 오타 생성"""
        if len(word) <= 2:
            return word + 'x'

        word_lower = word.lower()

        # 미리 정의된 오타 사용
        if word_lower in self.common_typos and random.random() < 0.5:
            typo_word = random.choice(self.common_typos[word_lower])
            return self._preserve_case(word, typo_word)

        # 강제 오타 패턴 적용
        typo_patterns = [
            self._transpose_chars,
            self._substitute_char,
            self._keyboard_typo
        ]

        typo_func = random.choice(typo_patterns)
        result = typo_func(word)

        if result == word:
            if len(word) > 3:
                char_list = list(word)
                char_list[1] = 'x'
                result = ''.join(char_list)
            else:
                result = word + 'x'

        return result

    def _guaranteed_word_shuffle(self, text: str) -> str:
        """확실한 단어 순서 섞기"""
        sentences = text.split('. ')
        if len(sentences) == 0:
            return text

        target_sentence_idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[target_sentence_idx]

        words = sentence.split()
        if len(words) > 2:
            if len(words) > 4:
                middle_words = words[1:-1]
                random.shuffle(middle_words)
                sentences[target_sentence_idx] = ' '.join([words[0]] + middle_words + [words[-1]])
            else:
                random.shuffle(words)
                sentences[target_sentence_idx] = ' '.join(words)
        elif len(words) == 2:
            sentences[target_sentence_idx] = f"{words[1]} {words[0]}"

        return '. '.join(sentences)

    def _guaranteed_punctuation_mess(self, text: str) -> str:
        """확실한 구두점 오류"""
        operations = [
            lambda x: x.replace('.', '!', 1),
            lambda x: x.replace(',', '', 1),
            lambda x: x.replace('?', '.', 1),
            lambda x: x + '???',
            lambda x: x.replace(' ', '  ', 1),
            lambda x: x.replace('.', '. .', 1),
        ]

        operation = random.choice(operations)
        return operation(text)

    def _guaranteed_grammar_errors(self, text: str) -> str:
        """확실한 문법 오류"""
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

        for pattern, replacement, count in grammar_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return re.sub(pattern, replacement, text, count=count, flags=re.IGNORECASE)

        words = text.split()
        if len(words) > 1:
            words.insert(1, "is")
            words.insert(2, "are")
            return ' '.join(words)

        return text + " is are"

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


# 하위 호환성을 위한 NoiseInjector 유지 (Alpaca 전용)
class NoiseInjector(MultiDatasetNoiseInjector):
    def __init__(self, random_seed=42):
        super().__init__(random_seed)

    def inject_noise(self, df: pd.DataFrame, noise_ratio: float = 0.2,
                     noise_strategy: str = 'balanced') -> Tuple[pd.DataFrame, List[int]]:
        """기존 Alpaca 전용 메서드 (하위 호환성)"""
        return super().inject_noise(df, 'alpaca', noise_ratio, noise_strategy)


# 분석 함수들
def compare_samples(original_df: pd.DataFrame, noisy_df: pd.DataFrame,
                    noisy_indices: List[int], num_examples: int = 5, dataset_name: str = 'alpaca'):
    """데이터셋별 노이즈 주입 전후 샘플 비교"""
    print(f"\n=== {dataset_name.upper()} 노이즈 주입 전후 비교 ({num_examples}개 샘플) ===")

    sample_indices = random.sample(noisy_indices, min(num_examples, len(noisy_indices)))

    for i, idx in enumerate(sample_indices):
        print(f"\n[예시 {i + 1}] 인덱스: {idx}")
        print("=" * 60)

        if dataset_name == 'alpaca':
            print("원본:")
            print(f"Instruction: {original_df.iloc[idx]['instruction']}")
            print(f"Output: {original_df.iloc[idx]['output']}")
            print("\n노이즈 주입 후:")
            print(f"Instruction: {noisy_df.iloc[idx]['instruction']}")
            print(f"Output: {noisy_df.iloc[idx]['output']}")

            inst_changed = original_df.iloc[idx]['instruction'] != noisy_df.iloc[idx]['instruction']
            out_changed = original_df.iloc[idx]['output'] != noisy_df.iloc[idx]['output']
            print(f"변경사항: Instruction {'O' if inst_changed else 'X'}, Output {'O' if out_changed else 'X'}")

        elif dataset_name == 'gsm8k':
            print("원본:")
            print(f"Question: {original_df.iloc[idx]['question']}")
            print(f"Answer: {original_df.iloc[idx]['answer']}")
            print("\n노이즈 주입 후:")
            print(f"Question: {noisy_df.iloc[idx]['question']}")
            print(f"Answer: {noisy_df.iloc[idx]['answer']}")

            q_changed = original_df.iloc[idx]['question'] != noisy_df.iloc[idx]['question']
            a_changed = original_df.iloc[idx]['answer'] != noisy_df.iloc[idx]['answer']
            print(f"변경사항: Question {'O' if q_changed else 'X'}, Answer {'O' if a_changed else 'X'}")

        elif dataset_name == 'sst2':
            print("원본:")
            print(f"Sentence: {original_df.iloc[idx]['sentence']}")
            print(f"Label: {original_df.iloc[idx]['label']}")
            print("\n노이즈 주입 후:")
            print(f"Sentence: {noisy_df.iloc[idx]['sentence']}")
            print(f"Label: {noisy_df.iloc[idx]['label']}")

            s_changed = original_df.iloc[idx]['sentence'] != noisy_df.iloc[idx]['sentence']
            l_changed = original_df.iloc[idx]['label'] != noisy_df.iloc[idx]['label']
            print(f"변경사항: Sentence {'O' if s_changed else 'X'}, Label {'O' if l_changed else 'X'}")

        print("-" * 60)


def analyze_noise_distribution(noisy_df: pd.DataFrame, original_df: pd.DataFrame,
                               noisy_indices: List[int], dataset_name: str = 'alpaca') -> Dict:
    """데이터셋별 노이즈 분포 분석"""
    analysis = {
        'dataset_name': dataset_name,
        'total_samples': len(original_df),
        'noisy_samples': len(noisy_indices),
        'noise_ratio': len(noisy_indices) / len(original_df),
        'length_changes': [],
        'field_changes': {},
        'actual_changes': 0,
        'change_types': {'typos': 0, 'grammar': 0, 'semantic': 0, 'quality': 0}
    }

    # 데이터셋별 분석
    injector = MultiDatasetNoiseInjector()
    config = injector.dataset_configs.get(dataset_name, {})

    text_columns = config.get('text_columns', [])
    label_columns = config.get('label_columns', [])

    # 필드별 변경 초기화
    for col in text_columns + label_columns:
        analysis['field_changes'][col] = 0

    for idx in noisy_indices:
        # 텍스트 컬럼 길이 변화 분석
        total_length_change = 0
        sample_changed = False

        for col in text_columns:
            orig_len = len(str(original_df.iloc[idx][col]))
            noisy_len = len(str(noisy_df.iloc[idx][col]))
            total_length_change += (noisy_len - orig_len)

            # 필드 변경 여부
            if original_df.iloc[idx][col] != noisy_df.iloc[idx][col]:
                analysis['field_changes'][col] += 1
                sample_changed = True

        # 라벨 변경 여부 (보존되어야 함)
        for col in label_columns:
            if original_df.iloc[idx][col] != noisy_df.iloc[idx][col]:
                analysis['field_changes'][col] += 1
                print(f"WARNING: 라벨이 변경됨! 인덱스 {idx}, 컬럼 {col}")

        analysis['length_changes'].append(total_length_change)

        if sample_changed:
            analysis['actual_changes'] += 1

            # 변경 유형 추측 (첫 번째 텍스트 컬럼 기준)
            if text_columns:
                first_text_col = text_columns[0]
                orig_text = str(original_df.iloc[idx][first_text_col])
                noisy_text = str(noisy_df.iloc[idx][first_text_col])

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