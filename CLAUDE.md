# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multi-Dataset DataInf noise injection tool** for the paper "DATAINF: EFFICIENTLY ESTIMATING DATA INFLUENCE IN LORA-TUNED LLMS AND DIFFUSION MODELS". The project supports **4 different NLP datasets** and systematically injects noise or performs label flipping to create experimental datasets for data influence research.

## Core Commands

### Basic Usage
```bash
# Demo mode (500 samples, 1-2 minutes) - All datasets
python main.py --demo --dataset alpaca   # Instruction-following
python main.py --demo --dataset gsm8k    # Math reasoning
python main.py --demo --dataset sst2     # Sentiment classification
python main.py --demo --dataset mrpc     # Paraphrase detection

# Full mode - Dataset-specific experiments
python main.py --full --dataset alpaca --noise-ratio 0.2 --strategy balanced
python main.py --full --dataset gsm8k --noise-ratio 0.15 --strategy balanced
python main.py --full --dataset sst2 --noise-ratio 0.2 --strategy semantic_heavy
python main.py --full --dataset mrpc --noise-ratio 0.25 --strategy grammar_heavy

# Label flipping experiments (SST-2, MRPC only)
python main.py --full --dataset sst2 --noise-ratio 0.2 --flip-labels
python main.py --full --dataset mrpc --noise-ratio 0.15 --flip-labels

# Quality analysis and cache management
python main.py --analysis
python main.py --cache
```

### Advanced Multi-Dataset Experiments
```bash
# Multiple datasets with same settings
python main.py --full --dataset alpaca --noise-ratios 0.1,0.15,0.2 --strategy all --yes
python main.py --full --dataset gsm8k --noise-ratios 0.1,0.15,0.2 --strategy all --yes
python main.py --full --dataset sst2 --noise-ratios 0.1,0.15,0.2 --strategy all --yes

# Label preservation vs label flipping comparison
python main.py --full --dataset sst2 --noise-ratio 0.2 --strategy balanced  # Text noise only
python main.py --full --dataset sst2 --noise-ratio 0.2 --flip-labels       # Label flipping only

# Combined experiments with auto-confirm
python main.py --full --dataset mrpc --noise-ratios 0.15,0.2 --strategy balanced,semantic_heavy --yes
```

### Testing and Validation
No specific test framework is defined. Validation is done through:
- Built-in analysis tools (`--analysis` mode)
- Sample comparison functions with dataset-specific formatting
- Quality metrics verification including label preservation checks
- DataInf experiment readiness assessment for multiple datasets

## Architecture

### Core Components

**main.py**: Main execution script with unified CLI interface
- Demo mode: Quick 500-sample experiments across all datasets
- Full mode: Complete dataset experiments (Alpaca: 52K, GSM8K: 7.5K, SST-2: 67K, MRPC: 3.7K)
- Analysis mode: 6 different quality analysis tools with multi-dataset support
- Cache management: Multi-dataset download/cache control
- **Label flipping support**: `--flip-labels` option for classification tasks

**src/data_loader.py** (`MultiDatasetLoader`): Unified dataset management
- **4 Dataset Support**: Alpaca, GSM8K, SST-2, MRPC via HuggingFace
- **Dataset-specific configs**: text_columns, label_columns definitions
- Implements caching system to avoid re-downloads
- Supports subset sampling for testing across all datasets
- JSON file I/O for dataset persistence

**src/noise_injection.py** (`MultiDatasetNoiseInjector`): Advanced noise injection system
- **Label preservation**: Classification datasets (GSM8K, SST-2, MRPC) preserve answer/label columns
- **Label flipping**: SST-2 (0↔1 sentiment), MRPC (0↔1 paraphrase) support label flipping
- **Dataset-aware sampling**: Alpaca uses instruction length, others use label balance
- **3 noise strategies**: balanced (40/35/25%), grammar_heavy (60/25/15%), semantic_heavy (20/60/20%)
- **3 noise types**: grammar, semantic, quality degradation
- **Guaranteed noise application**: Ensures all targeted samples are actually modified

**src/analysis.py**: Comprehensive multi-dataset quality analysis suite
- 6 analysis modes: basic info, demo detailed, full detailed, comparison, strategy effects, comprehensive
- **Dataset-specific analysis**: Label distribution tracking, field-specific change analysis
- **Label preservation verification**: Automatic checks for classification tasks
- DataInf experiment readiness assessment across multiple datasets

### Dataset Support Matrix

| Dataset | Task Type | Size | Text Columns | Label Columns | Label Preservation | Label Flipping |
|---------|-----------|------|--------------|---------------|-------------------|----------------|
| **Alpaca** | Instruction-following | ~52K | instruction, output | - | ❌ | ❌ |
| **GSM8K** | Math reasoning | ~7.5K | question | answer | ✅ | ❌ |
| **SST-2** | Sentiment classification | ~67K | sentence | label | ✅ | ✅ (0↔1) |
| **MRPC** | Paraphrase detection | ~3.7K | sentence1, sentence2 | label | ✅ | ✅ (0↔1) |

### Noise Injection Methodology

**Text Noise Types**:
- **Grammar Noise**: Typos, word shuffling, punctuation errors, grammar mistakes
- **Semantic Noise**: Wrong context, irrelevant additions, topic drift  
- **Quality Noise**: Truncation, duplication, incomplete answers, rambling

**Label Operations**:
- **Label Preservation**: Classification datasets keep labels intact while modifying text
- **Label Flipping**: SST-2 (negative↔positive), MRPC (not paraphrase↔paraphrase)

### Data Flow

1. **Load**: Multi-dataset support via HuggingFace (cached locally per dataset)
2. **Sample**: Dataset-aware stratified sampling (instruction length for Alpaca, label balance for others)
3. **Inject**: Apply guaranteed noise with label preservation OR label flipping
4. **Verify**: Ensure actual changes occurred with dataset-specific validation
5. **Save**: Output noisy datasets with comprehensive metadata
6. **Analyze**: Quality assessment and comparison tools across datasets

### File Naming Convention

```
# Original datasets
{dataset}_original_demo_500.json           # Demo original data
{dataset}_original_full_{size}.json        # Full original data

# Text noise experiments
{dataset}_demo_20percent_balanced_500.json      # Demo with 20% balanced text noise
{dataset}_full_20percent_balanced_{size}.json   # Full with 20% balanced text noise
{dataset}_full_15percent_grammar_{size}.json    # Grammar-heavy strategy
{dataset}_full_25percent_semantic_{size}.json   # Semantic-heavy strategy

# Label flipping experiments (SST-2, MRPC only)
{dataset}_full_20percent_flipped_{size}.json    # 20% label flipping

# Metadata
experiment_metadata_{dataset}_{timestamp}.json  # Experiment metadata per dataset
```

### Dependencies

**Core**: `datasets`, `pandas`, `numpy`  
**Optional**: Development tools commented out in requirements.txt
**HuggingFace Datasets**: 
- `tatsu-lab/alpaca` for Alpaca
- `gsm8k` for GSM8K  
- `glue/sst2` for SST-2
- `glue/mrpc` for MRPC

### Important Implementation Details

**Multi-Dataset Architecture**:
- **Unified interface**: Single CLI supports all 4 datasets
- **Dataset-specific logic**: Each dataset has tailored text/label column handling
- **Flexible sampling**: Instruction length (Alpaca) vs label balance (classification)

**Label Management**:
- **Automatic preservation**: Classification datasets auto-preserve labels during text noise
- **Validation checks**: Warns if labels accidentally change during text noise injection
- **Label flipping**: Clean 0↔1 mapping for supported classification datasets

**Noise Guarantees**:
- **No partial failures**: Force-apply mechanism prevents unchanged samples
- **Deterministic**: Uses fixed random seeds for reproducibility
- **Memory efficient**: Processes large datasets without loading everything into memory
- **Resumable**: Cache system allows interrupted downloads to resume

**Quality Assurance**:
- **Dataset-aware validation**: Text vs label column tracking per dataset
- **Change verification**: Ensures noise actually applied with dataset-specific checks
- **Comprehensive analysis**: 6 analysis modes with multi-dataset support