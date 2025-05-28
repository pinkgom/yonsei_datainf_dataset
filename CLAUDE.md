# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **DataInf noise injection dataset generation tool** for the paper "DATAINF: EFFICIENTLY ESTIMATING DATA INFLUENCE IN LORA-TUNED LLMS AND DIFFUSION MODELS". The project systematically injects noise into Alpaca datasets to create experimental datasets for data influence research.

## Core Commands

### Basic Usage
```bash
# Demo mode (500 samples, 1-2 minutes)
python main.py --demo

# Full mode (52K samples, 15-20 minutes)
python main.py --full --noise-ratio 0.2 --strategy balanced

# Quality analysis
python main.py --analysis

# Cache management
python main.py --cache
```

### Advanced Experiments
```bash
# Multiple noise ratios
python main.py --full --noise-ratios 0.1,0.15,0.2,0.25 --strategy balanced

# All strategies
python main.py --full --noise-ratio 0.2 --strategy all

# Combined experiments
python main.py --full --noise-ratios 0.15,0.2 --strategy balanced,grammar_heavy

# Auto-confirm mode
python main.py --full --noise-ratio 0.2 --strategy balanced --yes
```

### Testing and Validation
No specific test framework is defined. Validation is done through:
- Built-in analysis tools (`--analysis` mode)
- Sample comparison functions
- Quality metrics verification

## Architecture

### Core Components

**main.py**: Main execution script with CLI interface
- Demo mode: Quick 500-sample experiments
- Full mode: Complete 52K dataset experiments  
- Analysis mode: 6 different quality analysis tools
- Cache management: Dataset download/cache control

**src/data_loader.py** (`AlpacaDataLoader`): Dataset management
- Handles Stanford Alpaca dataset (52K samples) via HuggingFace
- Implements caching system to avoid re-downloads
- Supports subset sampling for testing
- JSON file I/O for dataset persistence

**src/noise_injection.py** (`NoiseInjector`): Core noise injection system
- **Guaranteed noise application**: Ensures all targeted samples are actually modified
- **Stratified sampling**: Instruction length-based balanced sampling
- **3 noise strategies**: balanced (40/35/25%), grammar_heavy (60/25/15%), semantic_heavy (20/60/20%)
- **3 noise types**: grammar, semantic, quality degradation

**src/analysis.py**: Comprehensive quality analysis suite
- 6 analysis modes: basic info, demo detailed, full detailed, comparison, strategy effects, comprehensive
- Noise distribution analysis and verification
- DataInf experiment readiness assessment

### Noise Injection Methodology

**Grammar Noise**: Typos, word shuffling, punctuation errors, grammar mistakes
**Semantic Noise**: Wrong context, irrelevant additions, topic drift  
**Quality Noise**: Truncation, duplication, incomplete answers, rambling

### Data Flow

1. **Load**: Alpaca dataset via HuggingFace (cached locally)
2. **Sample**: Stratified sampling based on instruction length
3. **Inject**: Apply guaranteed noise based on strategy weights
4. **Verify**: Ensure actual changes occurred
5. **Save**: Output noisy datasets with metadata
6. **Analyze**: Quality assessment and comparison tools

### File Naming Convention

```
alpaca_original_demo_500.json          # Original demo data
alpaca_demo_20percent_balanced_500.json # Demo with 20% balanced noise
alpaca_full_20percent_52002.json       # Full dataset with 20% noise
alpaca_original_full_52002.json        # Original full dataset
experiment_metadata_*.json              # Experiment metadata
```

### Dependencies

Core: `datasets`, `pandas`, `numpy`  
Optional development tools commented out in requirements.txt

### Important Implementation Details

- **Noise is guaranteed**: The system ensures every targeted sample actually gets modified
- **No partial failures**: Force-apply mechanism prevents unchanged samples
- **Deterministic**: Uses fixed random seeds for reproducibility
- **Memory efficient**: Processes large datasets without loading everything into memory
- **Resumable**: Cache system allows interrupted downloads to resume