# BERTScore Evaluation for Concept-Aware LLMs

This module provides BERTScore evaluation functionality to assess how well generated outputs match human intuition by comparing them against human reference responses.

## Overview

BERTScore is a text generation evaluation metric that computes a similarity score for each token in the candidate sentence with each token in the reference sentence using contextual embeddings. This helps evaluate how semantically similar the generated text is to human responses.

## Features

- **Automatic Evaluation**: Compare baseline and concept decoder outputs against human references
- **Multiple Reference Support**: Evaluate against multiple human responses for robust scoring
- **Comprehensive Metrics**: Precision, Recall, and F1 scores
- **Detailed Reporting**: Formatted output with improvement analysis
- **Result Persistence**: Save evaluation results to JSON files
- **Flexible Configuration**: Choose different BERT models for scoring

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The `bert-score` package will be automatically installed, which includes the necessary BERT models.

## Usage

### Basic Usage with Main Script

The BERTScore evaluation is integrated into the main concept decoder script:

```bash
python main.py --enable_bertscore
```

This will:
1. Generate outputs using both baseline and concept decoder methods
2. Compare them against human reference responses
3. Display detailed evaluation results
4. Save results to a JSON file

### Interactive Configuration

When running interactively, you can configure BERTScore settings:

```
--- BERTScore Evaluation Settings ---
Enable BERTScore evaluation? (yes/no) [default: yes]: yes
BERTScore model [default: microsoft/deberta-v3-large]: 
```

### Command Line Options

- `--enable_bertscore`: Enable/disable BERTScore evaluation (default: True)
- `--bertscore_model`: Choose BERT model for scoring (default: microsoft/deberta-v3-large)

### Standalone Testing

Test the BERTScore functionality independently:

```bash
python test_bertscore.py
```

## How It Works

### 1. Human Reference Collection

The system uses predefined human reference responses for common prompts:

```python
references = get_human_references_for_prompt("Name something parents would criticize their children for having.")
# Returns: ["messy room", "bad grades", "lazy attitude", ...]
```

### 2. BERTScore Calculation

For each generated output, BERTScore computes:
- **Precision**: How much of the candidate is semantically similar to the reference
- **Recall**: How much of the reference is captured in the candidate
- **F1**: Harmonic mean of precision and recall

### 3. Multiple Reference Evaluation

When multiple human references are available, the system:
- Evaluates against each reference separately
- Reports the best score (highest F1)
- Reports the average score across all references

### 4. Comparison Analysis

The system compares baseline vs concept decoder performance:
- Shows individual scores for each method
- Calculates improvement metrics
- Provides clear visual indicators of which method performs better

## Example Output

```
================================================================================
BERTScore Evaluation Results
================================================================================
Prompt: Name something parents would criticize their children for having.
Timestamp: 2024-01-15T10:30:45.123456

----------------------------------------
BASELINE DECODER
----------------------------------------
Output: lazy attitude and not doing homework
Best F1: 0.8234
Avg F1: 0.7123
Best Precision: 0.8567
Best Recall: 0.7890

----------------------------------------
CONCEPT DECODER
----------------------------------------
Output: messy room and bad grades
Best F1: 0.9123
Avg F1: 0.8234
Best Precision: 0.9234
Best Recall: 0.9012

----------------------------------------
IMPROVEMENT
----------------------------------------
F1 Improvement: +0.0889
Precision Improvement: +0.0667
Recall Improvement: +0.1122
✅ Concept decoder performs better than baseline

----------------------------------------
HUMAN REFERENCES
----------------------------------------
1. messy room
2. bad grades
3. lazy attitude
4. disrespectful behavior
5. spending too much time on phone
================================================================================
```

## Supported Prompts

The system includes human reference responses for common prompts:

1. **Parent Criticism**: "Name something parents would criticize their children for having."
2. **Travel Items**: "What is something people often forget to bring when traveling?"
3. **Polarizing Foods**: "Name a food that people either love or hate."

## Customization

### Adding New Prompts

To add support for new prompts, modify the `get_human_references_for_prompt()` function in `bertscore_evaluator.py`:

```python
reference_mappings = {
    "Your new prompt here": [
        "human response 1",
        "human response 2",
        "human response 3",
        # ... more responses
    ],
    # ... existing mappings
}
```

### Using Different BERT Models

You can specify different BERT models for scoring:

```python
evaluator = BERTScoreEvaluator(model_type="bert-base-uncased")
```

Available models include:
- `bert-base-uncased` (default, recommended)
- `roberta-base`
- `distilbert-base-uncased`
- `microsoft/deberta-v3-large` (may have tokenizer compatibility issues)

## Result Files

Evaluation results are saved as JSON files with timestamps:
- `bertscore_evaluation_20240115_103045.json`

The JSON contains:
- Complete evaluation metrics
- Generated outputs
- Human references
- Improvement analysis
- Timestamp and configuration

## Troubleshooting

### Common Issues

1. **BERTScore Model Download**: First run may take time to download the BERT model
2. **Memory Usage**: Large models may require significant RAM/VRAM
3. **CUDA Issues**: Ensure PyTorch is installed with CUDA support if using GPU

### Error Messages

- `ModuleNotFoundError: No module named 'bert_score'`: Install with `pip install bert-score`
- `CUDA out of memory`: Use a smaller BERT model or CPU evaluation
- `No human references found`: Add the prompt to the reference mappings

## Performance Considerations

- **Model Size**: Larger models (deberta-v3-large) provide better accuracy but slower evaluation
- **Batch Processing**: For multiple evaluations, consider batching for efficiency
- **Caching**: BERTScore models are cached after first load for faster subsequent runs

## Research Applications

This BERTScore evaluation is particularly useful for:

1. **Concept Decoder Validation**: Verify that concept-aware decoding produces more human-like outputs
2. **A/B Testing**: Compare different decoding strategies
3. **Model Fine-tuning**: Use scores as feedback for model improvement
4. **Human Alignment**: Ensure generated text aligns with human expectations

## Citation

If you use this BERTScore evaluation in your research, please cite:

```bibtex
@inproceedings{zhang2020bertscore,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Zhang, Tianyi and Kishore, Varsha and Wu, Felix and Weinberger, Kilian Q and Artzi, Yoav},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
