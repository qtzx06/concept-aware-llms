# Concept-Aware LLM Decoding Benchmark

This project provides a framework to benchmark and compare decoding strategies for language models, focusing on reasoning and truthful generation. The primary goal is to iterate on a `ConceptDecoder` to improve performance over a standard baseline.

### Current Status

The project is set up with a stable environment and two primary benchmarks. We have established baseline performance on the TruthfulQA dataset and are now focused on improving the logic of the `ConceptDecoder`.

**Latest Baseline Scores (TruthfulQA):**
- **ROUGE-1:** `0.1766`
- **ROUGE-L:** `0.1397`
- **BLEU:** `2.6144`

### Benchmarks

*   **TruthfulQA:** A generation dataset that measures a model's ability to be truthful and avoid generating common falsehoods. Performance is measured with metrics like ROUGE and BLEU.
*   **FOLIO:** A deductive reasoning dataset. The model is given premises and must determine if a conclusion is `True`, `False`, or `Unknown`.

### How to Run

1.  **Setup Environment:**
    *   Clone the required datasets into the project's root directory:
        ```bash
        git clone https://github.com/sylinrl/TruthfulQA.git
        git clone https://github.com/Yale-LILY/FOLIO.git
        ```
    *   Create and activate a virtual environment (Python 3.12 recommended):
        ```bash
        uv venv
        source .venv/bin/activate
        ```
    *   Install dependencies:
        ```bash
        uv pip install -r requirements.txt
        ```

2.  **Run an Experiment:**
    Use the `main.py` script with the `--dataset` and `--experiment` flags.

    *   **TruthfulQA Baseline:**
        ```bash
        python main.py --dataset truthfulqa --experiment baseline
        ```
    *   **TruthfulQA Concept-Aware:**
        ```bash
        python main.py --dataset truthfulqa --experiment concept_aware
        ```
    *   **FOLIO Baseline:**
        ```bash
        python main.py --dataset folio --experiment baseline


python main.py --experiment concept_aware --dataset truthfulqa --num_examples 1 --decoding_strategy entropy_gated 
python main.py --experiment baseline --dataset truthfulqa --num_examples 1 --decoding_strategy entropy_gated