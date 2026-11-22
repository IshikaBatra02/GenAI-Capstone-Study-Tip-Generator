# Capstone — Generative AI Study Tip Generator

A small GPT-2-based study-tip generator with a Streamlit interface. The project includes a fine-tuning script, a human-evaluation template, and a lightweight app for interactive generation.

## Requirements and setup
Python 3.10 or 3.11 recommended.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Fine-tune
```bash
python train_study_tip_model.py --epochs 5
```
The script fine-tunes the final GPT-2 Transformer block on the included example study-tip pairs and saves the model to `models/study-tip-gpt2/`.

## 2. Launch the demo
```bash
streamlit run app.py
```
The app loads `models/study-tip-gpt2/` if it exists; otherwise it uses the base GPT-2 model.

## 3. Evaluate at least 10 held-out prompts
Use prompts that are not present in the training examples. For each output, score:
- Relevance: 0–2
- Readability: 0–2
- Usefulness: 0–2

Maximum total: 6 per prompt. Record the score and a short rationale in `evaluation_template.csv`. Include failure analysis (irrelevant, repetitive, vague, or incorrect tips) and human review. Do not treat model-generated outputs as verified advice.

## Reproducibility
Seed 42 is used by default. A GPU is recommended but not mandatory. First-run model downloads require internet access. Training metrics and human scores should be generated and recorded by the person running the experiment.
