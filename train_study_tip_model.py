"""Fine-tune GPT-2 for a small study-tip generation demonstration."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

MODEL_NAME = "openai-community/gpt2"
TRAIN_PAIRS = [
    ("Give me a study tip for recursion.", "Trace a small recursive example by hand and identify the base case before writing code."),
    ("How should I study matrices?", "Practise matrix addition and multiplication on small examples, checking dimensions at every step."),
    ("Give a tip for learning binary search.", "Track low, high, and middle indices on paper and check the loop condition after each update."),
    ("How can I remember stack operations?", "Relate push and pop to a pile of plates and trace the top element after each operation."),
    ("How should I prepare for an exam?", "Use active recall: close your notes, explain the idea aloud, and revisit gaps."),
    ("Give me a tip for debugging Python.", "Read the traceback from the bottom, inspect the failing line, and reproduce the issue with a small input."),
]


class StudyTipDataset(Dataset):
    def __init__(self, tokenizer, max_length=128):
        self.examples = []
        for question, answer in TRAIN_PAIRS:
            text = f"Question: {question}\nStudy tip: {answer}{tokenizer.eos_token}"
            encoded = tokenizer(text, max_length=max_length, truncation=True, padding="max_length")
            ids = encoded["input_ids"]
            mask = encoded["attention_mask"]
            self.examples.append({
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(mask),
                "labels": torch.tensor([t if m else -100 for t, m in zip(ids, mask)]),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models/study-tip-gpt2")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.eos_token_id
    for p in model.parameters():
        p.requires_grad = False
    for p in model.transformer.h[-1].parameters():
        p.requires_grad = True
    for p in model.transformer.ln_f.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    dataset = StudyTipDataset(tokenizer)
    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "training"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        weight_decay=.01,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
