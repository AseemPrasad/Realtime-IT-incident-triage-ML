from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .artifacts import ensure_dir, dump_json
from .logs import HeuristicLogAnalyzer, build_chat_messages, derive_ground_truth


def build_dataset(
    frame: pd.DataFrame,
    limit: int | None = None,
    teacher_annotator: Any | None = None,
) -> list[dict]:
    analyzer = HeuristicLogAnalyzer()
    return analyzer.training_rows(frame, limit=limit, teacher_annotator=teacher_annotator)


def export_jsonl(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_train_lora(dataset_path: Path, output_dir: Path, model_id: str) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LoRA training requires torch, transformers, datasets, peft, and accelerate."
        ) from exc

    data = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hf_dataset = Dataset.from_list(data)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    def tokenize_row(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, truncation=True, max_length=256, padding="max_length")
        tokens["labels"] = [
            token if token != tokenizer.pad_token_id else -100 for token in tokens["input_ids"]
        ]
        return tokens

    tokenized = hf_dataset.map(tokenize_row, batched=False, remove_columns=["messages"])
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    args = TrainingArguments(
        output_dir=str(output_dir / "slm_lora_output"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        num_train_epochs=2,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        use_cpu=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()
    save_dir = output_dir / "slm_lora_adapter"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and optionally train the log SLM")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    frame = pd.read_csv(args.input)
    rows = build_dataset(frame, limit=args.limit)
    dataset_path = args.output_dir / "hdfs_sft_dataset.jsonl"
    export_jsonl(rows, dataset_path)
    dump_json(args.output_dir / "hdfs_sft_dataset_preview.json", rows[:3])
    print(f"Saved SFT dataset to {dataset_path}")

    if args.train and not args.dry_run:
        maybe_train_lora(dataset_path, args.output_dir, args.model_id)
        print(f"Saved LoRA adapter under {args.output_dir / 'slm_lora_adapter'}")
    else:
        print("Dry run complete. Pass --train to run LoRA fine-tuning.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

