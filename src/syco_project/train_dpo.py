from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA DPO training for exported pairs")
    parser.add_argument("--model", required=True, help="Base Hugging Face model/path")
    parser.add_argument("--train-file", required=True, help="DPO JSONL from syco-export-dpo")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7650)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Heavy training dependencies are imported only for this command.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    if len(dataset) < 2:
        raise ValueError("At least two preference pairs are required")
    if args.eval_fraction > 0 and len(dataset) >= 10:
        split = dataset.train_test_split(
            test_size=args.eval_fraction, seed=args.seed, shuffle=True
        )
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = dataset, None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=100 if eval_dataset is not None else None,
        bf16=not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs={
            "dtype": torch.float16 if args.fp16 else torch.bfloat16,
        },
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = DPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

