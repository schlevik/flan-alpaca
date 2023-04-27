from typing import Optional
import fire

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_int8_training


from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


def train(
    base_model: str = "google/flan-t5-xxl",
    train_file: str = "data/train.json",
    val_file: Optional[str] = None,
    output_dir: str = "results-xxl-more-adapter",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    per_device_train_batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_train_epochs: int = 4,
    is_ampere: bool = True,
):
    """
    This script is adopted from: https://www.philschmid.de/fine-tune-flan-t5-peft
    """
    dataset = load_dataset("json", data_files=train_file if not val_file else {"train": train_file, "val": val_file})
    print(f"Train dataset size: {len(dataset['train'])}")
    if val_file:
        print(f"Train dataset size: {len(dataset['val'])}")

    # Load tokenizer of FLAN-t5-XXL
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenized_inputs = dataset["train"].map(
        lambda x: tokenizer(x["source"], truncation=True), batched=True, remove_columns=["source", "target"]
    )

    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    max_source_length = int(np.percentile(input_lenghts, 95))
    print(f"Max source length: {max_source_length}")

    tokenized_targets = dataset["train"].map(
        lambda x: tokenizer(x["target"], truncation=True), batched=True, remove_columns=["source", "target"]
    )
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    max_target_length = int(np.percentile(target_lenghts, 95))
    print(f"Max target length: {max_target_length}")

    label_pad_token_id = -100

    def preprocess_function(sample, padding="max_length"):
        inputs = [item for item in sample["source"]]

        # tokenize source
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # tokenize target
        labels = tokenizer(
            text_target=sample["target"], max_length=max_target_length, padding=padding, truncation=True
        )

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else label_pad_token_id) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["source", "target"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, load_in_8bit=True, device_map="auto")

    # Define LoRA Config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # auto_find_batch_size=True,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,  # higher learning rate
        num_train_epochs=num_train_epochs,
        # gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=250,
        save_strategy="no",
        report_to="wandb",
        bf16=True if is_ampere else False,
        tf32=True if is_ampere else False,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = False

    trainer.train()

    model.config.use_cache = True

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
