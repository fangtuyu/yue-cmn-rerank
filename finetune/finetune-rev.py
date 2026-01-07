import torch
import os
import logging
import numpy as np
from datasets import Dataset
import sacrebleu

from transformers import (
    MBartForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model():
    set_seed(42) 
    model_name = "../mbart-large-50"
    data_dir = "../../phrase-based-canto-mando/data"
    output_dir = "../outputrev"
    batch_size = 16
    max_length = 128
    new_token = "<yue>"  # before target 
    dev_ratio = 0.1

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(data_dir, "data_train.txt")
    
    if not os.path.exists(train_file):
        logging.error(f"File not found: {train_file}")
        return
    else:
        file_size = os.path.getsize(train_file)
        logging.info(f"File found, size: {file_size} byte")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    if new_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([new_token])
        model.resize_token_embeddings(len(tokenizer))
        logging.info(f"add `{new_token}`")

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 3. cmn → yue
    def load_train_data(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if not lines:
            raise ValueError("empty")

        sources = [line.split("\t")[1].strip() for line in lines]          # cmn
        targets = [new_token + line.split("\t")[0].strip() for line in lines]  # yue
        return sources, targets

    train_sources, train_targets = load_train_data(train_file)
    full_dataset = Dataset.from_dict({"source": train_sources, "target": train_targets})
    split_dataset = full_dataset.train_test_split(test_size=dev_ratio, seed=42)
    train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]

    def save_dataset_to_txt(dataset, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(f"{item['source']}\t{item['target']}\n")

    save_dataset_to_txt(train_dataset, os.path.join(output_dir, "train_split.txt"))
    save_dataset_to_txt(eval_dataset, os.path.join(output_dir, "eval_split.txt"))

    logging.info(f"train set: {len(train_dataset)}, val set: {len(eval_dataset)}")

    # preprocess 
    def preprocess_function(examples):
        tokenizer.src_lang = "zh_CN"     
        tokenizer.tgt_lang = "zh_CN"   

        return tokenizer(
            examples["source"],
            text_target=examples["target"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=["source", "target"])
    eval_tokenized = eval_dataset.map(preprocess_function, batched=True, remove_columns=["source", "target"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    import evaluate

    metric = evaluate.load("sacrebleu", cache_dir="../cache")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]  

        try:
            result = metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                tokenize="zh"  
            )
            print(result)
            bleu_score = result["score"]
        except Exception as e:
            logging.warning(f"BLEU computation failed with evaluate: {e}")
            bleu_score = 0.0

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        gen_len = np.mean(prediction_lens)

        return {
            "bleu": round(bleu_score, 4),
            "gen_len": round(gen_len, 4)
        }

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=1000,
        save_total_limit=3,
        evaluation_strategy="steps",
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logging.info("training....")
    trainer.train()

    # save best model
    best_ckpt = trainer.state.best_model_checkpoint or output_dir
    best_model_dir = os.path.join(output_dir, "best_model")

    MBartForConditionalGeneration.from_pretrained(best_ckpt).save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    logging.info(f"best model is：{best_model_dir}")


if __name__ == "__main__":
    train_model()
