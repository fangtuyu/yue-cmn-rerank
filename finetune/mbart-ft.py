from transformers import MBartForConditionalGeneration, AutoTokenizer
import torch
import os

checkpoint_path = "../output/best_model"
test_file = "../data/data_test.txt"
output_dir = "../../phrase-based-canto-mando/result"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
model = MBartForConditionalGeneration.from_pretrained(checkpoint_path).to(device)

new_token = "<yue>"
if new_token not in tokenizer.get_vocab():
    tokenizer.add_tokens([new_token])
    model.resize_token_embeddings(len(tokenizer))

tokenizer.src_lang = "zh_CN"

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    sources = [line.split("\t")[0] for line in lines]
    return sources

test_sources = load_data(test_file)

def preprocess(sources):
    return [new_token + src for src in sources]

inputs = preprocess(test_sources)
all_predictions = []

model.eval()
with torch.no_grad():
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        encoded = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        generated = model.generate(
            **encoded,
            max_new_tokens=128,
            num_beams=10,
            num_return_sequences=1
        )
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_predictions.extend(preds)

        if (i // batch_size) % 10 == 0:
            print(f"deal {i}/{len(inputs)}")

decoded_preds = all_predictions

result_file = os.path.join(output_dir, "result.txt")
os.makedirs(output_dir, exist_ok=True)
with open(result_file, "w", encoding="utf-8") as f:
    f.write("\n".join(decoded_preds))

print(f"translate over，saved: {result_file}")