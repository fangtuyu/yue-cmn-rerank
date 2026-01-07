import os
os.environ["HF_HOME"] = "../hf_cache"         # model cashe
os.makedirs("../hf_cache", exist_ok=True)

import torch
import json
import logging
from typing import List, Dict, Tuple
from inspect import getfullargspec
from transformers import MBartForConditionalGeneration, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from sacrebleu.metrics import BLEU
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MART_MODEL_DIR = "../output/best_model"
REVERSE_MODEL_DIR = "../outputrev/best_model"
BERT_MODEL_PATH = "../../phrase-based-canto-mando/bert-base-chinese"
DEV_FILE = "../output/eval_split.txt"
TEST_FILE = "../data_test.txt"
OUTPUT_FILE = "../rerank_result.txt"

NUM_BEAMS = 10
CACHE_DIR = "../cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

LAMBDA_PLL_CANDIDATES = [0.0, 0.1, 0.15, 0.2]
LAMBDA_REV_CANDIDATES = [0.0, 0.1, 0.2, 0.3]
LAMBDA_LLM_CANDIDATES = [0.0, 0.1, 0.2, 0.3]  

SACREBLEU_CFG = {
    "tokenize": "zh",
    "smooth_method": "exp",
    "lowercase": False,
}

def compute_bleu_with_sample(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> float:
    kwargs = {}
    if sacrebleu_cfg:
        valid_keys = getfullargspec(BLEU).args
        for k, v in sacrebleu_cfg.items():
            if k in valid_keys:
                kwargs[k] = v
    metric = BLEU(**kwargs)
    score = metric.corpus_score(hypotheses=hypotheses, references=[references]).score
    logger.info(f"BLEU signature: {metric.get_signature()}")
    return score

# ——————————————————————————————
# PLLRescorer（BERT-based）
# ——————————————————————————————
class PLLRescorer:
    def __init__(self, bert_model_path: str, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(bert_model_path).to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    def _compute_pll_single(self, sentence: str) -> float:
        if not sentence.strip():
            return -1e5
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # 
            add_special_tokens=True
        ).to(self.device)
        input_ids = inputs["input_ids"][0]
        L = input_ids.size(0)
        total_log_prob = 0.0
        num_tokens = 0
        with torch.no_grad():
            for i in range(1, L - 1):
                masked_input = input_ids.clone()
                original_id = masked_input[i].item()
                masked_input[i] = self.tokenizer.mask_token_id
                logits = self.model(masked_input.unsqueeze(0)).logits[0, i]         
                log_probs = torch.log_softmax(logits, dim=-1)
                total_log_prob += log_probs[original_id].item()
                num_tokens += 1
        return total_log_prob / num_tokens if num_tokens > 0 else -1e5

    def compute_pll_batch(self, sentences: List[str]) -> List[float]:
        return [self._compute_pll_single(sent) for sent in sentences]
    
# ——————————————————————————————
# LLM Fluency Scorer（Qwen2.5-7B）
# ——————————————————————————————
class LLMFluencyScorer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B", device_map="auto", alpha=0.8):
        logger.info("load Qwen2.5-7B LLM for fluency scoring...")
        self.alpha = alpha
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        torch.set_grad_enabled(False)
        logger.info("Qwen2.5-7B load！")
        
    def score_sentence(self, sentence: str) -> float:
        if not sentence.strip():
            return -1e5
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]
        if seq_len <= 1:
            return -1e5
        with torch.no_grad():
            labels = input_ids.clone()
            labels[:, 0] = -100  
            outputs = self.model(input_ids=input_ids, labels=labels)
            per_token_nll = outputs.loss.item()
            return -per_token_nll
    
    def compute_llm_batch(self, sentences: List[str]) -> List[float]:
        return [self.score_sentence(sent) for sent in sentences]

# ——————————————————————————————
# candidate generate
# ——————————————————————————————
def load_mart_model(model_dir: str, device, add_yue_token=False):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = MBartForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    if add_yue_token:
        new_token = "<yue>"
        if new_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([new_token])
            model.resize_token_embeddings(len(tokenizer))
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(new_token)
    return model, tokenizer

def generate_nbest(model, tokenizer, source_text: str, num_beams: int, device) -> Tuple[List[str], List[float]]:
    if not source_text.startswith("<yue>"):
        source_text = "<yue>" + source_text
    inputs = tokenizer(
        source_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128, 
            num_beams=num_beams,
            num_return_sequences=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )
    candidates = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    candidates = [s.strip() for s in candidates]
    scores = outputs.sequences_scores.cpu().tolist()
    return candidates, scores

# ——————————————————————————————
# rev model scorer
# ——————————————————————————————
def compute_reverse_scores(reverse_model, reverse_tokenizer, original_source: str, candidate_targets: List[str], device) -> List[float]:
    scores = []
    for y in candidate_targets:
        if not y.strip():
            scores.append(-1e5)
            continue
        input_ids = reverse_tokenizer(
            y,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True
        ).input_ids.to(device)
        target_text = "<yue>" + original_source
        labels = reverse_tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True
        ).input_ids.to(device)
        with torch.no_grad():
            outputs = reverse_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            num_tokens = (labels != reverse_tokenizer.pad_token_id).sum().item()
            avg_log_prob = -loss.item() 
            scores.append(avg_log_prob)
    return scores

def load_or_generate_nbest_cache(model, tokenizer, input_file: str, num_beams: int, cache_file: str, device):
    if os.path.exists(cache_file):
        logger.info(f"Cache loading N-best: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
        return {k: (v["candidates"], v["scores"]) for k, v in raw_cache.items()}
    logger.info(f"Generate N-best and cache to: {cache_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    cache = {}
    for idx, line in enumerate(lines):
        parts = line.split("\t")
        src = parts[0].strip() if parts else ""
        if not src:
            continue
        try:
            candidates, scores = generate_nbest(model, tokenizer, src, num_beams, device)
            cache[src] = {"candidates": candidates, "scores": scores}
        except Exception as e:
            logger.warning(f"Generate No {idx+1} wrong: {e}")
            cache[src] = {"candidates": [""], "scores": [-1e5]}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return {k: (v["candidates"], v["scores"]) for k, v in cache.items()}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"use: {device}")

    mart_model, mart_tokenizer = load_mart_model(MART_MODEL_DIR, device)
    reverse_model, reverse_tokenizer = load_mart_model(REVERSE_MODEL_DIR, device, add_yue_token=True)
    yue_id = reverse_tokenizer.convert_tokens_to_ids("<yue>")
    assert yue_id != reverse_tokenizer.unk_token_id, "<yue> token no load"

    pll_rescorer = PLLRescorer(BERT_MODEL_PATH, device)

    llm_scorer = LLMFluencyScorer(model_name="Qwen/Qwen2.5-7B", alpha=0.8)

    with open(DEV_FILE, "r", encoding="utf-8") as f:
        dev_lines = [line.strip() for line in f if line.strip()]
    dev_sources, dev_references = [], []
    for line in dev_lines:
        parts = line.split("\t")
        dev_sources.append(parts[0].strip())
        dev_references.append(parts[1].strip() if len(parts) > 1 else "")
    logger.info(f"val set: {len(dev_lines)} ")

    dev_cache_file = os.path.join(CACHE_DIR, "nbest_cache_dev.json")
    nbest_cache = load_or_generate_nbest_cache(
        mart_model, mart_tokenizer, DEV_FILE, NUM_BEAMS, dev_cache_file, device
    )

    # Calculate all scores in advance
    logger.info("Calculate the PLL, reverse, and LLM fluency scores...")
    precomputed_scores = {}  # src -> (fw, mlm, rev, llm)
    for src in dev_sources:
        if src not in nbest_cache:
            precomputed_scores[src] = ([0.0], [0.0], [-1e5], [-1e5])
            continue
        nbest_list, nbest_scores = nbest_cache[src]
        pll_scores = pll_rescorer.compute_pll_batch(nbest_list)
        rev_scores = compute_reverse_scores(reverse_model, reverse_tokenizer, src, nbest_list, device)
        llm_scores = llm_scorer.compute_llm_batch(nbest_list)  
        
        assert len(nbest_scores) == len(pll_scores) == len(rev_scores) == len(llm_scores), \
        f"[DEV] Score length mismatch: fw={len(nbest_scores)}, pll={len(pll_scores)}, rev={len(rev_scores)}, llm={len(llm_scores)}"
        
        precomputed_scores[src] = (nbest_scores, pll_scores, rev_scores, llm_scores)

    # Hyperparameter search (4D combination)
    best_bleu = -1
    best_lambda_pll = best_lambda_rev = best_lambda_llm = None

    total = len(LAMBDA_PLL_CANDIDATES) * len(LAMBDA_REV_CANDIDATES) * len(LAMBDA_LLM_CANDIDATES)
    current = 0
    logger.info("search")

    GRID_LOG_FILE = "../grid_search_log.txt"
    os.makedirs(os.path.dirname(GRID_LOG_FILE), exist_ok=True)

    for lam_pll in LAMBDA_PLL_CANDIDATES:
        for lam_rev in LAMBDA_REV_CANDIDATES:
            for lam_llm in LAMBDA_LLM_CANDIDATES:
                current += 1
                logger.info(f"[{current}/{total}] test λ_pll={lam_pll}, λ_rev={lam_rev}, λ_llm={lam_llm}")
                predictions = []
                debug_printed = 0

                for src in dev_sources:
                    if src not in precomputed_scores:
                        predictions.append("")
                        continue
                    nbest_scores, pll_scores, rev_scores, llm_scores = precomputed_scores[src]
                    combined = [
                        nbest_scores[i] + lam_pll * pll_scores[i] + lam_rev * rev_scores[i] + lam_llm * llm_scores[i]
                        for i in range(len(nbest_scores))
                    ]
                    best_idx = int(torch.argmax(torch.tensor(combined)).item())
                    predictions.append(nbest_cache[src][0][best_idx])

                    # Debug output (only the first 3 samples of the first combination)
                    if (
                        lam_pll == LAMBDA_PLL_CANDIDATES[0]
                        and lam_rev == LAMBDA_REV_CANDIDATES[0]
                        and lam_llm == LAMBDA_LLM_CANDIDATES[0]
                        and debug_printed < 3
                    ):
                        nbest_list = nbest_cache[src][0]
                        logger.info(f"\n sample {debug_printed + 1}:")
                        logger.info(f"  source: {src}")
                        top_k = min(3, len(nbest_list))
                        for i in range(top_k):
                            cand = nbest_list[i]
                            fw = nbest_scores[i]
                            pll = pll_scores[i]
                            rev = rev_scores[i]
                            llm = llm_scores[i]
                            comb_val = fw + lam_pll * pll + lam_rev * rev + lam_llm * llm
                            logger.info(f"    [{i}] '{cand}'")
                            logger.info(f"        fw: {fw:.4f}, PLL: {pll:.4f}, rev: {rev:.4f}, LLM: {llm:.4f} → combined: {comb_val:.4f}")
                        debug_printed += 1

                bleu_score = compute_bleu_with_sample(predictions, dev_references, **SACREBLEU_CFG)
                logger.info(f"   → BLEU = {bleu_score:.4f}")

                with open(GRID_LOG_FILE, "a", encoding="utf-8") as log_f:
                    log_f.write(f"λ_pll={lam_pll}, λ_rev={lam_rev}, λ_llm={lam_llm} → BLEU={bleu_score:.4f}\n")
               
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    best_lambda_pll = lam_pll
                    best_lambda_rev = lam_rev
                    best_lambda_llm = lam_llm

    logger.info(f"best weights: λ_pll={best_lambda_pll}, λ_rev={best_lambda_rev}, λ_llm={best_lambda_llm} (BLEU={best_bleu:.2f})")

    # ——————————————————————————————
    # Test set inference (with best parameters)
    # ——————————————————————————————
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_sources = [line.split("\t")[0].strip() for line in f if line.strip()]

    test_cache_file = os.path.join(CACHE_DIR, "nbest_cache_test.json")
    test_nbest_cache = load_or_generate_nbest_cache(
        mart_model, mart_tokenizer, TEST_FILE, NUM_BEAMS, test_cache_file, device
    )

    final_results = []
    detailed_output = {}  

    for src in test_sources:
        if src not in test_nbest_cache:
            final_results.append("")
            detailed_output[src] = []
            continue
        nbest_list, nbest_scores = test_nbest_cache[src]
        pll_scores = pll_rescorer.compute_pll_batch(nbest_list)
        rev_scores = compute_reverse_scores(reverse_model, reverse_tokenizer, src, nbest_list, device)
        llm_scores = llm_scorer.compute_llm_batch(nbest_list)

        combined = [
            nbest_scores[i] + best_lambda_pll * pll_scores[i] + best_lambda_rev * rev_scores[i] + best_lambda_llm * llm_scores[i]
            for i in range(len(nbest_list))
        ]
        best_idx = int(torch.argmax(torch.tensor(combined)).item())
        final_results.append(nbest_list[best_idx])

        # save
        candidates_detail = []
        for i in range(len(nbest_list)):
            candidates_detail.append({
                "candidate": nbest_list[i],
                "forward_score": float(nbest_scores[i]),
                "pll_score": float(pll_scores[i]),
                "reverse_score": float(rev_scores[i]),
                "llm_score": float(llm_scores[i]),
                "combined_score": float(combined[i])
            })
        detailed_output[src] = candidates_detail

    # save result
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in final_results:
            f.write(res + "\n")

    # save JSON
    DETAILED_JSON_FILE = OUTPUT_FILE.replace(".txt", "results.json")
    with open(DETAILED_JSON_FILE, "w", encoding="utf-8") as f_json:
        json.dump(detailed_output, f_json, ensure_ascii=False, indent=2)

    logger.info(f"result saved: {OUTPUT_FILE}")
    logger.info(f"score of result saved: {DETAILED_JSON_FILE}")
    logger.info(f"best weights: λ_pll={best_lambda_pll}, λ_rev={best_lambda_rev}, λ_llm={best_lambda_llm}")
    
if __name__ == "__main__":
    main()
