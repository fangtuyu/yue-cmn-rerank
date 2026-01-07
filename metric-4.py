import os
import torch
from bert_score import BERTScorer
import sacrebleu
from sacrebleu import corpus_bleu
import jieba
import re

# Process: Punctuation standardization
def normalize_punctuation(text):
    """
    Convert half-width punctuation to full-width for Chinese text standardization.
    """
    half_width_chars = ",.!?;:\"'()[]{}<>/-"
    full_width_chars = "，。！？；：“‘（）【】｛｝《》／－"
    trans_table = str.maketrans(half_width_chars, full_width_chars)
    text = text.translate(trans_table)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tool Function: Chinese Segmentation
def tokenize_zh(text):
    """ jieba """
    return " ".join(jieba.lcut(text))

def read_and_normalize(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return [normalize_punctuation(line) for line in lines]

def compute_chrf(references, hypotheses):
    chrf = sacrebleu.CHRF()
    score = chrf.corpus_score(hypotheses, [references])
    return score.score

def compute_chrfpp(references, hypotheses):
    chrfpp = sacrebleu.CHRF(char_order=6, word_order=2, beta=2)
    score = chrfpp.corpus_score(hypotheses, [references])
    return score.score

def compute_bertscore(predictions, references, use_gpu=False):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    scorer = BERTScorer(model_type="bert-base-chinese", lang="zh", device=device)
    P, R, F = scorer.score(predictions, references)
    return {
        "bert_p": P.mean().item(),
        "bert_r": R.mean().item(),
        "bert_f1": F.mean().item()
    }

if __name__ == "__main__":

    ref_file = "../phrase-based-canto-mando/zh2_test.txt"
    pred_file = "../result.txt"
    out_path = '../metric_result.txt'

    normalized_ref_out = '../normalized_refs_qj_zh.txt'
    normalized_pred_out = '../normalized_preds_qj_bs_result.txt'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    refs = read_and_normalize(ref_file)
    preds = read_and_normalize(pred_file)

    with open(normalized_ref_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(refs) + '\n')
    with open(normalized_pred_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(preds) + '\n')

    chrf = compute_chrf(refs, preds)
    chrfpp = compute_chrfpp(refs, preds)
    assert len(preds) == len(refs), f"Length does not match: {len(preds)} vs {len(refs)}"
    sacre_corpus_bleu_score = corpus_bleu(preds, [refs], tokenize='zh')
    bert_scores = compute_bertscore(preds, refs, use_gpu=False)
    bert_p = bert_scores["bert_p"]
    bert_r = bert_scores["bert_r"]
    bert_f1 = bert_scores["bert_f1"]
    print("BERTScore finish")

    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(f"Prediction File: {pred_file}\n")
        out.write(f"Reference File:  {ref_file}\n")
        out.write(f"Normalized Reference Saved To: {normalized_ref_out}\n")
        out.write(f"Normalized Prediction Saved To: {normalized_pred_out}\n")
        out.write(f"sacreBLEU (corpus-level, tokenize='zh'): {sacre_corpus_bleu_score.score:.4f}\n")
        out.write(f"ChrF (sacreBLEU): {chrf:.4f}\n")
        out.write(f"ChrF++ (sacreBLEU, char+word): {chrfpp:.4f}\n")
        # --- BERTScore ---
        out.write(f"BERTScore Precision (P): {bert_p:.4f}\n")
        out.write(f"BERTScore Recall (R):    {bert_r:.4f}\n")
        out.write(f"BERTScore F1:           {bert_f1:.4f}\n")

    print(f"sacreBLEU (corpus-level, tokenize='zh'): {sacre_corpus_bleu_score.score:.2f}")
    print(f"ChrF (sacreBLEU): {chrf:.2f}")
    print(f"ChrF++ (char+word): {chrfpp:.2f}")
    # --- BERTScore ---
    print(f"BERTScore Precision:    {bert_p:.4f}")
    print(f"BERTScore Recall:       {bert_r:.4f}")
    print(f"BERTScore F1:           {bert_f1:.4f}")

    print(f"\nThe standardized reference translation has been saved to: {normalized_ref_out}")
    print(f"The standardized prediction results are saved at: {normalized_pred_out}")
    print(f"All metrics have been saved to: {out_path}")