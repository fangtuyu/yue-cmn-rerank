import os
import re
from comet import download_model, load_from_checkpoint

SRC_FILE = "yue510.txt"
MT_FILE = "ours510.txt"
REF_FILE = "zh510.txt"
BATCH_SIZE = 16
USE_CPU = False
OUTPUT_DIR = "normalized_comet_inputs"

# Process: Punctuation standardization
def normalize_punctuation(text):
    """
    Convert half-width punctuation to full-width for Chinese text standardization.
    """
    if not text:
        return ""
    half_width_chars = ",.!?;:\"'()[]{}<>/-"
    full_width_chars = "，。！？；：“‘（）【】｛｝《》／－"
    trans_table = str.maketrans(half_width_chars, full_width_chars)
    text = text.translate(trans_table)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_triplets(src_file, mt_file, ref_file):
    def read_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip('\n\r').strip() for line in f]

    src_lines = read_lines(src_file)
    mt_lines = read_lines(mt_file)
    ref_lines = read_lines(ref_file)

    assert len(src_lines) == len(mt_lines) == len(ref_lines), \
        f"lines wrong: src={len(src_lines)}, mt={len(mt_lines)}, ref={len(ref_lines)}"

    normalized_src = []
    normalized_mt = []
    normalized_ref = []

    for s, m, r in zip(src_lines, mt_lines, ref_lines):
        normalized_src.append(normalize_punctuation(s))
        normalized_mt.append(normalize_punctuation(m))
        normalized_ref.append(normalize_punctuation(r))

    return normalized_src, normalized_mt, normalized_ref


def save_scores(filename, scores):
    with open(filename, "w", encoding="utf-8") as f:
        for score in scores:
            f.write(f"{score:.6f}\n")

def main():
    norm_src, norm_mt, norm_ref = load_triplets(SRC_FILE, MT_FILE, REF_FILE)
    n = len(norm_src)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    src_out = os.path.join(OUTPUT_DIR, "normalized_src.txt")
    mt_out = os.path.join(OUTPUT_DIR, "normalized_ot.txt")
    ref_out = os.path.join(OUTPUT_DIR, "normalized_ref.txt")

    with open(src_out, "w", encoding="utf-8") as f:
        f.write("\n".join(norm_src) + "\n")
    with open(mt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(norm_mt) + "\n")
    with open(ref_out, "w", encoding="utf-8") as f:
        f.write("\n".join(norm_ref) + "\n")

    print(f"saved'{OUTPUT_DIR}' have：")
    print(f"   - {src_out}")
    print(f"   - {mt_out}")
    print(f"   - {ref_out}")

    comet_data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(norm_src, norm_mt, norm_ref)]

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    scores = model.predict(
        comet_data,
        batch_size=BATCH_SIZE,
        gpus=0
    )

    print(f"COMET: {scores.system_score:.4f}")
    save_scores("comet_scores.txt", scores.scores)

if __name__ == "__main__":
    for f in [SRC_FILE, MT_FILE, REF_FILE]:
        assert os.path.exists(f), f"No files: {os.path.abspath(f)}"

    main()