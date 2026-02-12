import csv
import json
from collections import Counter

EVAL_CSV = ""

EVENT_VERBS = []
DEFINITE_TERMS = []
ACOUSTIC_TERMS = []

with open("data/lexical_vocab.json", "r", encoding="utf-8") as f:
    VOCAB = json.load(f)

EVENT_VERBS = VOCAB["EVENT_VERBS"]
DEFINITE_TERMS = VOCAB["DEFINITE_TERMS"]
ACOUSTIC_TERMS = VOCAB["ACOUSTIC_TERMS"]

event_freqs = []
def_freqs = []
acoustic_freqs = []

event_freqs_hall = []
def_freqs_hall = []
acoustic_freqs_hall = []


def count_matches(text, vocab):
    count = 0
    for term in vocab:
        if term in text:
            count += 1
    return count

def evaluate_hallucination(csv_path: str):
    total = 0
    hall_count = 0

    type_counter_all = Counter()      
    type_counter_hall = Counter()     
    combo_counter = Counter()         

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1

            h = row.get("hallucination_detected", "")
            if isinstance(h, str):
                h = h.strip().lower()
                if h in ["true", "1", "yes"]:
                    h_flag = True
                elif h in ["false", "0", "no", ""]:
                    h_flag = False
                else:
                    h_flag = False
            else:
                h_flag = bool(h)

            types_str = row.get("hallucination_types", "").strip()
            if not types_str:
                types = []
            else:
                try:
                    types = json.loads(types_str)
                    if not isinstance(types, list):
                        types = [types]
                except Exception:
                    types = []


            if h_flag:
                hall_count += 1

            unique_types = set(t for t in types if t)
            for t in unique_types:
                type_counter_all[t] += 1
                if h_flag:
                    type_counter_hall[t] += 1

            if unique_types:
                combo_key = tuple(sorted(unique_types))
                combo_counter[combo_key] += 1

            caption = row.get("model_caption", "")

            caption_lc = caption.lower()
            tokens = caption_lc.split()
            token_count = max(len(tokens), 1)  
            event_cnt = count_matches(caption_lc, EVENT_VERBS)
            def_cnt = count_matches(caption_lc, DEFINITE_TERMS)
            acoustic_cnt = count_matches(caption_lc, ACOUSTIC_TERMS)

            event_freq = event_cnt / token_count
            def_freq = def_cnt / token_count
            acoustic_freq = acoustic_cnt / token_count

            event_freqs.append(event_freq)
            def_freqs.append(def_freq)
            acoustic_freqs.append(acoustic_freq)

            if h_flag:
                event_freqs_hall.append(event_freq)
                def_freqs_hall.append(def_freq)
                acoustic_freqs_hall.append(acoustic_freq)

    # ==== 计算指标 ====
    if total == 0:
        print("No samples found.")
        return

    hr = hall_count / total  # Hallucination Rate
    score = 100 * (1 - hr)   

    print("========== Hallucination Evaluation ==========")
    print(f"Total samples            : {total}")
    print(f"Hallucinated samples     : {hall_count}")
    print(f"Hallucination Rate (HR)  : {hr:.4f}")
    print(f"Non-hallucination Score  : {score:.2f} / 100")
    print()

    print("---- Type occurrence over ALL samples ----")
    for t, c in type_counter_all.items():
        print(f"{t:18s}: {c:5d} ({c/total:.4f})")

    print()

    if hall_count > 0:
        print("---- Type occurrence among HALLUCINATED samples ----")
        for t, c in type_counter_hall.items():
            print(f"{t:18s}: {c:5d} ({c/hall_count:.4f})")
        print()
    else:
        print("No hallucinations detected; type breakdown is empty.")
        print()

    if combo_counter:
        print("---- Type combination distribution ----")
        for combo, c in combo_counter.most_common():
            print(f"{list(combo)} : {c}")
    print("=============================================")


    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("========== Lexical Commitment Analysis ==========")
    print(f"Event-level verbs freq      : {mean(event_freqs):.4f}")
    print(f"Definite commitments freq   : {mean(def_freqs):.4f}")
    print(f"Acoustic descriptors freq   : {mean(acoustic_freqs):.4f}")
    print()

    if event_freqs_hall:
        print("---- Among hallucinated samples ----")
        print(f"Event-level verbs freq      : {mean(event_freqs_hall):.4f}")
        print(f"Definite commitments freq   : {mean(def_freqs_hall):.4f}")
        print(f"Acoustic descriptors freq   : {mean(acoustic_freqs_hall):.4f}")


if __name__ == "__main__":
    evaluate_hallucination(EVAL_CSV)
