import os
import json
import spacy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Configuration
MAX_WORKERS = 64
DATASETS_NAME = ["MQuAKE-T.json", "MQuAKE-CF-3k-v2.json"]

_nlp = None


def get_spacy():
    global _nlp
    if _nlp is None:
        print(f"[Process {os.getpid()}] loading spaCy...")
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


# Entity extraction
def extract_entity_candidates(text: str) -> list[str]:
    nlp = get_spacy()
    doc = nlp(text)
    merged_tokens, buffer = [], []

    for token in doc:

        if token.i > 0 and token.i < len(doc) - 1:
            if doc[token.i - 1].text == "-" and doc[token.i + 1].text == "-":
                buffer.append(token.text)
                continue
        if token.dep_ in ("det", "pcomp", "amod", "prep", "advmod"):
            buffer.append(token.text)
        elif token.pos_ in ("NOUN", "PROPN", "PRON", "ADJ", "NUM"):
            buffer.append(token.text)
        elif token.text.lower() in ("de", "re"):
            buffer.append(token.text)
        elif token.text.lower() in ("of", "and", "'s", "'") and buffer:
            buffer.append(token.text)
        elif token.text in {"-", ","} and buffer:
            buffer.append(token.text)
        else:
            if buffer:
                merged_tokens.append(" ".join(buffer))
                buffer = []
            if token.pos_ != "PUNCT":
                merged_tokens.append(token.text)

    if buffer:
        merged_tokens.append(" ".join(buffer))

    return merged_tokens


# Subject check
def is_diff_word_subject(diff_phrase: str, sentence: str) -> bool:
    spans = extract_entity_candidates(sentence)

    diff_phrase_norm = diff_phrase.strip().lower()

    for idx, span in enumerate(spans):
        if diff_phrase_norm in span.lower():
            if idx >= len(spans) // 2:
                return False
            else:
                return True

    return True


# Edit classification
def classify_edit_case(old: str, new: str, old_entities=None):

    if old_entities is None:
        old_entities = extract_entity_candidates(old)

    new_entities = extract_entity_candidates(new)

    diff_old = [x for x in old_entities if x not in new_entities]
    diff_new = [x for x in new_entities if x not in old_entities]

    if old == new:
        return "Same", old_entities, new_entities

    if len(diff_old) == 1 and len(diff_new) == 1:
        if not is_diff_word_subject(diff_old[0], old) and not is_diff_word_subject(diff_new[0], new):
            return "Replace", old_entities, new_entities

    if len(diff_old) == 2 and len(diff_new) == 2:
        return "Add", old_entities, new_entities

    return "Skip", old_entities, new_entities


# Case processing
def process_case(case, old_knowledge_set, precomputed_old):
    case_id = case["case_id"]

    new_list = []
    for hop in case.get("new_single_hops", []):
        cloze = hop.get("cloze", "").strip()
        answer = hop.get("answer", "").strip()
        if cloze and answer:
            new_list.append(f"{cloze} {answer}.")

    knowledge_edits = []

    for new_sent in new_list:

        # Same case
        if new_sent in old_knowledge_set:
            ents = extract_entity_candidates(new_sent)
            knowledge_edits.append(
                {
                    "old": new_sent,
                    "old_entities": ents,
                    "new": new_sent,
                    "new_entities": ents,
                    "category": "Same",
                }
            )
            continue

        # Replace case
        replaced = False

        for old_sent in old_knowledge_set:
            old_entities = precomputed_old[old_sent]["entities"]
            category, _, new_entities = classify_edit_case(old_sent, new_sent, old_entities=old_entities)

            if category in ["Replace"]:
                knowledge_edits.append(
                    {
                        "old": old_sent,
                        "old_entities": old_entities,
                        "new": new_sent,
                        "new_entities": new_entities,
                        "category": category,
                    }
                )
                replaced = True
                break

        if replaced:
            continue

        # Add case
        added = False
        for old_sent in old_knowledge_set:
            old_entities = precomputed_old[old_sent]["entities"]
            category, _, new_entities = classify_edit_case(old_sent, new_sent, old_entities=old_entities)
            if category == "Add":
                knowledge_edits.append(
                    {
                        "old": old_sent,
                        "old_entities": old_entities,
                        "new": new_sent,
                        "new_entities": new_entities,
                        "category": category,
                    }
                )
                added = True
                break
        if added:
            continue

        # Skip case
        new_entities = extract_entity_candidates(new_sent)
        knowledge_edits.append(
            {
                "old": "",
                "old_entities": [],
                "new": new_sent,
                "new_entities": new_entities,
                "category": "Skip",
            }
        )

    # Save questions
    single_hop_q = [
        {"question": q["question"], "answer": q["answer"], "answer_alias": q.get("answer_alias", [])}
        for q in case.get("new_single_hops", [])
    ]
    multi_hop_q = [
        {"question": q, "answer": case["new_answer"], "answer_alias": case.get("new_answer_alias", [])}
        for q in case.get("questions", [])
    ]

    return {
        "case_id": case_id,
        "knowledge_edits": knowledge_edits,
        "single_hop_questions": single_hop_q,
        "multi_hop_questions": multi_hop_q,
    }


def wrapped_process_case(args):
    idx, case, old_knowledge_set, precomputed_old = args
    result = process_case(case, old_knowledge_set, precomputed_old)
    return idx, result


def process_dataset(filename):

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build old knowledge set
    dataset_name = os.path.splitext(filename)[0]
    old_list = []
    for case in tqdm(data, desc=f"Scanning old knowledge ({dataset_name})", ncols=100):
        for hop in case.get("single_hops", []):
            cloze = hop.get("cloze", "").strip()
            answer = hop.get("answer", "").strip()
            if cloze and answer:
                old_list.append(f"{cloze} {answer}.")

    old_knowledge_set = list(set(old_list))

    # Precompute all old sentence entities
    precomputed_old = {}
    for sent in tqdm(old_knowledge_set, desc="Precompute old", ncols=100):
        ents = extract_entity_candidates(sent)
        precomputed_old[sent] = {
            "entities": ents,
        }

    merged_entries = [None] * len(data)
    question_entries = [None] * len(data)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        task_args = [(idx, case, old_knowledge_set, precomputed_old) for idx, case in enumerate(data)]

        results = list(
            tqdm(
                executor.map(wrapped_process_case, task_args),
                total=len(data),
                desc=f"Processing {dataset_name}",
                ncols=100,
            )
        )

        for idx, result in results:
            merged_entries[idx] = {
                "case_id": result["case_id"],
                "knowledge_edits": result["knowledge_edits"],
            }
            question_entries[idx] = {
                "case_id": result["case_id"],
                "single_hop_questions": result["single_hop_questions"],
                "multi_hop_questions": result["multi_hop_questions"],
            }

    # Write output files
    merged_file = os.path.join(output_merged_folder, f"{dataset_name}_knowledge.json")
    with open(merged_file, "w", encoding="utf-8") as f:
        json.dump(merged_entries, f, indent=2, ensure_ascii=False)

    question_file = os.path.join(output_question_folder, f"{dataset_name}_questions.json")
    with open(question_file, "w", encoding="utf-8") as f:
        json.dump(question_entries, f, indent=2, ensure_ascii=False)

    print(f"\nFinished: {filename}")


if __name__ == "__main__":

    spacy.load("en_core_web_lg")

    output_merged_folder = "./knowledge"
    output_question_folder = "./questions"

    os.makedirs(output_merged_folder, exist_ok=True)
    os.makedirs(output_question_folder, exist_ok=True)

    for filename in tqdm(DATASETS_NAME, desc="Datasets", ncols=100):
        process_dataset(filename)
