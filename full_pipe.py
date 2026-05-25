import os
import json
import argparse
import asyncio
import datetime
from dotenv import load_dotenv
from hypergraphrag.hypergraphrag import HyperGraphRAG
from hypergraphrag.base import QueryParam
from tqdm.asyncio import tqdm_asyncio
from GNN_pretrain import pretrain_base_gnn
from hypergraphrag.utils import set_logger
import networkx as nx
import torch
import logging
import time
from openai import OpenAI
import re
import random
import gc
import ctypes
import psutil

load_dotenv()

local_client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:11452/v1")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")


def free_memory(tag=""):
    gc.collect()

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as e:
        print(f"malloc_trim failed: {e}")

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024


def format_range(start, end):
    return f"{str(start).zfill(3)}-{str(end).zfill(3)}"


def make_exp_dir(base_dir, batch_size, batch_range, query_mode):
    parent_dir = os.path.join(base_dir, f"{batch_size}_edited_{query_mode}")
    os.makedirs(parent_dir, exist_ok=True)
    exp_dir = os.path.join(parent_dir, f"edited_{batch_range}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, parent_dir


async def extract_old_sentences(batch_edits):
    old_sentences = []
    seen = set()
    for case in batch_edits:
        for edit in case["knowledge_edits"]:
            old = edit["old"].strip()
            if old not in seen:
                seen.add(old)
                old_sentences.append(old)

    return old_sentences


def pretrain_gnn(G, config, rag):
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    model = pretrain_base_gnn(G, config, device, rag.entities_vdb, rag.hyperedges_vdb)
    return model


async def apply_edits(rag, edits):
    await rag.init_new_gnn_vdbs()
    try:
        await rag.build_simhash_faiss_index(f_bits=128, top_k=1200)
    except Exception as e:
        logging.error(f"Failed to build SimHash index: {e}")
        rag.simhash_failed = True
        return

    await rag.edit_hypergraph(edits)
    await rag._insert_done()
    print(f"Applied {len(edits)} edits.")


def run_llm_divide2(query):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        }
    ]
    message_prompt = {"role": "user", "content": query}
    messages.append(message_prompt)
    f = 0

    while f == 0:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                frequency_penalty=0,
                presence_penalty=0,
            )
            result = response.choices[0].message.content
            f = 1
        except Exception as e:
            print(f"openai error, retry {e}")
            time.sleep(10)

    return result


with open("./prompts/divide.txt", "r") as f:
    divide_prompt = f.read()


# Main pipeline
async def query_all(rag, all_questions, case_ids, query_mode="hybrid", max_concurrency=12):

    cases = [c for c in all_questions if c.get("case_id") in case_ids]

    semaphore = asyncio.Semaphore(max_concurrency)

    if query_mode == "hybrid":
        result_name = "mix_results"
    elif query_mode == "gnn":
        result_name = "gnn_results"
    else:
        result_name = "lm_results"

    # ---- SINGLE SUB-QUESTION QUERY ----
    async def query_one(qobj, case_id):
        # query by open source LLM
        # result = await rag.aquery_with_projector_local(qobj["question"], QueryParam(mode="hybrid"), mode=query_mode)

        result = await rag.aquery_my(qobj["question"], QueryParam(mode="hybrid"), mode=query_mode)

        return {
            "case_id": case_id,
            "question": qobj["question"],
            "answer": qobj.get("answer", ""),
            "answer_alias": qobj.get("answer_alias", []),
            "result": result,
        }

    async def process_case(case):
        async with semaphore:
            case_id = case["case_id"]
            new_case = {"case_id": case_id, "multi_hop_questions": []}

            mhq_list = case.get("multi_hop_questions", [])
            gold_single = case.get("single_hop_questions", [])

            case_results = []

            for mhq in mhq_list:
                prompt = divide_prompt.replace("<<<<QUESTION>>>>", mhq["question"])
                output = run_llm_divide2(prompt)

                sub_questions = output.split("\n")
                single_results = []
                entity = ""
                path = []

                for idx, sub_q in enumerate(sub_questions):

                    gold = gold_single[idx] if idx < len(gold_single) else {}

                    sub_q = sub_q.replace("[ENT]", entity)

                    qobj = {
                        "question": sub_q,
                        "answer": gold.get("answer", ""),
                        "answer_alias": gold.get("answer_alias", []),
                    }

                    result = await query_one(qobj, case_id)
                    single_results.append(result)

                    match = re.search(r"Answer:(.*)", result["result"][result_name])
                    entity = match.group(1).strip() if match else ""

                    path.append(entity)

                case_results.append(
                    {
                        "multi_hop_question": mhq,
                        "single_hop_questions": single_results,
                        "number_of_hops": len(gold_single),
                        "path": path,
                        "final_answer": path[-1] if path else "",
                    }
                )

            new_case["multi_hop_questions"] = case_results
            return new_case

    tasks = [asyncio.create_task(process_case(case)) for case in cases]

    final_output = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing Cases"):
        result = await coro
        final_output.append(result)
    final_output.sort(key=lambda x: x["case_id"])
    return final_output


def set_logger_my(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    handler = logging.FileHandler(log_file_path, mode="w")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def record_timing(timing_dict, phase_name, t_start, t_end, summary_dict=None):
    duration = round(t_end - t_start, 3)
    timing_dict[phase_name] = {"seconds": duration}

    if summary_dict is not None:
        if phase_name not in summary_dict:
            summary_dict[phase_name] = {"seconds": 0.0}
        summary_dict[phase_name]["seconds"] += duration


async def main(args):

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    timing_log_path = os.path.join(
        args.output_base_dir, f"{args.batch_size}_edited_{args.query_mode}", "timing_summary.json"
    )
    if os.path.exists(timing_log_path):
        try:
            with open(timing_log_path, "r") as f:
                timing_log = json.load(f)
        except Exception as e:
            print(f"Failed to load previous timing_summary: {e}")
            timing_log = {"batches": [], "summary": {}}
    else:
        timing_log = {"batches": [], "summary": {}}

    with open(args.knowledge_file, "r") as f:
        all_edits = json.load(f)
    with open(args.question_file, "r") as f:
        all_questions = json.load(f)

    edit_map = {item["case_id"]: item for item in all_edits}
    question_map = {item["case_id"]: item for item in all_questions}

    pairs = []
    for cid in edit_map:
        pairs.append((edit_map[cid], question_map[cid]))

    random.shuffle(pairs)

    all_edits, all_questions = zip(*pairs)

    total_cases = len(all_edits)

    if args.batch_size == "All":
        batch_size = total_cases
    else:
        batch_size = int(args.batch_size)

    for batch_start in range(args.start_index, total_cases + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, total_cases)
        batch_range_str = format_range(batch_start, batch_end)
        batch_edits = all_edits[batch_start - 1 : batch_end]
        batch_case_ids = [c["case_id"] for c in batch_edits]

        exp_dir, parent_result_dir = make_exp_dir(args.output_base_dir, batch_size, batch_range_str, args.query_mode)

        result_path = os.path.join(exp_dir, "results.json")
        if os.path.exists(result_path):
            print(f"Skipping batch {batch_range_str} (already completed)")
            continue
        timing_log["batches"] = [b for b in timing_log["batches"] if b["batch_range"] != batch_range_str]
        print(f"\nRunning batch {batch_range_str} → {exp_dir}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(exp_dir, f"{batch_range_str}_{timestamp}.log")
        set_logger_my(log_file_path)

        rag = HyperGraphRAG(working_dir=os.path.join(exp_dir, "data_base"))

        if args.pretrain:
            config = {
                "input_dim": 1536,
                "output_dim": args.GNN_dimension,
                "lora_rank": 32,
                "gnn_type": "GAT",
                "warmup_steps": 1,
                "gnn_layer_num": 2,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "num_epochs": 200,
                "print_every": 50,
                "save_dir": exp_dir,
                "parent_result_dir": parent_result_dir,
                "pretrain_gnn_base_name": "pretrain_gnn_base.pth",
                "save_name": "pretrain_gnn_lora.pth",
                "device": args.device,
            }
            rag.fine_tune_config = config
        rag.fine_tune_config["save_dir"] = exp_dir

        batch_timing = {"batch_range": batch_range_str, "timings": {}}
        timing_log["batches"].append(batch_timing)
        t0 = time.time()
        # Step 1: Construct hypergraph with old knowledge
        t_start = time.time()
        old_sentences = await extract_old_sentences(batch_edits)
        await rag.ainsert(old_sentences)
        t_end = time.time()
        record_timing(batch_timing["timings"], "insert_old", t_start, t_end, timing_log["summary"])

        with open(os.path.join(parent_result_dir, "timing_summary.json"), "w") as f:
            json.dump(timing_log, f, indent=2)

        # Step 2: Save graph before edits
        G_before = rag.chunk_entity_relation_graph._graph.copy()
        graph_before_path = os.path.join(exp_dir, "graph_before_edit.graphml")
        nx.write_graphml(G_before, graph_before_path)

        # Step 3: Pretrain GNN base model
        if args.pretrain:
            t_start = time.time()
            await asyncio.to_thread(pretrain_gnn, G_before, config, rag)
            t_end = time.time()
            record_timing(batch_timing["timings"], "pretrain_gnn", t_start, t_end, timing_log["summary"])

            with open(os.path.join(parent_result_dir, "timing_summary.json"), "w") as f:
                json.dump(timing_log, f, indent=2)

        # Step 4: Apply edits to hypergraph
        t_start = time.time()
        await apply_edits(rag, batch_edits)
        t_end = time.time()
        record_timing(batch_timing["timings"], "apply_edits", t_start, t_end, timing_log["summary"])

        with open(os.path.join(parent_result_dir, "timing_summary.json"), "w") as f:
            json.dump(timing_log, f, indent=2)

        # Step 5: Query the questions
        t_start = time.time()
        if getattr(rag, "simhash_failed", False):
            print("SimHash failed. Generating fallback results...")
            results = []

            for case in all_questions:
                if case["case_id"] not in batch_case_ids:
                    continue

                # fallback result
                fallback_result = {
                    "case_id": case["case_id"],
                    "multi_hop_questions": [
                        {
                            "multi_hop_question": mhq,
                            "single_hop_questions": [
                                {
                                    "case_id": case["case_id"],
                                    "question": sub["question"],
                                    "answer": sub.get("answer", ""),
                                    "answer_alias": sub.get("answer_alias", []),
                                    "result": {
                                        "mix_results": "SimHash build failed — skipping query.",
                                        "gnn_results": "SimHash build failed — skipping query.",
                                        "lm_results": "SimHash build failed — skipping query.",
                                    },
                                }
                                for sub in case.get("single_hop_questions", [])
                            ],
                            "number_of_hops": len(case.get("single_hop_questions", [])),
                            "path": ["N/A"],
                            "final_answer": "N/A",
                        }
                        for mhq in case.get("multi_hop_questions", [])
                    ],
                }
                results.append(fallback_result)

        else:
            results = await query_all(rag, all_questions, batch_case_ids, args.query_mode)

        t_end = time.time()
        record_timing(batch_timing["timings"], "RAG", t_start, t_end, timing_log["summary"])

        if "query_latency" not in timing_log["summary"]:
            timing_log["summary"]["query_latency"] = {"seconds": 0.0}

        timing_log["summary"]["query_latency"]["seconds"] += rag.query_latency

        batch_timing["timings"]["query_latency"] = {"seconds": rag.query_latency}

        with open(os.path.join(parent_result_dir, "timing_summary.json"), "w") as f:
            json.dump(timing_log, f, indent=2)

        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump({"batch_range": batch_range_str, "args": vars(args)}, f, indent=2)

        del G_before
        del rag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_time = time.time() - t0
        batch_timing["timings"]["total"] = {"seconds": round(total_time, 3)}
        summary = {}

        for b in timing_log["batches"]:
            for stage, timing in b["timings"].items():
                if "seconds" in timing:
                    if stage not in summary:
                        summary[stage] = {"seconds": 0.0}
                    summary[stage]["seconds"] += timing["seconds"]

        timing_log["summary"] = summary
        with open(os.path.join(parent_result_dir, "timing_summary.json"), "w") as f:
            json.dump(timing_log, f, indent=2)

        free_memory(tag=batch_range_str)

    all_results = []
    for subdir in sorted(os.listdir(parent_result_dir)):
        result_path = os.path.join(parent_result_dir, subdir, "results.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                try:
                    batch_result = json.load(f)
                    all_results.extend(batch_result)
                except Exception as e:
                    print(f"Failed to load {result_path}: {e}")

    with open(os.path.join(parent_result_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll batch results saved to {parent_result_dir}/all_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_file", required=True)
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--output_base_dir", default="experiments")
    parser.add_argument("--batch_size", default=100, help="integer or 'All'")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--query_mode", type=str, default="hybrid")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--GNN_dimension", type=int, default=256)

    args = parser.parse_args()
    asyncio.run(main(args))
