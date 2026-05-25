import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    write_json,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
import numpy as np
from .GNN_LoRA_finetune import warmup_gnn, finetune_gnn_lora, preprocess_graph


def chunking_by_token_size(content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_content = decode_tokens_by_tiktoken(tokens[start : start + max_token_size], model_name=tiktoken_model)
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(tokens[:llm_max_tokens], model_name=tiktoken_model_name)
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
    content: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    return dict(
        hyper_relation="<hyperedge>" + knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
        original_content=content,
    )


async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        already_weights.append(already_hyperedge["weight"])
        already_source_ids.extend(split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP]))

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(set([dp["source_id"] for dp in nodes_data] + already_source_ids))
    original_content = [dp["original_content"] for dp in nodes_data if "original_content" in dp]
    node_data = dict(role="hyperedge", weight=weight, source_id=source_id, original_content=original_content[0])
    await knowledge_graph_inst.upsert_node(
        hyperedge_name,
        node_data=node_data,
    )
    node_data["hyperedge_name"] = hyperedge_name
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP]))
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter([dp["entity_type"] for dp in nodes_data] + already_entity_types).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in nodes_data] + already_description)))
    source_id = GRAPH_FIELD_SEP.join(set([dp["source_id"] for dp in nodes_data] + already_source_ids))
    description = await _handle_entity_relation_summary(entity_name, description, global_config)
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []

    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]

        already_weights = []
        already_source_ids = []

        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP]))

        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(set([source_id] + already_source_ids))

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                role="edge",
                weight=weight,
                source_id=source_id,
            ),
        )

        edge_data.append(
            dict(
                src_id=hyper_relation,
                tgt_id=entity_name,
                weight=weight,
            )
        )

    return edge_data


async def extract_doc(
    chunks: dict[str, TextChunkSchema],
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(
            **context_base, input_text=content
        )

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation = ""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
            if_relation = await _handle_single_hyperrelation_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                maybe_edges[if_relation["hyper_relation"]].append(if_relation)
                now_hyper_relation = if_relation["hyper_relation"]

            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key, now_hyper_relation)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)

    return maybe_nodes, maybe_edges


async def update_db(
    maybe_nodes: dict,
    maybe_edges: dict,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
):

    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [_merge_hyperedges_then_upsert(k, v, knowledge_graph_inst, global_config) for k, v in maybe_edges.items()]
        ),
        total=len(maybe_edges),
        desc="Inserting hyperedges",
        unit="entity",
    ):
        all_hyperedges_data.append(await result)

    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [_merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config) for k, v in maybe_nodes.items()]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [_merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config) for k, v in maybe_nodes.items()]
        ),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning("Didn't extract any hyperedges and entities, maybe your LLM is not working")
        return None

    if not len(all_hyperedges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if hyperedge_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
            }
            for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def extract_entities_new(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:

    maybe_nodes, maybe_edges = await extract_doc(chunks, global_config)
    new_kg_inst = await update_db(maybe_nodes, maybe_edges, knowledge_graph_inst, entity_vdb, hyperedge_vdb)

    return new_kg_inst


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        max_retry = global_config.get("entity_extract_max_retries", 10)

        async def _try_extract_once() -> tuple[dict, dict]:
            hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
            final_result = await use_llm_func(hint_prompt)
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(continue_prompt, history_messages=history)
                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
                final_result += glean_result

                if now_glean_index == entity_extract_max_gleaning - 1:
                    break

                if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break

            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )

            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)
            now_hyper_relation = ""

            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])

                if_relation = await _handle_single_hyperrelation_extraction(record_attributes, chunk_key, content)
                if if_relation is not None:
                    maybe_edges[if_relation["hyper_relation"]].append(if_relation)
                    now_hyper_relation = if_relation["hyper_relation"]

                if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key, now_hyper_relation)
                if if_entities is not None:
                    maybe_nodes[if_entities["entity_name"]].append(if_entities)

            return dict(maybe_nodes), dict(maybe_edges)

        attempt = 0
        maybe_nodes, maybe_edges = {}, {}

        while attempt <= max_retry:
            maybe_nodes, maybe_edges = await _try_extract_once()
            if maybe_nodes or maybe_edges:
                break
            attempt += 1
            if attempt <= max_retry:
                logger.warning(f"Extraction failed for chunk: {chunk_key}. Retrying (attempt {attempt}/{max_retry})...")

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return maybe_nodes, maybe_edges

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)

    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [_merge_hyperedges_then_upsert(k, v, knowledge_graph_inst, global_config) for k, v in maybe_edges.items()]
        ),
        total=len(maybe_edges),
        desc="Inserting hyperedges",
        unit="entity",
    ):
        all_hyperedges_data.append(await result)

    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [_merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config) for k, v in maybe_nodes.items()]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [_merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config) for k, v in maybe_nodes.items()]
        ),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning("Didn't extract any hyperedges and entities, maybe your LLM is not working")
        return None

    if not len(all_hyperedges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if hyperedge_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
                "original_content": dp["original_content"],
            }
            for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def my_extract_entities(
    chunks: dict[str, TextChunkSchema],
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> list[str]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(
            **context_base, input_text=content
        )

        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation = ""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
            if_relation = await _handle_single_hyperrelation_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                maybe_edges[if_relation["hyper_relation"]].append(if_relation)
                now_hyper_relation = if_relation["hyper_relation"]

            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key, now_hyper_relation)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        logger.info(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)

    if hyperedge_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
            }
            for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    entity_vectors = {
        entity_name: {"vector": "dummy_vector"} for entity_name in maybe_nodes.keys()  # 這裡應該是計算向量的邏輯
    }
    return maybe_nodes, maybe_edges, entity_vectors


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    use_gnn_vdb: bool = False,
) -> dict:
    """
    改良版 kg_query：
    - 查詢邏輯完全相同
    - 若 use_gnn_vdb=True，會改用 gnn_entities_vdb / gnn_hyperedges_vdb
    """
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(hashing_kv, args_hash, query, query_param.mode)

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )
    hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(
        **context_base, input_text=query
    )

    final_result = await use_model_func(hint_prompt)
    logger.info("kw_prompt result:")
    logger.info(final_result)

    hl_keywords, ll_keywords = [], []
    try:
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
            if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                hl_keywords.append("<hyperedge>" + clean_str(record_attributes[1]))
            elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                ll_keywords.append(clean_str(record_attributes[1]).upper())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error: {e} {final_result}")

        return {"response": PROMPTS["fail_response"], "context": f"JSON parsing error: {e} {final_result}"}

    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords and high_level_keywords is empty"}

    if ll_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("low_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords is empty"}
    else:
        ll_keywords = ", ".join(ll_keywords)

    if hl_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "high_level_keywords is empty"}
    else:
        hl_keywords = ", ".join(hl_keywords)

    keywords = [ll_keywords, hl_keywords]

    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        hyperedges_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:

        return {"response": context, "context": None}
    if context is None:

        return {"response": PROMPTS["fail_response"], "context": "context is None"}

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(context_data=context, response_type=query_param.response_type)
    if query_param.only_need_prompt:

        return {"response": sys_prompt, "context": None}

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return {"response": response, "context": context}


# ablation: open source LLM
async def kg_query_local(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    use_gnn_vdb: bool = False,
) -> dict:
    """
    改良版 kg_query：
    - 查詢邏輯完全相同
    - 若 use_gnn_vdb=True，會改用 gnn_entities_vdb / gnn_hyperedges_vdb
    """
    use_model_func = global_config["llm_model_func"]
    local_use_model_func = global_config.get("local_llm_model_func", use_model_func)
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(hashing_kv, args_hash, query, query_param.mode)

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )
    hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(
        **context_base, input_text=query
    )

    final_result = await use_model_func(hint_prompt)
    logger.info("kw_prompt result:")
    logger.info(final_result)

    hl_keywords, ll_keywords = [], []
    try:
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
            if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                hl_keywords.append("<hyperedge>" + clean_str(record_attributes[1]))
            elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                ll_keywords.append(clean_str(record_attributes[1]).upper())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error: {e} {final_result}")

        return {"response": PROMPTS["fail_response"], "context": f"JSON parsing error: {e} {final_result}"}

    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords and high_level_keywords is empty"}

    if ll_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("low_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords is empty"}
    else:
        ll_keywords = ", ".join(ll_keywords)

    if hl_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "high_level_keywords is empty"}
    else:
        hl_keywords = ", ".join(hl_keywords)

    keywords = [ll_keywords, hl_keywords]

    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        hyperedges_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:

        return {"response": context, "context": None}
    if context is None:

        return {"response": PROMPTS["fail_response"], "context": "context is None"}

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(context_data=context, response_type=query_param.response_type)
    if query_param.only_need_prompt:

        return {"response": sys_prompt, "context": None}

    response = await local_use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return {"response": response, "context": context}


async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):

    ll_keywords, hl_keywords = query[0], query[1]
    if query_param.mode in ["local", "hybrid"]:
        if ll_keywords == "":
            ll_entities_context, ll_relations_context, ll_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn("Low Level context is None. Return empty Low entity/relationship/source")
            query_param.mode = "global"
        else:
            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
            ) = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            )
    if query_param.mode in ["global", "hybrid"]:
        if hl_keywords == "":
            hl_entities_context, hl_relations_context, hl_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn("High Level context is None. Return empty High entity/relationship/source")
            query_param.mode = "local"
        else:
            (
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
            ) = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                hyperedges_vdb,
                text_chunks_db,
                query_param,
            )

            if hl_entities_context == "" and hl_relations_context == "" and hl_text_units_context == "":
                logger.warning("No high level context found. Switching to local mode.")
                query_param.mode = "local"
    if query_param.mode == "hybrid":
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    elif query_param.mode == "local":
        entities_context, relations_context, text_units_context = (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        )
    return f"""
-----Entities-----
```csv
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
```
"""


async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):

    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r["entity_name"]) for r in results])
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_degrees = await asyncio.gather(*[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results])
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    use_relations = await _find_most_related_edges_from_entities(node_datas, query_param, knowledge_graph_inst)
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    entites_section_list = [["id", "entity", "type"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [["id", "hyperedge", "related_entities"]]
    for i, e in enumerate(use_relations):
        relations_section_list.append([i, e["description"], e["related_nodes"]])
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP]) for dp in node_datas]
    edges = await asyncio.gather(*[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas])
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(*[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes])

    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if e[1] in all_one_hop_text_units_lookup and c_id in all_one_hop_text_units_lookup[e[1]]:
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(all_text_units, key=lambda x: (x["order"], -x["relation_counts"]))

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(e)
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges])
    all_edges_degree = await asyncio.gather(*[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges])
    all_edges_data = [
        {"src_tgt": k, "rank": d, "description": k[1], **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True)
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    all_related_nodes = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["src_tgt"][1]) for edge in all_edges_data]
    )
    all_nodes = []
    for this_nodes in all_related_nodes:
        all_nodes.append("|".join([n[1] for n in this_nodes]))
    all_edges_data = [{**e, "related_nodes": n} for e, n in zip(all_edges_data, all_nodes)]
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):

    results = await hyperedges_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r["hyperedge_name"]) for r in results])

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    edge_datas = [
        {"hyperedge": k["hyperedge_name"], "rank": k["distance"], **v}
        for k, v in zip(results, edge_datas)
        if v is not None
    ]
    edge_datas = sorted(edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True)
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["hyperedge"],
        max_token_size=query_param.max_token_for_global_context,
    )
    all_related_nodes = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas]
    )
    all_nodes = []
    for this_nodes in all_related_nodes:
        all_nodes.append("|".join([n[1] for n in this_nodes]))
    edge_datas = [{**e, "related_nodes": n} for e, n in zip(edge_datas, all_nodes)]

    use_entities = await _find_most_related_entities_from_relationships(edge_datas, query_param, knowledge_graph_inst)
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    relations_section_list = [["id", "hyperedge", "related_entities"]]
    for i, e in enumerate(edge_datas):
        relations_section_list.append([i, e["hyperedge"], e["related_nodes"]])
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):

    node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas])

    entity_names = []
    seen = set()

    for node_data in node_datas:
        for e in node_data:
            if e[1] not in seen:
                entity_names.append(e[1])
                seen.add(e[1])

    node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names])

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [{**n, "entity_name": k, "rank": d} for k, n, d in zip(entity_names, node_datas, node_degrees)]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP]) for dp in edge_datas]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)

                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    valid_text_units = [t for t in all_text_units if t["data"] is not None and "content" in t["data"]]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):

    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]

    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    combined_relationships = process_combine_contexts(hl_relationships, ll_relationships)

    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources


async def store_gnn_embeddings(vdb: BaseVectorStorage, gnn_embeddings: dict[str, np.ndarray]) -> None:

    await vdb.store_gnn_embeddings(gnn_embeddings)
    logger.info(f"Stored GNN embeddings for {len(gnn_embeddings)} nodes.")


async def kg_query_mixed(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb_lm: BaseVectorStorage,
    hyperedges_vdb_lm: BaseVectorStorage,
    entities_vdb_gnn: BaseVectorStorage,
    hyperedges_vdb_gnn: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
) -> dict:

    use_model_func = global_config["llm_model_func"]
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])
    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )
    hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(
        **context_base, input_text=query
    )

    max_retries = global_config.get("kw_extract_max_retries", 10)
    hl_keywords, ll_keywords = [], []
    for attempt in range(1, max_retries + 1):
        final_result = await use_model_func(hint_prompt)
        logger.info(f"[KW Extract Attempt {attempt}/{max_retries}] kw_prompt result:")
        logger.info(final_result)

        try:
            hl_keywords, ll_keywords = [], []
            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
                if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                    hl_keywords.append("<hyperedge>" + clean_str(record_attributes[1]))
                elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                    ll_keywords.append(clean_str(record_attributes[1]).upper())

            if hl_keywords or ll_keywords:
                logger.info(f"Keyword extraction succeeded on attempt {attempt}.")
                break

        except Exception as e:
            logger.warning(f"[KW Extract Attempt {attempt}] parse error: {e}")
            if attempt == max_retries:
                return {"response": PROMPTS["fail_response"], "context": f"Keyword parse error: {e}"}

        if not (hl_keywords or ll_keywords) and attempt < max_retries:
            logger.warning(f"[KW Extract Attempt {attempt}] No keywords, retrying...")
            await asyncio.sleep(2 ** (attempt - 1))
        elif not (hl_keywords or ll_keywords):
            logger.error(f"[KW Extract Failed] After {max_retries} attempts, still no keywords.")
            return {"response": PROMPTS["fail_response"], "context": "Keyword extraction failed"}

    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords and high_level_keywords is empty"}

    if ll_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("low_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords is empty"}
    else:
        ll_keywords = ", ".join(ll_keywords)

    if hl_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "high_level_keywords is empty"}
    else:
        hl_keywords = ", ".join(hl_keywords)

    keywords = [ll_keywords, hl_keywords]

    logger.info("Querying LM-based VDB...")
    context_lm = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb_lm,
        hyperedges_vdb_lm,
        text_chunks_db,
        query_param,
    )

    logger.info("Querying GNN-based VDB...")
    context_gnn = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb_gnn,
        hyperedges_vdb_gnn,
        text_chunks_db,
        query_param,
    )

    merged_context = await _merge_contexts(context_lm, context_gnn)

    if query_param.only_need_context:
        return {"response": merged_context, "context": None}

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(context_data=merged_context, response_type=query_param.response_type)
    response = await use_model_func(query, system_prompt=sys_prompt, stream=query_param.stream)
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return {"response": response, "context": merged_context}


async def kg_query_mixed_local(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb_lm: BaseVectorStorage,
    hyperedges_vdb_lm: BaseVectorStorage,
    entities_vdb_gnn: BaseVectorStorage,
    hyperedges_vdb_gnn: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
) -> dict:

    use_model_func = global_config["llm_model_func"]
    local_use_model_func = global_config["local_llm_model_func"]

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])
    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )
    hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(
        **context_base, input_text=query
    )

    max_retries = global_config.get("kw_extract_max_retries", 10)
    hl_keywords, ll_keywords = [], []
    for attempt in range(1, max_retries + 1):
        final_result = await use_model_func(hint_prompt)
        logger.info(f"[KW Extract Attempt {attempt}/{max_retries}] kw_prompt result:")
        logger.info(final_result)

        try:
            hl_keywords, ll_keywords = [], []
            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
                if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                    hl_keywords.append("<hyperedge>" + clean_str(record_attributes[1]))
                elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                    ll_keywords.append(clean_str(record_attributes[1]).upper())

            if hl_keywords or ll_keywords:
                logger.info(f"Keyword extraction succeeded on attempt {attempt}.")
                break

        except Exception as e:
            logger.warning(f"[KW Extract Attempt {attempt}] parse error: {e}")
            if attempt == max_retries:
                return {"response": PROMPTS["fail_response"], "context": f"Keyword parse error: {e}"}

        if not (hl_keywords or ll_keywords) and attempt < max_retries:
            logger.warning(f"[KW Extract Attempt {attempt}] No keywords, retrying...")
            await asyncio.sleep(2 ** (attempt - 1))
        elif not (hl_keywords or ll_keywords):
            logger.error(f"[KW Extract Failed] After {max_retries} attempts, still no keywords.")
            return {"response": PROMPTS["fail_response"], "context": "Keyword extraction failed"}

    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords and high_level_keywords is empty"}

    if ll_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("low_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords is empty"}
    else:
        ll_keywords = ", ".join(ll_keywords)

    if hl_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "high_level_keywords is empty"}
    else:
        hl_keywords = ", ".join(hl_keywords)

    keywords = [ll_keywords, hl_keywords]

    logger.info("Querying LM-based VDB...")
    context_lm = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb_lm,
        hyperedges_vdb_lm,
        text_chunks_db,
        query_param,
    )

    logger.info("Querying GNN-based VDB...")
    context_gnn = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb_gnn,
        hyperedges_vdb_gnn,
        text_chunks_db,
        query_param,
    )

    merged_context = await _merge_contexts(context_lm, context_gnn)

    if query_param.only_need_context:
        return {"response": merged_context, "context": None}

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(context_data=merged_context, response_type=query_param.response_type)
    response = await local_use_model_func(query, system_prompt=sys_prompt, stream=query_param.stream)
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return {"response": response, "context": merged_context}


async def _merge_contexts(context_lm: str, context_gnn: str) -> str:
    """
    合併 LM 與 GNN 的 context，針對 Entities / Relationships / Sources 區段內容去重。
    """

    def extract_section(context, header):
        pattern = rf"-----{header}-----\n```csv\n(.*?)```"
        match = re.search(pattern, context, re.DOTALL)
        return match.group(1).strip() if match else ""

    lm_entities = extract_section(context_lm, "Entities").splitlines()
    gnn_entities = extract_section(context_gnn, "Entities").splitlines()
    lm_rels = extract_section(context_lm, "Relationships").splitlines()
    gnn_rels = extract_section(context_gnn, "Relationships").splitlines()
    lm_sources = extract_section(context_lm, "Sources").splitlines()
    gnn_sources = extract_section(context_gnn, "Sources").splitlines()

    def dedup_ignore_id(lines):
        seen, result = set(), []
        for line in lines:
            parts = line.split(",", maxsplit=1)
            content = parts[1].strip() if len(parts) > 1 else line
            if content not in seen:
                seen.add(content)
                result.append(content)
        return result

    merged_entities = dedup_ignore_id(lm_entities + gnn_entities)
    merged_entities = "\n".join(f"{i},{line}" for i, line in enumerate(merged_entities))

    merged_rels = dedup_ignore_id(lm_rels + gnn_rels)
    merged_rels = "\n".join(f"{i},{line}" for i, line in enumerate(merged_rels))

    merged_sources = dedup_ignore_id(lm_sources + gnn_sources)
    merged_sources = "\n".join(f"{i},{line}" for i, line in enumerate(merged_sources))

    return f"""
-----Entities-----
```csv
{merged_entities}
-----Relationships-----
```csv
{merged_rels}
-----Sources-----
```csv
{merged_sources}
"""


# Ablation KG Functions
async def _handle_single_relation_extraction_kg(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relation"':
        return None

    knowledge_fragment = clean_str(record_attributes[1])
    source_entity = clean_str(record_attributes[2]).upper()
    target_entity = clean_str(record_attributes[3]).upper()
    edge_source_id = chunk_key
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    return dict(
        relation=knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
        source_entity=source_entity,
        target_entity=target_entity,
    )


async def _handle_single_entity_extraction_kg(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"':
        return None

    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        source_id=entity_source_id,
    )


async def _merge_edges_then_upsert_kg(
    edge_name: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):

    source_entity = edges_data[0]["source_entity"]
    target_entity = edges_data[0]["target_entity"]
    logger.info(f"Merging edge '{edge_name}' between '{source_entity}' and '{target_entity}'")
    already_weights = []
    already_source_ids = []

    already_edge = await knowledge_graph_inst.get_edge(source_entity, target_entity)
    if already_edge is not None:
        already_weights.append(already_edge.get("weight", 0))
        already_source_ids.extend(split_string_by_multi_markers(already_edge.get("source_id", ""), [GRAPH_FIELD_SEP]))

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(set([dp["source_id"] for dp in edges_data] + already_source_ids))

    edge_data = dict(
        role="edge",
        weight=weight,
        source_id=source_id,
        source_entity=source_entity,
        target_entity=target_entity,
        relation=edge_name,
    )
    await knowledge_graph_inst.upsert_edge(
        source_entity,
        target_entity,
        edge_data=edge_data,
    )
    edge_data["edge_name"] = edge_name
    return edge_data


async def extract_entities_kg(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples_kg"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples_kg"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples_kg"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )

    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction_kg"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        max_retry = global_config.get("entity_extract_max_retries", 5)

        async def _try_extract_once() -> tuple[dict, dict]:
            hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
            final_result = await use_llm_func(hint_prompt)
            logger.info(f"Result: {final_result}")
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(continue_prompt, history_messages=history)
                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
                final_result += glean_result

                if now_glean_index == entity_extract_max_gleaning - 1:
                    break

                if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break

            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )

            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)

            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])

                if_relation = await _handle_single_relation_extraction_kg(record_attributes, chunk_key)
                if if_relation is not None:
                    maybe_edges[if_relation["relation"]].append(if_relation)

                if_entities = await _handle_single_entity_extraction_kg(record_attributes, chunk_key)
                if if_entities is not None:
                    maybe_nodes[if_entities["entity_name"]].append(if_entities)

            return dict(maybe_nodes), dict(maybe_edges)

        attempt = 0
        maybe_nodes, maybe_edges = {}, {}

        while attempt <= max_retry:
            maybe_nodes, maybe_edges = await _try_extract_once()
            if len(maybe_nodes) == 2 and len(maybe_edges) == 1:
                break
            attempt += 1
            if attempt <= max_retry:
                logger.warning(f"Extraction failed for chunk: {chunk_key}. Retrying (attempt {attempt}/{max_retry})...")

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return maybe_nodes, maybe_edges

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)

    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [_merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config) for k, v in maybe_nodes.items()]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relation into storage...")
    all_edges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [_merge_edges_then_upsert_kg(k, v, knowledge_graph_inst, global_config) for k, v in maybe_edges.items()]
        ),
        total=len(maybe_edges),
        desc="Inserting relations",
        unit="relation",
    ):
        all_edges_data.append(await result)

    if not len(all_edges_data) and not len(all_entities_data):
        logger.warning("Didn't extract any hyperedges and entities, maybe your LLM is not working")
        return None

    if not len(all_edges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def only_extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict,
) -> list[dict]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples_kg"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples_kg"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples_kg"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )

    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction_kg"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        max_retry = global_config.get("entity_extract_max_retries", 10)

        async def _try_extract_once() -> tuple[dict, dict]:
            hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
            final_result = await use_llm_func(hint_prompt)
            logger.info(f"Result: {final_result}")
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(continue_prompt, history_messages=history)
                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
                final_result += glean_result

                if now_glean_index == entity_extract_max_gleaning - 1:
                    break

                if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break

            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )

            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)

            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])

                if_relation = await _handle_single_relation_extraction_kg(record_attributes, chunk_key)
                if if_relation is not None:
                    maybe_edges[if_relation["relation"]].append(if_relation)

                if_entities = await _handle_single_entity_extraction_kg(record_attributes, chunk_key)
                if if_entities is not None:
                    maybe_nodes[if_entities["entity_name"]].append(if_entities)

            return dict(maybe_nodes), dict(maybe_edges)

        attempt = 0
        maybe_nodes, maybe_edges = {}, {}

        while attempt <= max_retry:
            maybe_nodes, maybe_edges = await _try_extract_once()

            if len(maybe_nodes) == 2 and len(maybe_edges) == 1:
                break
            attempt += 1
            if attempt <= max_retry:
                logger.warning(f"Extraction failed for chunk: {chunk_key}. Retrying (attempt {attempt}/{max_retry})...")

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return maybe_nodes, maybe_edges

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    return results


async def _find_most_related_edges_from_entities_kg(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_edges_data(dp["entity_name"]) for dp in node_datas]
    )

    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for _, _, data in this_edges:
            s = data.get("source_entity", "").strip('"')
            r = data.get("relation", "").strip('"')
            t = data.get("target_entity", "").strip('"')

            edge_key = (s, r, t)

            if edge_key not in seen:
                seen.add(edge_key)
                all_edges.append(edge_key)

    return all_edges


async def _get_node_data_kg(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):

    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r["entity_name"]) for r in results])
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_degrees = await asyncio.gather(*[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results])
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    use_relations = await _find_most_related_edges_from_entities_kg(node_datas, query_param, knowledge_graph_inst)
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    entites_section_list = [["id", "entity", "type"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [["id", "triplet"]]

    for i, e in enumerate(use_relations):
        relations_section_list.append([i, e])
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entities_context, relations_context, text_units_context


async def _build_query_context_kg(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):

    ll_keywords, hl_keywords = query[0], query[1]
    if query_param.mode in ["local", "hybrid"]:
        if ll_keywords == "":
            ll_entities_context, ll_relations_context, ll_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn("Low Level context is None. Return empty Low entity/relationship/source")
            query_param.mode = "global"
        else:
            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
            ) = await _get_node_data_kg(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            )
    if query_param.mode in ["global", "hybrid"]:
        if hl_keywords == "":
            hl_entities_context, hl_relations_context, hl_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn("High Level context is None. Return empty High entity/relationship/source")
            query_param.mode = "local"
        else:
            (
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
            ) = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                hyperedges_vdb,
                text_chunks_db,
                query_param,
            )

            if hl_entities_context == "" and hl_relations_context == "" and hl_text_units_context == "":
                logger.warning("No high level context found. Switching to local mode.")
                query_param.mode = "local"
    if query_param.mode == "hybrid":
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    elif query_param.mode == "local":
        entities_context, relations_context, text_units_context = (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        )
    return f"""
-----Entities-----
```csv
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
```
"""


async def kg_query_kg(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    use_gnn_vdb: bool = False,
) -> dict:
    """
    改良版 kg_query：
    - 查詢邏輯完全相同
    - 若 use_gnn_vdb=True，會改用 gnn_entities_vdb / gnn_hyperedges_vdb
    """
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(hashing_kv, args_hash, query, query_param.mode)

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )
    hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(
        **context_base, input_text=query
    )

    final_result = await use_model_func(hint_prompt)
    logger.info("kw_prompt result:")
    logger.info(final_result)

    hl_keywords, ll_keywords = [], []
    try:
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
            if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                hl_keywords.append("<hyperedge>" + clean_str(record_attributes[1]))
            elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                ll_keywords.append(clean_str(record_attributes[1]).upper())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error: {e} {final_result}")

        return {"response": PROMPTS["fail_response"], "context": f"JSON parsing error: {e} {final_result}"}

    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords and high_level_keywords is empty"}

    if ll_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("low_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords is empty"}
    else:
        ll_keywords = ", ".join(ll_keywords)

    if hl_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "high_level_keywords is empty"}
    else:
        hl_keywords = ", ".join(hl_keywords)

    keywords = [ll_keywords, hl_keywords]

    if use_gnn_vdb:
        logger.info("Using GNN vector database for retrieval.")
        entities_vdb = getattr(knowledge_graph_inst, "gnn_entities_vdb", entities_vdb)
        hyperedges_vdb = getattr(knowledge_graph_inst, "gnn_hyperedges_vdb", hyperedges_vdb)

    context = await _build_query_context_kg(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        hyperedges_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:

        return {"response": context, "context": None}
    if context is None:

        return {"response": PROMPTS["fail_response"], "context": "context is None"}

    sys_prompt_temp = PROMPTS["rag_response_kg"]
    sys_prompt = sys_prompt_temp.format(context_data=context, response_type=query_param.response_type)
    if query_param.only_need_prompt:

        return {"response": sys_prompt, "context": None}

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return {"response": response, "context": context}


async def kg_query_mixed_kg(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb_lm: BaseVectorStorage,
    hyperedges_vdb_lm: BaseVectorStorage,
    entities_vdb_gnn: BaseVectorStorage,
    hyperedges_vdb_gnn: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
) -> dict:

    use_model_func = global_config["llm_model_func"]
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])
    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )
    hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(
        **context_base, input_text=query
    )

    max_retries = global_config.get("kw_extract_max_retries", 10)
    hl_keywords, ll_keywords = [], []
    for attempt in range(1, max_retries + 1):
        final_result = await use_model_func(hint_prompt)
        logger.info(f"[KW Extract Attempt {attempt}/{max_retries}] kw_prompt result:")
        logger.info(final_result)

        try:
            hl_keywords, ll_keywords = [], []
            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
                if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                    hl_keywords.append("<hyperedge>" + clean_str(record_attributes[1]))
                elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                    ll_keywords.append(clean_str(record_attributes[1]).upper())

            if hl_keywords or ll_keywords:
                logger.info(f"Keyword extraction succeeded on attempt {attempt}.")
                break

        except Exception as e:
            logger.warning(f"[KW Extract Attempt {attempt}] parse error: {e}")
            if attempt == max_retries:
                return {"response": PROMPTS["fail_response"], "context": f"Keyword parse error: {e}"}

        if not (hl_keywords or ll_keywords) and attempt < max_retries:
            logger.warning(f"[KW Extract Attempt {attempt}] No keywords, retrying...")
            await asyncio.sleep(2 ** (attempt - 1))
        elif not (hl_keywords or ll_keywords):
            logger.error(f"[KW Extract Failed] After {max_retries} attempts, still no keywords.")
            return {"response": PROMPTS["fail_response"], "context": "Keyword extraction failed"}

    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords and high_level_keywords is empty"}

    if ll_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("low_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "low_level_keywords is empty"}
    else:
        ll_keywords = ", ".join(ll_keywords)

    if hl_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("high_level_keywords is empty")

        return {"response": PROMPTS["fail_response"], "context": "high_level_keywords is empty"}
    else:
        hl_keywords = ", ".join(hl_keywords)

    keywords = [ll_keywords, hl_keywords]

    logger.info("Querying LM-based VDB...")
    context_lm = await _build_query_context_kg(
        keywords,
        knowledge_graph_inst,
        entities_vdb_lm,
        hyperedges_vdb_lm,
        text_chunks_db,
        query_param,
    )

    logger.info("Querying GNN-based VDB...")
    context_gnn = await _build_query_context_kg(
        keywords,
        knowledge_graph_inst,
        entities_vdb_gnn,
        hyperedges_vdb_gnn,
        text_chunks_db,
        query_param,
    )

    merged_context = await _merge_contexts(context_lm, context_gnn)

    if query_param.only_need_context:
        return {"response": merged_context, "context": None}

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(context_data=merged_context, response_type=query_param.response_type)
    response = await use_model_func(query, system_prompt=sys_prompt, stream=query_param.stream)
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return {"response": response, "context": merged_context}
