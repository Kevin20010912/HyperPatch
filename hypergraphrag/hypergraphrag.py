import asyncio
import os
import difflib
import networkx as nx
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, List
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from types import SimpleNamespace
from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
    openai2GNN_embedding,
    set_global_projector,
    Qwen3_8B_complete,
    hf_model_complete,
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    extract_entities_kg,
    kg_query,
    extract_entities_new,
    kg_query_mixed,
    only_extract_entities,
    kg_query_kg,
    kg_query_mixed_kg,
    kg_query_local,
    kg_query_mixed_local,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from torch import nn
import numpy as np
from .GNN_LoRA_finetune import warmup_gnn, finetune_gnn_lora, preprocess_graph
from .GNN_model import GNN, GNNLoRA, SentenceProjector
import torch
import spacy
import faiss
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from simhash import Simhash
import json
from tqdm import tqdm
import spacy
from collections import Counter
import copy
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import re
import time


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name, package=package)

        # Get the class from the module and instantiate it
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class HyperGraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"hypergraphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 0  # try to prevent too many entities
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    gnn_embedding_func: EmbeddingFunc = field(default_factory=lambda: openai2GNN_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 8

    # LLM
    # ablation: open source LLM
    # llm_model_func: callable = hf_model_complete
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    local_llm_model_func: callable = Qwen3_8B_complete

    llm_model_name: str = "Qwen/Qwen2-7B"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 8
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):

        log_file = os.path.join("hypergraphrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"HyperGraphRAG init with param:\n  {_print_config}\n")

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = self._get_storage_class()[self.kv_storage]
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[self.vector_storage]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[self.graph_storage]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)
        self.gnn_embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.gnn_embedding_func)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.hyperedges_vdb = self.vector_db_storage_cls(
            namespace="hyperedges",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"hyperedge_name"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.gnn_entities_vdb: Type[NanoVectorDBStorage]
        self.gnn_hyperedges_vdb: Type[NanoVectorDBStorage]

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

        # ablation: open source LLM
        self.local_llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.local_llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

        self.nlp = spacy.load("en_core_web_lg")
        self.query_latency = 0.0

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()} for c in string_or_strings}
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(new_docs.items(), desc="Chunking documents", unit="doc"):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                hyperedge_vdb=self.hyperedges_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new hyperedges and entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping.")

                # Prepare node data
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(entity_name, node_data=node_data)
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (await self.chunk_entity_relation_graph.has_node(need_insert_id)):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.hyperedges_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"] + dp["src_id"] + dp["tgt_id"] + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.hyperedges_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode in ["hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.hyperedges_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,
            )
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.hyperedges_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(f"Entity '{entity_name}' and its relationships have been deleted.")
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def clean_orphan_gnn_vectors(self):
        for base_vdb, gnn_vdb, role in [
            (self.entities_vdb, self.gnn_entities_vdb, "entity"),
            (self.hyperedges_vdb, self.gnn_hyperedges_vdb, "hyperedge"),
        ]:
            try:
                base_ids = {entry["__id__"] for entry in base_vdb._client._NanoVectorDB__storage["data"]}
                gnn_storage = gnn_vdb._client._NanoVectorDB__storage
                gnn_ids = [entry["__id__"] for entry in gnn_storage["data"]]

                keep_indices = []
                new_data = []
                for i, entry in enumerate(gnn_storage["data"]):
                    if entry["__id__"] in base_ids:
                        keep_indices.append(i)
                        new_data.append(entry)

                old_len = len(gnn_storage["data"])
                gnn_storage["data"] = new_data
                gnn_storage["matrix"] = (
                    [gnn_storage["matrix"][i] for i in keep_indices]
                    if isinstance(gnn_storage["matrix"], list)
                    else np.array([gnn_storage["matrix"][i] for i in keep_indices])
                )
                gnn_vdb._client.save()
                logger.info(f"[Cleanup] Removed {old_len - len(new_data)} stale {role} embeddings from GNN VDB")
            except Exception as e:
                logger.error(f"Error during GNN VDB cleanup for {role}: {e}")

    def generate_gnn_embeddings(self) -> dict[str, np.ndarray]:

        gpu_id = self.fine_tune_config["device"]
        if torch.cuda.is_available() and torch.cuda.device_count() > int(gpu_id):
            device = torch.device(f"cuda:{gpu_id}")
        else:
            logger.warning("CUDA unavailable or GPU ID out of range. Using CPU.")
            device = torch.device("cpu")

        gnn_base_load_path = os.path.join(
            self.fine_tune_config["save_dir"], self.fine_tune_config["pretrain_gnn_base_name"]
        )
        gnn_lora_save_path = os.path.join(self.fine_tune_config["save_dir"], self.fine_tune_config["save_name"])

        if hasattr(self, "gnn_lora_model") and self.gnn_lora_model is not None:

            logger.info("Reusing existing in-memory GNN LoRA model (no reload).")

        elif os.path.exists(gnn_lora_save_path):

            logger.info(f"Resuming from fine-tuned LoRA checkpoint: {gnn_lora_save_path}")
            self.gnn_lora_model = warmup_gnn(
                self.chunk_entity_relation_graph._graph,
                self.fine_tune_config,
                device,
                self.entities_vdb,
                self.hyperedges_vdb,
                gnn_lora_save_path,
                mode="incremental",
            )

        else:

            logger.info(f"Initializing LoRA model from base GNN: {gnn_base_load_path}")
            self.gnn_lora_model = warmup_gnn(
                self.chunk_entity_relation_graph._graph,
                self.fine_tune_config,
                device,
                self.entities_vdb,
                self.hyperedges_vdb,
                gnn_base_load_path,
                mode="initial",
            )

        self.gnn_lora_model.to(device)

        G = self.chunk_entity_relation_graph._graph
        logger.info(f"Using full graph for GNN embedding propagation ({len(G.nodes)} nodes).")

        embeds, relations = preprocess_graph(G, device, self.entities_vdb, self.hyperedges_vdb)

        logger.info(f"Fine-tuning GNN LoRA...")
        self.gnn_lora_model = finetune_gnn_lora(
            G,
            self.fine_tune_config,
            self.gnn_lora_model,
            device,
            self.entities_vdb,
            self.hyperedges_vdb,
        )

        self.gnn_lora_model.eval()
        with torch.no_grad():
            node_embeddings, _, _ = self.gnn_lora_model(embeds, relations)

        node_embeddings_dict = {}
        for i, node in enumerate(G.nodes):
            emb = node_embeddings[i].cpu().numpy()
            node_embeddings_dict[node] = emb

        torch.save(self.gnn_lora_model.state_dict(), gnn_lora_save_path)
        logger.info(f"GNN embeddings fine-tuned & updated for {len(G.nodes)} nodes.")
        logger.info(f"Model weights saved to: {gnn_lora_save_path}")
        return node_embeddings_dict

    async def clone_vdb(
        self, vdb_instance: NanoVectorDBStorage, name: str, meta_fields: set[str]
    ) -> NanoVectorDBStorage:

        try:
            new_namespace = f"gnn_{name}"
            new_storage_file = os.path.join(self.working_dir, f"vdb_{new_namespace}.json")

            new_vdb = NanoVectorDBStorage(
                namespace=new_namespace,
                global_config={"working_dir": self.working_dir, "embedding_batch_num": self.embedding_batch_num},
                embedding_func=self.embedding_func,
                meta_fields=meta_fields,
            )

            if os.path.exists(new_storage_file):
                logger.info(f"Existing VDB file found: {new_storage_file}")
                old_data = new_vdb._client._NanoVectorDB__storage.get("data", [])
                old_count = len(old_data)
                new_count = len(vdb_instance._client._NanoVectorDB__storage.get("data", []))

                if new_count != old_count:
                    logger.info(f"Updating existing GNN VDB ({old_count} → {new_count} records).")
                    new_vdb._client._NanoVectorDB__storage = copy.deepcopy(vdb_instance._client._NanoVectorDB__storage)
                    new_vdb._client.save()
                else:
                    logger.info("Existing GNN VDB already up-to-date, skipping overwrite.")
            else:

                new_vdb._client._NanoVectorDB__storage = copy.deepcopy(vdb_instance._client._NanoVectorDB__storage)
                new_vdb._client.save()
                logger.info(f"Cloned new GNN VDB to {new_storage_file}")

            return new_vdb

        except Exception as e:
            logger.error(f"Error while cloning VDB: {e}")
            raise

    async def clone_or_pass_gnn_vdb(self):
        self.gnn_entities_vdb = await self.clone_vdb(self.entities_vdb, "entities", {"entity_name"})
        self.gnn_hyperedges_vdb = await self.clone_vdb(self.hyperedges_vdb, "hyperedges", {"hyperedge_name"})
        print(self.entities_vdb._client._NanoVectorDB__storage is self.gnn_entities_vdb._client._NanoVectorDB__storage)

    async def init_gnn_vdb(
        self,
        name: str,
        meta_fields: set[str],
        dim: int,
    ) -> NanoVectorDBStorage:
        try:
            new_namespace = f"gnn_{name}"
            new_storage_file = os.path.join(self.working_dir, f"vdb_{new_namespace}.json")

            embedding_func = self.gnn_embedding_func
            embedding_func.embedding_dim = self.fine_tune_config["output_dim"]
            if os.path.exists(new_storage_file):
                logger.info(f"Existing GNN VDB found: {new_storage_file}")
                vdb = NanoVectorDBStorage(
                    namespace=new_namespace,
                    global_config={"working_dir": self.working_dir, "embedding_batch_num": self.embedding_batch_num},
                    embedding_func=embedding_func,
                    meta_fields=meta_fields,
                )

                existing_dim = vdb._client._NanoVectorDB__storage.get("embedding_dim", None)
                if existing_dim == dim:
                    logger.info(f"Loaded existing GNN VDB ({existing_dim}-dim).")
                    return vdb
                else:
                    logger.warning(f"Dim mismatch: file={existing_dim}, expected={dim}. Overwriting.")

            new_vdb = NanoVectorDBStorage(
                namespace=new_namespace,
                global_config={"working_dir": self.working_dir, "embedding_batch_num": self.embedding_batch_num},
                embedding_func=embedding_func,
                meta_fields=meta_fields,
            )
            new_vdb._client._NanoVectorDB__storage = {
                "embedding_dim": dim,
                "data": [],
                "matrix": np.empty((0, dim), dtype=np.float32),
            }
            new_vdb._client.save()
            logger.info(f"Initialized new GNN VDB ({dim}-dim) → {new_storage_file}")
            return new_vdb

        except Exception as e:
            logger.error(f"Error while initializing GNN VDB: {e}")
            raise

    async def init_new_gnn_vdbs(self):

        self.gnn_entities_vdb = await self.init_gnn_vdb(
            "entities", {"entity_name"}, dim=self.fine_tune_config["output_dim"]
        )
        self.gnn_hyperedges_vdb = await self.init_gnn_vdb(
            "hyperedges", {"hyperedge_name"}, dim=self.fine_tune_config["output_dim"]
        )

    @staticmethod
    def dataset_unduplicate(knowledge_pair: List[dict], mode: str) -> List[str]:
        seen = set()
        result = []

        for case in knowledge_pair:
            for edit in case["knowledge_edits"]:

                if edit["old"].strip() == edit["new"].strip():
                    continue

                text = edit[mode].strip()
                if text not in seen:
                    seen.add(text)
                    result.append(text)

        return result

    @staticmethod
    def int_to_binvec(x: int, bits: int) -> np.ndarray:
        bits_arr = np.unpackbits(np.frombuffer(x.to_bytes(bits // 8, "big"), dtype=np.uint8))
        return bits_arr

    def extract_entity_candidates(self, text: str) -> list[str]:

        doc = self.nlp(text)

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

    async def build_simhash_faiss_index(
        self,
        f_bits: int = 128,
        top_k: int = 1200,
    ):

        self.F_BITS = f_bits
        self.TOP_K = top_k

        old_sentences = await self.hyperedges_vdb.get_all_hyperedges()

        logger.info(f"Old sentences: {len(old_sentences)}")

        logger.info("Precomputing NLP tables (spans)...")

        self.precomputed_old = {}

        for s in tqdm(old_sentences, desc="NLP Precompute", ncols=100):

            old_spans = self.extract_entity_candidates(s)

            self.precomputed_old[s] = {
                "spans": old_spans,
            }

        logger.info(f"Precomputed NLP table ready: {len(self.precomputed_old)}")

        logger.info("Building FAISS binary index...")

        old_hashes = [Simhash(s, f=self.F_BITS).value for s in tqdm(old_sentences, desc="Simhash encoding (old)")]

        binary_matrix = np.packbits([self.int_to_binvec(h, self.F_BITS) for h in old_hashes], axis=1).astype(np.uint8)

        index = faiss.IndexBinaryFlat(self.F_BITS)

        index.add(binary_matrix)
        logger.info(f"FAISS index built with {len(old_sentences)} vectors")

        # Step 5: 存成 class 屬性
        self.faiss_index = index
        self.old_sentences = old_sentences

        print("SimHash + FAISS index ready for query\n")

    def simhash_query(self, text: str) -> tuple[np.ndarray, np.ndarray, List[str]]:

        if not hasattr(self, "faiss_index"):
            raise RuntimeError("FAISS index not built yet. Please call build_simhash_faiss_index() first.")

        h = Simhash(text, f=self.F_BITS).value
        bits_arr = np.unpackbits(np.frombuffer(h.to_bytes(self.F_BITS // 8, "big"), dtype=np.uint8))
        packed = np.packbits(bits_arr).astype(np.uint8).reshape(1, -1)
        start = time.time()
        D, I = self.faiss_index.search(packed, self.TOP_K)
        elapsed = time.time() - start

        self.query_latency += elapsed

        return D, I, [self.old_sentences[i] for i in I[0]]

    def _update_faiss_index(self, new_text: str, mode: str = "add"):

        if not hasattr(self, "faiss_index"):
            logger.warning("FAISS index not found. Please call build_simhash_faiss_index() first.")
            return

        if mode == "add":

            h = Simhash(new_text, f=self.F_BITS).value
            bits_arr = self.int_to_binvec(h, self.F_BITS)
            packed = np.packbits(bits_arr).astype(np.uint8).reshape(1, -1)
            self.faiss_index.add(packed)
            self.old_sentences.append(new_text)
            logger.info(f"FAISS index updated: added '{new_text}'")

        elif mode == "rebuild":

            logger.info("Rebuilding FAISS index after edit...")
            hashes = [Simhash(s, f=self.F_BITS).value for s in tqdm(self.old_sentences, desc="Rebuild SimHash")]
            binary_matrix = np.packbits([self.int_to_binvec(h, self.F_BITS) for h in hashes], axis=1).astype(np.uint8)

            new_index = faiss.IndexBinaryFlat(self.F_BITS)
            new_index.add(binary_matrix)
            self.faiss_index = new_index
            logger.info(f"Rebuilt FAISS index with {len(self.old_sentences)} sentences.")

    async def edit_hypergraph(self, knowledge_pair: List[dict]):

        def is_diff_word_subject(diff_phrase: str, spans: List[str]) -> bool:

            diff_phrase_norm = diff_phrase.strip().lower()

            for idx, span in enumerate(spans):
                if diff_phrase_norm in span.lower():
                    if idx >= len(spans) // 2:
                        return False
                    else:
                        return True

            return True

        new_knowledges = self.dataset_unduplicate(knowledge_pair, mode="new")
        node_embeddings_dict = {}
        record_edits_process = []

        memory_save_path = os.path.join(self.fine_tune_config["save_dir"], "memory_footprint.json")
        with open(memory_save_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        for new_knowledge in tqdm_async(new_knowledges, desc="Processing edits"):
            new_knowledge_explicit = f'<hyperedge>"{new_knowledge}"'

            if await self.chunk_entity_relation_graph.has_node(new_knowledge_explicit):
                logger.info(f"[Skip] New knowledge already exists in graph: {new_knowledge_explicit}")
                record_edits_process.append(
                    {"new": new_knowledge, "old": new_knowledge, "category": "Already exists in graph."}
                )

                continue

            new_spans = self.extract_entity_candidates(new_knowledge)
            _, _, candidates = self.simhash_query(new_knowledge)

            replace_target = None
            replace_index = None
            add_target = None
            add_index = None

            for candidate in candidates:

                old_spans = self.precomputed_old[candidate]["spans"]

                diff_old = [tok for tok in old_spans if tok not in new_spans]
                diff_new = [tok for tok in new_spans if tok not in old_spans]

                if len(diff_old) == 1 and len(diff_new) == 1:
                    old_only = diff_old[0]
                    if not is_diff_word_subject(old_only, old_spans) and not is_diff_word_subject(
                        diff_new[0], new_spans
                    ):
                        replace_target = candidate
                        replace_index = candidates.index(candidate)
                        logger.info(f"Candidate {candidate} marked as Replace at rank {replace_index}")
                        break
                    else:
                        if add_target is None:
                            add_target = candidate
                            add_index = candidates.index(candidate)
                        continue

                elif add_target is None and len(diff_old) == 2 and len(diff_new) == 2:
                    add_target = candidate
                    add_index = candidates.index(candidate)
                    continue

            if replace_target is not None:
                await self.edit_hyperedge(replace_target, new_knowledge)
                logger.info(f"[Replace] Replaced '{replace_target}' with '{new_knowledge}'")

                self.precomputed_old[new_knowledge] = {
                    "spans": new_spans,
                }

                if replace_target in self.precomputed_old:
                    del self.precomputed_old[replace_target]
                # --- FAISS 更新 ---
                if replace_target in self.old_sentences:
                    self.old_sentences.remove(replace_target)
                self.old_sentences.append(new_knowledge)
                self._update_faiss_index(new_knowledge, mode="rebuild")
                node_embeddings_dict = await asyncio.to_thread(self.generate_gnn_embeddings)

                record_edits_process.append(
                    {"new": new_knowledge, "old": replace_target, "category": "Replace", "replace_rank": replace_index}
                )
                self.record_memory_after_each_edit(
                    new_knowledge=new_knowledge,
                    old_knowledge=replace_target,
                    category="Replace",
                    rank=replace_index,
                )
                continue

            if add_target is not None:
                await self.insert_hyperedge(new_knowledge)
                logger.info(f"[Add] Added new hyperedge '{new_knowledge}', based on candidate '{add_target}'")

                self.precomputed_old[new_knowledge] = {
                    "spans": new_spans,
                }

                self._update_faiss_index(new_knowledge, mode="add")
                node_embeddings_dict = await asyncio.to_thread(self.generate_gnn_embeddings)

                record_edits_process.append(
                    {"new": new_knowledge, "old": add_target, "category": "Add", "add_rank": add_index}
                )
                self.record_memory_after_each_edit(
                    new_knowledge=new_knowledge,
                    old_knowledge=add_target,
                    category="Add",
                    rank=add_index,
                )
                continue

            logger.info(f"[Skip] No replace/add case found for '{new_knowledge}'")
            node_embeddings_dict = await asyncio.to_thread(self.generate_gnn_embeddings)
            record_edits_process.append({"new": new_knowledge, "old": None, "category": "Skip"})
            self.record_memory_after_each_edit(
                new_knowledge=new_knowledge,
                old_knowledge=None,
                category="Skip",
            )

        with open(
            os.path.join(self.fine_tune_config["save_dir"], "edit_hypergraph_process_log.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(record_edits_process, f, ensure_ascii=False, indent=2)

        logger.info("Writing final GNN embeddings to NanoVectorDB...")

        G = self.chunk_entity_relation_graph._graph

        for i, node in enumerate(G.nodes):
            target_vdb = self.gnn_hyperedges_vdb if node.startswith("<hyperedge>") else self.gnn_entities_vdb
            target_vdb.store_gnn_embeddings({node: node_embeddings_dict[node]})

        self.gnn_hyperedges_vdb._client.save()
        self.gnn_entities_vdb._client.save()

        logger.info("GNN embeddings written to NanoVectorDB.")

        self.clean_orphan_gnn_vectors()
        logger.info("Cleaned orphan GNN vectors after edit.")

        memory_stats = self.collect_memory_footprint()

        save_path = os.path.join(self.fine_tune_config["save_dir"], "memory_footprint.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        history.append(memory_stats)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Memory footprint saved: "
            f"FAISS={memory_stats['faiss_index_size_kb']:.2f} KB, "
            f"Embedding={memory_stats['embedding_size_mb']:.2f} MB, "
            f"Hypergraph={memory_stats['hypergraph_size_mb']:.2f} MB"
        )

        def summarize_param_counts(model: GNNLoRA):
            total_params = sum(p.numel() for p in model.parameters())
            lora_params = sum(p.numel() for p in model.conv.parameters())
            base_params = total_params - lora_params
            return {
                "pretrain_epochs": self.fine_tune_config["num_epochs"],
                "weight_decay": self.fine_tune_config["weight_decay"],
                "learning_rate": self.fine_tune_config["learning_rate"],
                "gnn_type": self.fine_tune_config["gnn_type"],
                "gnn_layer_num": self.fine_tune_config["gnn_layer_num"],
                "lora_rank": self.fine_tune_config["lora_rank"],
                "gnn_base_params": base_params,
                "lora_params": lora_params,
                "total_params": total_params,
            }

        if hasattr(self, "gnn_lora_model"):
            param_stats = summarize_param_counts(self.gnn_lora_model)
            save_path = os.path.join(self.fine_tune_config["save_dir"], "param_stats.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(param_stats, f, indent=2)
            logger.info(f"GNN parameter stats saved to {save_path}")
        else:
            logger.warning("self.gnn_lora_model not found. Cannot log parameter statistics.")

    async def edit_hyperedge(self, old_knowledge: str, new_knowledge: str):

        old_knowledge_explicit = f'<hyperedge>"{old_knowledge}"'

        try:

            if not await self.chunk_entity_relation_graph.has_node(old_knowledge_explicit):
                logger.warning(f"Old knowledge '{old_knowledge_explicit}' not found in the graph.")
                return

            edges_to_remove = list(self.chunk_entity_relation_graph._graph.edges(old_knowledge_explicit))

            if not edges_to_remove:
                logger.warning(f"No edges found for old knowledge '{old_knowledge_explicit}'.")
            else:

                find_related_ids_result = await self.find_related_ids(old_knowledge_explicit)
                for edge in edges_to_remove:
                    self.chunk_entity_relation_graph._graph.remove_edge(*edge)
                logger.info(f"Removed all edges connected to '{old_knowledge_explicit}'.")

            self.chunk_entity_relation_graph._graph.remove_node(old_knowledge_explicit)
            logger.info(f"Removed node '{old_knowledge_explicit}' from the graph.")

            if not find_related_ids_result:
                logger.warning(f"No related IDs found for {old_knowledge_explicit}, skipping deletion.")
                return

            chunk_id = find_related_ids_result["chunk_id"]
            if chunk_id:
                chunk_data = await self.text_chunks.get_by_id(chunk_id)
                if chunk_data:
                    full_doc_id = chunk_data.get("full_doc_id")
                    if full_doc_id:
                        await self.full_docs.delete_by_id(full_doc_id)
                        await self.full_docs.index_done_callback()

            delete_targets = [
                (self.text_chunks, chunk_id),
                (self.chunks_vdb, chunk_id),
            ]

            for store, key in delete_targets:
                if not key:
                    continue
                await store.delete_by_id(key)
                await store.index_done_callback()

            degrees = await asyncio.gather(
                *[self.chunk_entity_relation_graph.node_degree(key) for key in find_related_ids_result["entity_ids"]]
            )

            delete_keys = [key for key, deg in zip(find_related_ids_result["entity_ids"], degrees) if deg == 0]

            await asyncio.gather(*[self.entities_vdb.delete_entity(key) for key in delete_keys])

            await asyncio.gather(
                *[self.hyperedges_vdb.delete_hyperedge(key) for key in find_related_ids_result["hyperedge_ids"]]
            )

            for node in delete_keys:
                if self.chunk_entity_relation_graph._graph.has_node(node):
                    self.chunk_entity_relation_graph._graph.remove_node(node)
                    logger.info(f"Removed isolated entity node '{node}' from the graph.")

            await self.entities_vdb.index_done_callback()
            await self.hyperedges_vdb.index_done_callback()
            await self.chunk_entity_relation_graph.index_done_callback()
            logger.info(
                f"Deleted '{old_knowledge_explicit}' from full_docs, text_chunks, graph, entities_vdb, hyperedges_vdb, and chunks_vdb."
            )

            await self.ainsert(new_knowledge)
            logger.info(f"Inserted new knowledge node '{new_knowledge}'.")

        except Exception as e:
            logger.error(f"Error while editing hyperedge: {e}")

    def collect_memory_footprint(self) -> dict:
        G = self.chunk_entity_relation_graph._graph

        total_nodes = len(G.nodes)
        entity_num = sum(1 for n in G.nodes if not str(n).startswith("<hyperedge>"))
        hyperedge_num = sum(1 for n in G.nodes if str(n).startswith("<hyperedge>"))

        faiss_index_num = len(self.old_sentences) if hasattr(self, "old_sentences") else 0
        faiss_bits = getattr(self, "F_BITS", 128)
        faiss_bytes_per_vector = faiss_bits // 8
        faiss_index_size_bytes = faiss_index_num * faiss_bytes_per_vector

        text_dim = 1536
        structural_dim = self.fine_tune_config.get("output_dim", 256)
        bytes_per_float = 4

        embedding_size_bytes = (text_dim + structural_dim) * bytes_per_float * (entity_num + hyperedge_num)

        graphml_candidates = [
            os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml"),
        ]

        graphml_path = None
        for p in graphml_candidates:
            if os.path.exists(p):
                graphml_path = p
                break

        hypergraph_size_bytes = os.path.getsize(graphml_path) if graphml_path else 0

        return {
            "total_nodes": total_nodes,
            "entity_num": entity_num,
            "hyperedge_num": hyperedge_num,
            "faiss_index_num": faiss_index_num,
            "faiss_bits": faiss_bits,
            "faiss_bytes_per_vector": faiss_bytes_per_vector,
            "faiss_index_size_bytes": faiss_index_size_bytes,
            "faiss_index_size_kb": faiss_index_size_bytes / 1024,
            "embedding_size_bytes": embedding_size_bytes,
            "embedding_size_mb": embedding_size_bytes / 1024 / 1024,
            "hypergraph_size_bytes": hypergraph_size_bytes,
            "hypergraph_size_mb": hypergraph_size_bytes / 1024 / 1024,
        }

    def save_memory_footprint_step(self, step_info: dict):
        import os
        import json

        save_path = os.path.join(self.fine_tune_config["save_dir"], "memory_footprint.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        step_info["edit_step"] = len(history) + 1
        history.append(step_info)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def record_memory_after_each_edit(self, new_knowledge: str, old_knowledge: str, category: str, rank: int = None):
        memory_stats = self.collect_memory_footprint()
        memory_stats["new"] = new_knowledge
        memory_stats["old"] = old_knowledge
        memory_stats["category"] = category
        if rank is not None:
            memory_stats["rank"] = rank

        self.save_memory_footprint_step(memory_stats)

        logger.info(
            f"[Memory Step] "
            f"category={category}, "
            f"faiss_num={memory_stats['faiss_index_num']}, "
            f"nodes={memory_stats['total_nodes']}, "
            f"embedding={memory_stats['embedding_size_mb']:.2f} MB, "
            f"hypergraph={memory_stats['hypergraph_size_mb']:.2f} MB"
        )

    async def find_related_ids(self, old_knowledge_explicit: str):

        try:
            connected_edges = await self.chunk_entity_relation_graph.get_node_edges(old_knowledge_explicit)

            if not connected_edges:
                logger.warning(f"No connected edges found for '{old_knowledge_explicit}'.")
                return {
                    "chunk_id": None,
                    "hyperedge_ids": [],
                    "entity_ids": [],
                }

            chunk_id = None
            hyperedge_ids = []
            entity_ids = []

            for source, target in connected_edges:

                node_data = await self.chunk_entity_relation_graph.get_node(old_knowledge_explicit)
                if node_data:
                    chunk_id = node_data.get("source_id", None)

                if source == old_knowledge_explicit:
                    hyperedge_ids.append(source)
                if target == old_knowledge_explicit:
                    hyperedge_ids.append(target)

                if source != old_knowledge_explicit:
                    entity_ids.append(source)
                if target != old_knowledge_explicit:
                    entity_ids.append(target)

            # 去重
            entity_ids = list(set(entity_ids))
            hyperedge_ids = list(set(hyperedge_ids))

            logger.info(
                f"Found related IDs for '{old_knowledge_explicit}': chunk_id={chunk_id}, hyperedge_ids={hyperedge_ids}, entity_ids={entity_ids}"
            )
            return {
                "chunk_id": chunk_id,
                "hyperedge_ids": hyperedge_ids,
                "entity_ids": entity_ids,
            }

        except Exception as e:
            logger.error(f"Error while finding related IDs for '{old_knowledge_explicit}': {e}")

            return {
                "chunk_id": None,
                "hyperedge_ids": [],
                "entity_ids": [],
            }

    async def insert_hyperedge(self, new_knowledge: str):

        await self.ainsert(new_knowledge)

    def is_entity_node(self, token: str) -> bool:

        token = token.upper()

        if not self.chunk_entity_relation_graph._graph.has_node(token):
            return False

        node_data = self.chunk_entity_relation_graph._graph.nodes[token]
        return node_data.get("role") == "entity"

    async def aquery_with_projector(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        mode: str = "lm",  # "lm", "gnn", "hybrid"
    ):

        gpu_id = self.fine_tune_config["device"]
        if torch.cuda.is_available() and torch.cuda.device_count() > int(gpu_id):
            device = torch.device(f"cuda:{gpu_id}")
        else:
            logger.warning("CUDA unavailable, using CPU for query projector.")
            device = torch.device("cpu")

        if not hasattr(self, "projector") or self.projector is None:
            projector = SentenceProjector(sbert_dim=1536, gnn_dim=self.fine_tune_config["output_dim"])
            model_path = os.path.join(self.fine_tune_config["save_dir"], self.fine_tune_config["save_name"])
            state_dict = torch.load(model_path, map_location=device)

            projector_weights = {
                k.replace("gnn.projector.", ""): v for k, v in state_dict.items() if "gnn.projector" in k
            }
            projector.load_state_dict(projector_weights, strict=False)

            self.projector = projector.to(device).eval()
            logger.info(f"Projector loaded once and cached on {device}.")

            set_global_projector(self.projector, str(device))

        if mode == "lm":
            lm_output = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.hyperedges_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,
                use_gnn_vdb=False,
            )
            result = {"lm_results": lm_output["response"], "lm_context": lm_output["context"]}

        elif mode == "gnn":
            gnn_output = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.gnn_entities_vdb,
                self.gnn_hyperedges_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=None,
                use_gnn_vdb=True,
            )
            result = {"gnn_results": gnn_output["response"], "gnn_context": gnn_output["context"]}

        elif mode == "hybrid":

            result = await kg_query_mixed(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.hyperedges_vdb,
                self.gnn_entities_vdb,
                self.gnn_hyperedges_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
            await self._query_done()

            result = {
                "mix_results": result["response"],
                "mix_context": result["context"],
            }

        else:
            raise ValueError(f"Unknown query mode: {mode}")

        await self._query_done()
        return result

    async def aquery_my(self, query: str, param: QueryParam = QueryParam(), mode: str = "lm"):
        """
        mode = "lm" | "gnn" | "hybrid"
        """
        return await self.aquery_with_projector(query, param, mode)

    # ablation: open source LLM
    async def aquery_with_projector_local(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        mode: str = "lm",  # "lm", "gnn", "hybrid"
    ):

        gpu_id = self.fine_tune_config["device"]
        if torch.cuda.is_available() and torch.cuda.device_count() > int(gpu_id):
            device = torch.device(f"cuda:{gpu_id}")
        else:
            logger.warning("CUDA unavailable, using CPU for query projector.")
            device = torch.device("cpu")

        if not hasattr(self, "projector") or self.projector is None:
            projector = SentenceProjector(sbert_dim=1536, gnn_dim=256)
            model_path = os.path.join(self.fine_tune_config["save_dir"], self.fine_tune_config["save_name"])
            state_dict = torch.load(model_path, map_location=device)

            projector_weights = {
                k.replace("gnn.projector.", ""): v for k, v in state_dict.items() if "gnn.projector" in k
            }
            projector.load_state_dict(projector_weights, strict=False)

            self.projector = projector.to(device).eval()
            logger.info(f"Projector loaded once and cached on {device}.")

            set_global_projector(self.projector, str(device))

        if mode == "lm":
            lm_output = await kg_query_local(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.hyperedges_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,
                use_gnn_vdb=False,
            )
            result = {"lm_results": lm_output["response"], "lm_context": lm_output["context"]}

        elif mode == "gnn":
            gnn_output = await kg_query_local(
                query,
                self.chunk_entity_relation_graph,
                self.gnn_entities_vdb,
                self.gnn_hyperedges_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=None,
                use_gnn_vdb=True,
            )
            result = {"gnn_results": gnn_output["response"], "gnn_context": gnn_output["context"]}

        elif mode == "hybrid":

            result = await kg_query_mixed_local(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.hyperedges_vdb,
                self.gnn_entities_vdb,
                self.gnn_hyperedges_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
            await self._query_done()

            result = {
                "mix_results": result["response"],
                "mix_context": result["context"],
            }

        else:
            raise ValueError(f"Unknown query mode: {mode}")

        await self._query_done()
        return result

    async def aquery_my_local(self, query: str, param: QueryParam = QueryParam(), mode: str = "lm"):
        """
        mode = "lm" | "gnn" | "hybrid"
        """
        return await self.aquery_with_projector_local(query, param, mode)
