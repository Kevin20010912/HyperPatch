import asyncio
import html
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB
import json
from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
)
import torch
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)
from .GNN_model import SentenceProjector


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            ({k: v for k, v in self._data[id].items() if k in fields} if self._data.get(id, None) else None)
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}

    async def delete_by_id(self, ids: Union[str, list[str]]):
        try:
            if isinstance(ids, str):
                ids = [ids]

            for id in ids:
                if id in self._data:
                    del self._data[id]
                    logger.info(f"Deleted entry with id '{id}' from JsonKVStorage.")
                else:
                    logger.warning(f"Entry with id '{id}' not found in JsonKVStorage.")
        except Exception as e:
            logger.error(f"Error while deleting ids '{ids}' from JsonKVStorage: {e}")


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = None

    def __post_init__(self):
        self._client_file_name = os.path.join(self.global_config["working_dir"], f"vdb_{self.namespace}.json")
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(self.embedding_func.embedding_dim, storage_file=self._client_file_name)
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
                "original_content": v.get("original_content", ""),
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [contents[i : i + self._max_batch_size] for i in range(0, len(contents), self._max_batch_size)]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(total=len(embedding_tasks), desc="Generating embeddings", unit="batch")
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            try:
                storage = self._client._NanoVectorDB__storage
                data_len = len(storage["data"])
                matrix_len = (
                    storage["matrix"].shape[0] if isinstance(storage["matrix"], np.ndarray) else len(storage["matrix"])
                )
                if data_len > matrix_len:
                    missing = data_len - matrix_len
                    vec_dim = self.embedding_func.embedding_dim
                    logger.warning(
                        f"[Fix] Expanding NanoVectorDB matrix by {missing} rows (from {matrix_len} → {data_len})"
                    )
                    pad = np.zeros((missing, vec_dim), dtype=np.float32)
                    if isinstance(storage["matrix"], list):
                        storage["matrix"].extend(pad.tolist())
                    else:
                        storage["matrix"] = np.vstack([storage["matrix"], pad])
                    self._client.save()
            except Exception as e:
                logger.error(f"Matrix sync patch failed: {e}")
            return results

        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}")

    async def query(self, query: str, top_k=5):
        logger.info(f"query {query}")
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )

        results = [{**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results]

        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_hyperedge(self, hyperedge_name: str):
        try:
            hyperedge_id = [compute_mdhash_id(hyperedge_name, prefix="rel-")]

            if self._client.get(hyperedge_id):
                self._client.delete(hyperedge_id)
                logger.info(f"Hyperedge {hyperedge_name} have been deleted.")
            else:
                logger.info(f"No hyperedge found with name {hyperedge_name}.")
        except Exception as e:
            logger.error(f"Error while deleting hyperedge {hyperedge_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            relations = [
                dp for dp in self.client_storage["data"] if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(f"All relations related to entity {entity_name} have been deleted.")
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting relations for entity {entity_name}: {e}")

    async def index_done_callback(self):
        self._client.save()

    def get_vector_from_matrix_by_name(self, name: str) -> Union[np.ndarray, None]:
        try:
            storage = self._client._NanoVectorDB__storage
            for i, data in enumerate(storage["data"]):
                if name in (data.get("entity_name", ""), data.get("hyperedge_name", "")):
                    return storage["matrix"][i]
            logger.warning(f"Vector for name '{name}' not found in the matrix.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector for name '{name}': {e}")
            return None

    async def delete_by_id(self, ids: Union[str, list[str]]):
        try:
            if isinstance(ids, str):
                ids = [ids]

            for id in ids:
                result = self._client.get([id])
                if result:
                    self._client.delete([id])
                    logger.info(f"Deleted entry with id '{id}' from NanoVectorDBStorage.")
                else:
                    logger.warning(f"Entry with id '{id}' not found in NanoVectorDBStorage.")
        except Exception as e:
            logger.error(f"Error while deleting ids '{ids}' from NanoVectorDBStorage: {e}")

    async def get_info_from_id(self, node_id: str):
        result = self._client.get([node_id])
        return result[0] if result else None

    def store_gnn_embeddings(self, embeddings: dict[str, np.ndarray]):
        """
        將 GNN 生成的嵌入存儲到 NanoVectorDB 中，並同步 data 和 matrix。
        自動判斷實體 (entity) 或 超邊 (hyperedge)。
        """

        try:
            storage = self._client._NanoVectorDB__storage

            if not hasattr(self, "_id2idx"):
                self._id2idx = {entry["__id__"]: i for i, entry in enumerate(storage["data"])}

            for node_name, embed in embeddings.items():
                is_hyperedge = node_name.startswith("<hyperedge>")
                meta_field = "hyperedge_name" if is_hyperedge else "entity_name"
                prefix = "rel-" if is_hyperedge else "ent-"
                node_id = compute_mdhash_id(node_name, prefix=prefix)

                if node_id in self._id2idx:
                    i = self._id2idx[node_id]
                    if isinstance(storage["matrix"], list):
                        storage["matrix"][i] = embed.tolist()
                    else:
                        storage["matrix"][i] = embed
                else:
                    new_entry = {"__id__": node_id, meta_field: node_name}
                    storage["data"].append(new_entry)
                    self._id2idx[node_id] = len(storage["data"]) - 1

                    embed_row = np.array(embed, dtype=np.float32).reshape(1, -1)
                    if isinstance(storage["matrix"], list):
                        storage["matrix"].append(embed.tolist())
                    elif isinstance(storage["matrix"], np.ndarray):
                        storage["matrix"] = np.vstack([storage["matrix"], embed_row])
                    else:
                        storage["matrix"] = embed_row

        except Exception as e:
            logger.error(f"Error while storing GNN embeddings: {e}")

    async def get_all_hyperedges(self) -> list[str]:
        try:
            storage = self._client._NanoVectorDB__storage
            data_entries = storage.get("data", [])

            hyperedges = []
            for entry in data_entries:
                name = entry.get("original_content", "")

                if not name:
                    continue

                hyperedges.append(name)

            logger.info(f"Retrieved {len(hyperedges)} cleaned hyperedges from {self.namespace}.")
            return hyperedges

        except Exception as e:
            logger.error(f"Error while retrieving hyperedges from {self.namespace}: {e}")
            return []

    async def query_by_vector(self, vector: np.ndarray, top_k=5):
        """
        根據提供的向量查詢最相似的項目，用於 override_embedding 查詢模式。
        :param vector: 形狀為 (dim,) 的 numpy array
        :param top_k: 回傳前幾個最相近的結果
        :return: list[dict]，包含 id, distance, 以及 meta 資訊
        """
        if vector.ndim == 1:
            query_vector = vector
        elif vector.ndim == 2 and vector.shape[0] == 1:
            query_vector = vector[0]
        else:
            raise ValueError(f"query_by_vector() expects a 1D vector or shape (1, dim), got {vector.shape}")

        results = self._client.query(
            query=query_vector,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [{**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results]
        return results

    def get_matrix(self) -> np.ndarray:
        """
        回傳目前儲存在 NanoVectorDB 裡的向量矩陣 matrix。
        """
        return self._client._NanoVectorDB__storage["matrix"]

    async def has_many(self, ids: list[str]) -> dict[str, bool]:
        """
        批次檢查哪些 id 存在於 NanoVectorDB 中。
        回傳一個 dict，key 為 id，value 為 bool。
        """
        try:
            result = self._client.get(ids)
            result_ids = set([r["__id__"] for r in result])
            return {id_: id_ in result_ids for id_ in ids}
        except Exception as e:
            logger.error(f"Error in has_many for ids={ids}: {e}")
            return {id_: False for id_ in ids}


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(self.global_config["working_dir"], f"graph_{self.namespace}.graphml")
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

        self._gnn_embeddings_file = os.path.join(
            self.global_config["working_dir"], f"gnn_embeddings_{self.namespace}.json"
        )
        self._gnn_embeddings = load_json(self._gnn_embeddings_file) or {}

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    # mydesign
    async def delete_edge(self, source_node_id: str, target_node_id: str):
        """
        Delete an edge from the graph based on the specified source and target node IDs.

        :param source_node_id: The source node ID of the edge
        :param target_node_id: The target node ID of the edge
        """
        if self._graph.has_edge(source_node_id, target_node_id):
            self._graph.remove_edge(source_node_id, target_node_id)
            logger.info(f"Edge from {source_node_id} to {target_node_id} deleted from the graph.")
        else:
            logger.warning(f"Edge from {source_node_id} to {target_node_id} not found in the graph for deletion.")

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def get_edges_data(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id, data=True))
        return None

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
