import networkx as nx
from .NodeInfo import NodeInfo
import torch
import uuid
from .storage import NanoVectorDBStorage


def add_index_and_uuid(G: nx.Graph):
    for idx, (node_name, attrs) in enumerate(G.nodes(data=True)):
        attrs["uuid"] = str(uuid.uuid4())
        attrs["index"] = idx


def graph_to_nodeinfo_list(G: nx.Graph, evdb: NanoVectorDBStorage, hvdb: NanoVectorDBStorage) -> list[NodeInfo]:
    node_info_list = []

    for node_name, attrs in G.nodes(data=True):
        if node_name.startswith("<hyperedge>"):
            embed = hvdb.get_vector_from_matrix_by_name(node_name)
        else:
            embed = evdb.get_vector_from_matrix_by_name(node_name)

        if embed is None:
            raise ValueError(f"Node {node_name} does not have an embedding in the vector database.")

        relation = [G.nodes[member].get("index") for member in G.neighbors(node_name)]

        is_hyperedge = attrs.get("role") == "hyperedge"

        node_info_construct = dict(
            index=attrs.get("index"),
            role=attrs.get("role", "default_role"),
            name=(
                node_name.replace("<hyperedge>", "").strip('"').strip()
                if is_hyperedge
                else node_name.strip('"').strip()
            ),
            source_id=attrs.get("source_id"),
            uuid=attrs.get("uuid"),
            embed=embed,
            relation=relation,
        )

        if is_hyperedge:
            node_info_construct["weight"] = float(attrs.get("weight"))
        else:
            entity_type = attrs.get("entity_type").strip('"').strip()
            node_info_construct["entity_type"] = entity_type
            description = attrs.get("description").strip('"').strip()
            node_info_construct["description"] = description

        node_info = NodeInfo(**node_info_construct)

        node_info_list.append(node_info)

    return node_info_list
