import torch
import numpy as np
import random
import networkx as nx
from .graphml_loader import add_index_and_uuid, graph_to_nodeinfo_list
from .NodeInfo import get_list_node_info_to_matrix
from .GNN_model import GNN, GNNLoRA
from torch.optim import Adam
from time import time
import os
import torch.nn.functional as F
from .storage import NanoVectorDBStorage
import torch.nn as nn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_graph(filepath: str) -> nx.Graph:
    return nx.read_graphml(filepath)


def preprocess_graph(
    G: nx.Graph, device: torch.device, evdb: NanoVectorDBStorage, hvdb: NanoVectorDBStorage
) -> tuple[torch.Tensor, torch.Tensor]:
    add_index_and_uuid(G)
    node_info_list = graph_to_nodeinfo_list(G, evdb, hvdb)
    embeds, relations = get_list_node_info_to_matrix(node_info_list)
    return embeds.to(device), relations.to(device)


def initialize_gnn_lora_from_gnn_base(
    embeds: torch.Tensor,
    config: dict,
    device: torch.device,
    gnn_base_path: str,
) -> GNNLoRA:

    input_dim = embeds.shape[1]
    output_dim = config["output_dim"]
    gnn_type = config["gnn_type"]
    gnn_layer_num = config["gnn_layer_num"]
    lora_rank = config["lora_rank"]
    activation = nn.ReLU()

    gnn_base = GNN(
        text_emb_dim=input_dim,  # 1536
        gnn_input_dim=output_dim,  # 256
        out_dim=output_dim,  # 256
        activation=activation,
        gnn_type=gnn_type,
        gnn_layer_num=gnn_layer_num,
    )
    state_dict = torch.load(gnn_base_path, map_location=device)
    gnn_base.load_state_dict(state_dict, strict=True)

    # 包成 LoRA 模型
    model = GNNLoRA(
        input_dim=output_dim,  # 256
        out_dim=output_dim,  # 256
        activation=activation,
        gnn=gnn_base,
        gnn_type=gnn_type,
        gnn_layer_num=gnn_layer_num,
        r=lora_rank,
    ).to(device)
    return model


def initialize_gnn_lora(
    embeds: torch.Tensor,
    config: dict,
    device: torch.device,
    gnn_lora_path: str,
) -> GNNLoRA:

    input_dim = embeds.shape[1]
    output_dim = config["output_dim"]
    gnn_type = config["gnn_type"]
    gnn_layer_num = config["gnn_layer_num"]
    lora_rank = config["lora_rank"]
    activation = nn.ReLU()

    gnn_base = GNN(
        text_emb_dim=input_dim,  # 1536
        gnn_input_dim=output_dim,  # 256
        out_dim=output_dim,  # 256
        activation=activation,
        gnn_type=gnn_type,
        gnn_layer_num=gnn_layer_num,
    )
    model = GNNLoRA(
        input_dim=output_dim,  # 256
        out_dim=output_dim,  # 256
        activation=activation,
        gnn=gnn_base,
        gnn_type=gnn_type,
        gnn_layer_num=gnn_layer_num,
        r=lora_rank,
    ).to(device)

    state_dict = torch.load(gnn_lora_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model


def warmup_gnn(
    G: nx.Graph,
    config: dict,
    device: torch.device,
    evdb: NanoVectorDBStorage,
    hvdb: NanoVectorDBStorage,
    gnn_path: str,
    mode: str,
) -> GNNLoRA:

    embeds, relations = preprocess_graph(G, device, evdb, hvdb)

    if mode == "initial":
        model = initialize_gnn_lora_from_gnn_base(embeds, config, device, gnn_path)
    else:
        model = initialize_gnn_lora(embeds, config, device, gnn_path)

    model.eval()

    with torch.no_grad():
        for _ in range(config.get("warmup_steps", 1)):
            _ = model(embeds, relations)

    return model


def info_bce_loss(z, edge_index, num_nodes, neg_sample_ratio=1.0):

    src = edge_index[0]
    dst = edge_index[1]
    num_pos = src.size(0)
    num_neg = int(num_pos * neg_sample_ratio)

    pos_score = torch.sum(z[src] * z[dst], dim=1)
    pos_labels = torch.ones_like(pos_score)

    neg_src = src[torch.randint(0, num_pos, (num_neg,), device=z.device)]
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=z.device)
    neg_score = torch.sum(z[neg_src] * z[neg_dst], dim=1)
    neg_labels = torch.zeros_like(neg_score)

    all_scores = torch.cat([pos_score, neg_score], dim=0)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)

    loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)

    return loss


def gae_loss(z, edge_index, num_nodes, neg_sample_ratio=1.0):
    """
    GAE reconstruction loss with negative sampling.
    z: [N, d]
    edge_index: [2, E]
    """

    src = edge_index[0]
    dst = edge_index[1]
    pos_pred = torch.sum(z[src] * z[dst], dim=1)
    pos_label = torch.ones_like(pos_pred)

    num_pos = src.size(0)
    num_neg = int(num_pos * neg_sample_ratio)

    neg_src = torch.randint(0, num_nodes, (num_neg,), device=z.device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=z.device)

    neg_pred = torch.sum(z[neg_src] * z[neg_dst], dim=1)
    neg_label = torch.zeros_like(neg_pred)

    pred = torch.cat([pos_pred, neg_pred])
    label = torch.cat([pos_label, neg_label])

    loss = F.binary_cross_entropy_with_logits(pred, label)
    return loss


def finetune_gnn_lora(
    G: nx.Graph,
    config: dict,
    model: GNNLoRA,
    device: torch.device,
    evdb: NanoVectorDBStorage,
    hvdb: NanoVectorDBStorage,
):

    embeds, relations = preprocess_graph(G, device, evdb, hvdb)

    model.to(device)
    model.train()
    model.gnn.eval()
    for param in model.gnn.parameters():
        param.requires_grad = False

    for param in model.gnn.projector.parameters():
        param.requires_grad = True

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])

    print("Start fine-tuning LoRA...")

    model_path = os.path.join(config["save_dir"], f"{config['save_name']}")

    optimizer.zero_grad()

    z, _, _ = model(embeds, relations)

    loss = info_bce_loss(z, relations, num_nodes=embeds.size(0), neg_sample_ratio=1.0)
    # loss = gae_loss(z, relations, num_nodes=embeds.size(0), neg_sample_ratio=1.0)

    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), model_path)
    print("Fine-tuning complete!")

    return model
