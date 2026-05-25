import torch
import torch.nn.functional as F
from torch.optim import Adam
from hypergraphrag.GNN_model import GNN
from hypergraphrag.graphml_loader import add_index_and_uuid, graph_to_nodeinfo_list
from hypergraphrag.NodeInfo import get_list_node_info_to_matrix
import numpy as np
import random
import networkx as nx
import os
from hypergraphrag.storage import NanoVectorDBStorage
import matplotlib.pyplot as plt
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_graph(filepath: str) -> nx.Graph:
    return nx.read_graphml(filepath)


def gae_loss(z, edge_index, num_nodes, neg_sample_ratio=1.0):
    """
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


def preprocess_graph(
    G: nx.Graph, device: torch.device, evdb: NanoVectorDBStorage, hvdb: NanoVectorDBStorage
) -> tuple[torch.Tensor, torch.Tensor]:
    add_index_and_uuid(G)
    node_info_list = graph_to_nodeinfo_list(G, evdb, hvdb)
    embeds, relations = get_list_node_info_to_matrix(node_info_list)
    return embeds.to(device), relations.to(device)


def pretrain_base_gnn(G, config, device, evdb: NanoVectorDBStorage, hvdb: NanoVectorDBStorage):

    try:
        embeds, relations = preprocess_graph(G, device, evdb, hvdb)
        print(f"Graph preprocessed: Embeds shape {embeds.shape}, Relations shape {relations.shape}")
    except Exception as e:
        print(f"Error during graph preprocessing: {e}")
        return None

    if embeds.shape[0] == 0 or relations.shape[1] == 0:
        print("Error: Empty graph or no relations found. Pre-training aborted.")
        return None

    model = GNN(
        text_emb_dim=config["input_dim"],
        gnn_input_dim=config["output_dim"],
        out_dim=config["output_dim"],
        activation=torch.nn.ReLU(),
        gnn_type=config["gnn_type"],
        gnn_layer_num=config["gnn_layer_num"],
    ).to(device)

    print(f"GNN Model Initialized ({config['gnn_type']}, {config['gnn_layer_num']} layers).")
    print(f"Architecture: {config['input_dim']} -> ... -> {config['output_dim']}")

    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)

    print("Starting GAE pre-training for base GNN...")
    model.train()

    os.makedirs(config["save_dir"], exist_ok=True)

    save_path = os.path.join(config["save_dir"], config["pretrain_gnn_base_name"])
    plot_path = os.path.join(config["save_dir"], f"{config['pretrain_gnn_base_name']}_loss_plot.png")

    loss_history = []
    best_loss = float("inf")

    for epoch in tqdm(range(1, config["num_epochs"] + 1), desc="Pretraining GNN", unit="epoch"):

        optimizer.zero_grad()

        z = model(embeds, relations)

        loss = info_bce_loss(z, relations, num_nodes=embeds.size(0), neg_sample_ratio=1.0)

        # loss = gae_loss(z, relations, num_nodes=embeds.size(0), neg_sample_ratio=1.0)

        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)

    tqdm.write(f"Best loss = {best_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config["num_epochs"] + 1), loss_history, label="Loss")
    plt.title(f"Pre-training Loss Over Epochs (Best Loss: {best_loss:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path)
    print(f"Loss plot saved to: {plot_path}")

    return model
