from dataclasses import dataclass
import torch
import numpy as np


@dataclass(slots=True)
class NodeInfo:
    index: int
    role: str
    name: str
    source_id: str
    uuid: str
    embed: torch.Tensor
    relation: list[int]
    weight: float = None
    entity_type: str = None
    description: str = None

    def __post_init__(self):
        self.embed = torch.as_tensor(self.embed)

    def get_embedding_and_relations(self) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        return (
            self.embed,
            [(self.index, member) for member in self.relation],
        )


def get_list_node_info_to_matrix(node_infos: list[NodeInfo]) -> tuple[torch.Tensor, torch.Tensor]:
    embeds, relations = [], []
    for node in node_infos:
        embed, relation = node.get_embedding_and_relations()
        embeds.append(embed)
        relations.extend(relation)

    relations = torch.tensor(list(set(relations)), dtype=torch.long).T
    embeds = torch.stack(embeds)

    return embeds, relations
