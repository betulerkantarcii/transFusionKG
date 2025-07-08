#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:26:07 2025

@author: betulerkantarci
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:26:07 2025
@author: betulerkantarci
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from torch_geometric.nn import TransformerConv
import networkx as nx
import numpy as np
import warnings
from pykeen.evaluation import RankBasedEvaluator
from pykeen.sampling import NegativeSampler
from pykeen.typing import MappedTriples
from typing import Dict, Optional, Any, Mapping
from tqdm import tqdm


warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is deprecated")

# === Device setup ===
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# === Step 1: Load datasets ===
df_positive = pd.read_csv('positive_dataset.csv')
df_test = pd.read_csv('test_dataset.csv')

df_positive = df_positive[['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']]
df_test = df_test[['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']]

combined_triples = pd.concat([df_positive, df_test]).values

# === Step 2: Create TriplesFactory ===
combined_factory = TriplesFactory.from_labeled_triples(triples=combined_triples)

triples_factory = TriplesFactory.from_labeled_triples(
    triples=df_positive.values,
    entity_to_id=combined_factory.entity_to_id,
    relation_to_id=combined_factory.relation_to_id
)

testing_factory = TriplesFactory.from_labeled_triples(
    triples=df_test.values,
    entity_to_id=combined_factory.entity_to_id,
    relation_to_id=combined_factory.relation_to_id
)

# === Step 3: Train TransE ===
result = pipeline(
    training=triples_factory,
    testing=testing_factory,
    model='TransE',
    model_kwargs={'embedding_dim': 100, 'scoring_fct_norm': 2},
    training_kwargs={'num_epochs': 100},
    random_seed=42,
    device=device
)

# Extract and move embeddings to correct device
entity_embeddings = result.model.entity_representations[0](indices=None).detach().to(device)
relation_embeddings = result.model.relation_representations[0](indices=None).detach().to(device)
entity_to_id = combined_factory.entity_to_id
relation_to_id = combined_factory.relation_to_id

# === Step 4: Build edge_index for PyG ===
G = nx.DiGraph()
for h, r, t in combined_triples:
    h_id = entity_to_id[h]
    t_id = entity_to_id[t]
    G.add_edge(h_id, t_id)
edge_index = torch.tensor(list(G.edges)).t().contiguous().to(device)

# === Step 5: Graph Transformer Refinement ===
class GraphTransformerRefiner(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.transformer = TransformerConv(in_channels=in_dim, out_channels=out_dim, heads=heads)
        self.proj_in = nn.Linear(in_dim, in_dim)
        self.proj_out = nn.Linear(out_dim * heads, out_dim)

    def forward(self, x, edge_index):
        x = self.proj_in(x)
        x = self.transformer(x, edge_index)
        return self.proj_out(x)

transformer_model = GraphTransformerRefiner(in_dim=100, out_dim=100).to(device)
x_refined = transformer_model(entity_embeddings, edge_index)

# === Step 6: Add Noise and Apply Denoiser ===
class DenoisingModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x_noisy):
        return x_noisy - self.score_net(x_noisy)

denoiser = DenoisingModule(embed_dim=100).to(device)
noise = torch.randn_like(x_refined) * 0.05
x_noisy = x_refined + noise
x_denoised = denoiser(x_noisy)

# === Step 7: Score Function ===
def score_triple(head, tail, relation, entity_embed, relation_embed):
    return -torch.norm(entity_embed[head] + relation_embed[relation] - entity_embed[tail], p=1)

# === Example Use Case: Scoring a known or candidate pair ===
def get_entity_id(name):
    return entity_to_id.get(name)

drug_name = 'C0004057'
disease_name = 'C0002395'
relation_name = 'TREATS'

head = get_entity_id(drug_name)
tail = get_entity_id(disease_name)
rel = relation_to_id[relation_name]

if head is not None and tail is not None and rel is not None:
    score = score_triple(head, tail, rel, x_denoised, relation_embeddings)
    print(f'Score for ({drug_name}, {relation_name}, {disease_name}): {score.item():.4f}')
else:
    print("Triple contains unknown entity or relation.")

# === Step 8: Score Function for Test Set ===
def score_triple_tensor(head_ids, rel_ids, tail_ids, entity_embed, relation_embed):
    head_embed = entity_embed[head_ids]
    tail_embed = entity_embed[tail_ids]
    rel_embed = relation_embed[rel_ids]
    return -torch.norm(head_embed + rel_embed - tail_embed, p=1, dim=1)

test_triples = testing_factory.mapped_triples
head_ids = test_triples[:, 0].to(device)
rel_ids = test_triples[:, 1].to(device)
tail_ids = test_triples[:, 2].to(device)

scores = score_triple_tensor(head_ids, rel_ids, tail_ids, x_denoised, relation_embeddings)

# Convert to list of tuples with scores
scored_triples = []
for idx, score in enumerate(scores):
    h_idx = head_ids[idx].item()
    r_idx = rel_ids[idx].item()
    t_idx = tail_ids[idx].item()
    h_label = testing_factory.entity_labeling.id_to_label[h_idx]
    r_label = testing_factory.relation_labeling.id_to_label[r_idx]
    t_label = testing_factory.entity_labeling.id_to_label[t_idx]
    scored_triples.append((h_label, r_label, t_label, score.item()))

# Normalize scores to [-1, 1]
raw_scores = np.array([s[-1] for s in scored_triples])
min_score = raw_scores.min()
max_score = raw_scores.max()
normalized_scores = 2 * (raw_scores - min_score) / (max_score - min_score) - 1

for i in range(len(scored_triples)):
    scored_triples[i] = (
        scored_triples[i][0],
        scored_triples[i][1],
        scored_triples[i][2],
        normalized_scores[i]
    )

df_scores = pd.DataFrame(scored_triples, columns=["Head", "Relation", "Tail", "Score"])
df_scores.sort_values("Score", ascending=False, inplace=True)
df_scores.to_csv("test_triple_scores.csv", index=False)
print(df_scores.head())

# Save transformer and denoiser
torch.save(transformer_model.state_dict(), "graph_transformer.pt")
torch.save(denoiser.state_dict(), "denoising_module.pt")

