import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.fgl_dataset import FGLDataset

import os
import torch






def preprocess_adj_matrix(subgraph,  dataset, seed, client_id, num_clients, batch_size=50):
    device = subgraph.x.device
    cache_dir = "./normalized_adj_matrix"
    os.makedirs(cache_dir, exist_ok=True)
    adj_cache_file = f"{cache_dir}/adj_{dataset}_clients{num_clients}_client{client_id}.pt"

    if os.path.exists(adj_cache_file):
        print(f"Loading cached normalized adj matrix and similarity matrix for client {client_id} from {adj_cache_file}")
    else:
        W = subgraph.adj_label.clone()
        # W = W + torch.eye(W.size(0), device=W.device)
        node_features = subgraph.x
        n_nodes = node_features.size(0)
        similarity_matrix = torch.zeros((n_nodes, n_nodes), device=device)

        for start in range(0, n_nodes, batch_size):
            end = min(start + batch_size, n_nodes)
            features_expanded_in_dim1 = node_features[start:end].unsqueeze(1)
            features_expanded_in_dim0 = node_features.unsqueeze(0)
            batch_similarity_matrix = torch.nn.functional.cosine_similarity(features_expanded_in_dim1,
                                                                            features_expanded_in_dim0, dim=2)
            similarity_matrix[start:end] = batch_similarity_matrix

        similarity_matrix.fill_diagonal_(1)
        similarity_matrix = similarity_matrix

        W *= similarity_matrix
        row_sum = W.sum(dim=1, keepdim=True)
        nonzero_count = (W != 0).sum(dim=1, keepdim=True)
        row_mean = (row_sum - 1) / (nonzero_count.clamp(min=1) - 1)
        mask = W >= row_mean * 2
        W = W * mask.float()
        row_sums = torch.sum(W, dim=1)
        D = row_sums.float()
        D[D < 1] = 1.0
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt = torch.diag(D_inv_sqrt).to(device)
        normalized_adj_matrix = torch.mm(torch.mm(D_inv_sqrt, W), D_inv_sqrt)
        torch.save(normalized_adj_matrix, adj_cache_file)



def calculate_cosine_similarity_matrix(normalized_ckr, dataset):
    similarity_matrix = torch.nn.functional.cosine_similarity(normalized_ckr.unsqueeze(1), normalized_ckr.unsqueeze(0),dim=2)
    if dataset == 'Cora':
        row_sums = similarity_matrix.sum(dim=1, keepdim=True)
        normalized_similarity_matrix = similarity_matrix / row_sums
    elif dataset == 'CiteSeer':
        similarity_matrix = similarity_matrix / 1.5
        normalized_similarity_matrix = torch.softmax(similarity_matrix, dim=1)
    else:
        similarity_matrix = similarity_matrix / 1
        normalized_similarity_matrix = torch.softmax(similarity_matrix, dim=1)


    return normalized_similarity_matrix



def label_propagation(subgraph, alpha, I, num_classes, dataset, seed,num_clients,client_id,tau_lp,pre=False):
    device = subgraph.x.device
    num_samples = subgraph.num_nodes
    Y = torch.zeros((num_samples, num_classes), device=device)
    if pre == True:
        Y[torch.arange(num_samples), subgraph.y] = 1
        Y_input = torch.zeros_like(Y)
        Y_input[subgraph.train_idx] = Y[subgraph.train_idx]
    else:


        Y[torch.arange(num_samples), subgraph.y_with_pseudo] = 1
        Y_input = torch.zeros_like(Y)
        Y_input[subgraph.idx_train_with_pseudo] = Y[subgraph.idx_train_with_pseudo]
    device = subgraph.x.device
    cache_dir = "./normalized_adj_matrix"
    adj_cache_file = f"{cache_dir}/adj_{dataset}_clients{num_clients}_client{client_id}.pt"
    S = torch.load(adj_cache_file).to(device)
    F_propagation = Y_input.clone().to(device)  # F^(0)
    for _ in range(I):
        F_propagation = alpha * (S @ F_propagation) + (1 - alpha) * Y_input
    non_zero_rows = torch.any(F_propagation != 0, dim=1)
    F_propagation[non_zero_rows] = F_propagation[non_zero_rows] / F_propagation[non_zero_rows].sum(dim=1, keepdim=True)
    max_probs = F_propagation.max(dim=1).values
    kept_mask = (max_probs > tau_lp) | subgraph.train_idx
    kept = kept_mask.unsqueeze(1).float()
    F_propagation = F_propagation * kept


    return F_propagation, max_probs


def PL_Ncontrast(x_dis: torch.Tensor,indicate_matrix: torch.Tensor, hop,beta,dataset,seed,num_clients,client_id) -> torch.Tensor:
    device = x_dis.device
    cache_dir = "./normalized_adj_matrix"
    adj_cache_file = f"{cache_dir}/adj_{dataset}_clients{num_clients}_client{client_id}.pt"
    S_contrast = torch.load(adj_cache_file).to(device)
    adj_label = (S_contrast != 0).float()
    adj_label = (adj_label > 0).float()

    if torch.any(torch.diag(adj_label) == 0):
        adj_label = adj_label + torch.eye(adj_label.size(0), device=adj_label.device)

    client_adj_r_hop_no_diag = ((torch.matrix_power(adj_label, hop) > 0).float()  if hop > 1 else torch.zeros_like(adj_label))
    generated_matrix = ((adj_label == 0) & (client_adj_r_hop_no_diag == 0)).float()
    positive_adj_mask = ((adj_label == 1) | (client_adj_r_hop_no_diag == 1)).float()
    positive_mask = ((indicate_matrix == 1) & (positive_adj_mask == 1)).float()
    negative_mask = ((indicate_matrix == -1) & (generated_matrix == 1)).float()
    all_zero_mask = (indicate_matrix.sum(dim=1) == 0).unsqueeze(1)
    relaxed_positive_mask = (all_zero_mask & (positive_adj_mask == 1)).float()
    relaxed_negative_mask = (all_zero_mask & (generated_matrix == 1)).float()

    final_positive_mask = positive_mask + relaxed_positive_mask
    final_negative_mask = negative_mask + relaxed_negative_mask
    x_exp = torch.exp(beta * x_dis)
    sum_pos = torch.sum(x_exp * final_positive_mask, dim=1)  # (N,)
    sum_neg = torch.sum(x_exp * final_negative_mask, dim=1)  # (N,)
    sum_total = sum_pos + sum_neg  # (N,)
    eps = 1e-8
    sum_total = sum_total + eps
    log_scores = torch.log(x_exp / sum_total.unsqueeze(1) + eps)  # (N, N)
    masked_log_scores = log_scores * final_positive_mask
    sum_log = torch.sum(masked_log_scores, dim=1)  # (N,)
    pos_counts = torch.sum(final_positive_mask, dim=1) + eps  # (N,)
    loss_per_node = -(sum_log / pos_counts)
    loss = loss_per_node.mean()
    del final_positive_mask, final_negative_mask, positive_mask, negative_mask, S_contrast
    torch.cuda.empty_cache()
    return loss


def calculate_class_wise_reliability(subgraphs, num_clients, num_classes):
    device = subgraphs[0].x.device
    ckr = torch.zeros((num_clients, num_classes), device=device)
    for client_id in range(num_clients):
        data = subgraphs[client_id]
        N = data.x.size(0)
        norm_x = F.normalize(data.x, p=2, dim=1)
        train_mask = data.idx_train_with_pseudo.bool()
        src_all, dst_all = data.edge_index  # [E]
        mask = train_mask[src_all]
        filtered_src = src_all[mask]
        filtered_dst = dst_all[mask]

        neighbor_count = torch.zeros(N, device=device)
        node_sim_sum = torch.zeros(N, device=device)

        total_edges = filtered_src.size(0)
        for start in range(0, total_edges, 512):
            end = min(start + 512, total_edges)
            src_chunk = filtered_src[start:end]
            dst_chunk = filtered_dst[start:end]

            sim_chunk = (norm_x[src_chunk] * norm_x[dst_chunk]).sum(dim=1)
            ones = torch.ones_like(sim_chunk)

            neighbor_count.index_add_(0, src_chunk, ones)
            node_sim_sum.index_add_(0, src_chunk, sim_chunk)

            del src_chunk, dst_chunk, sim_chunk, ones
            torch.cuda.empty_cache()

        train_idx = torch.nonzero(train_mask, as_tuple=False).view(-1)
        scores = (node_sim_sum[train_idx] + 1.0) / (neighbor_count[train_idx] + 1.0)
        labels = data.y_with_pseudo[train_idx].long()

        ckr_row = torch.zeros(num_classes, device=device)
        ckr_row.index_add_(0, labels, scores)
        ckr[client_id] = ckr_row

        del filtered_src, filtered_dst, neighbor_count, node_sim_sum
        torch.cuda.empty_cache()

    return ckr

def get_num_classes(dataset_name: str) -> int:
    mapping = {
        'Cora': 7,
        'CiteSeer': 6,
        'PubMed': 3,
        'CS': 15,
        'Physics': 5,
        'Computers': 10
    }
    try:
        return mapping[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    
def calculate_accuracy(predicted_labels, true_labels, mask):
    correct = (predicted_labels[mask] == true_labels[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy

def cal_class_learning_status(logits):
    """
    Compute class-wise learning status as the max softmax probability per class.
    No moving factor; purely uses current batch statistics.
    """

    prob = torch.softmax(logits, dim=1)
    max_probs, max_idx = torch.max(prob, dim=1)
    num_classes = prob.size(1)
    p_model = torch.zeros(num_classes, device=logits.device)

    for cls in range(num_classes):
        mask = (max_idx == cls)
        if mask.any():
            p_model[cls] = max_probs[mask].max()

    max_val = p_model.max()
    if max_val > 0:
        p_model = p_model / max_val

    return p_model




def load_dataset(args, train_size,num_clients):

    cache_dir = "./dataset_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/dataset_{args.dataset}_clients{num_clients}_train{train_size}.pt"

    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        dataset = torch.load(cache_file)
        return dataset

    if args.dataset in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Computers",
        "Photo",
        "CS",
        "Physics",
        "NELL",
    ]:
        dataset = FGLDataset(
            args,
            root=args.root,
            name=args.dataset,
            num_clients=num_clients,
            partition=args.partition,
            train=train_size,
            val=0.4,
            test=0.6-train_size,
            part_delta=args.part_delta,
        )
    elif args.dataset in ["ogbn-arxiv", "Flickr"]:
        dataset = FGLDataset(
            args,
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=train_size,
            val=0.2,
            test=0.8 - train_size,
            part_delta=args.part_delta,
        )
    elif args.dataset in ["Reddit"]:
        dataset = FGLDataset(
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=0.8,
            val=0.1,
            test=0.1,
        )
    elif args.dataset in ["ogbn-products"]:
        dataset = FGLDataset(
            root=args.root,
            name=args.dataset,
            num_clients=args.num_clients,
            partition=args.partition,
            train=train_size,
            val=0.0,
            test=1 - train_size,
            part_delta=args.part_delta,
        )

    torch.save(dataset, cache_file)
    print(f"Dataset saved to {cache_file}")

    return dataset



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

