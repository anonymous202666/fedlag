import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
# ========= Modified Version A: use proxy validation metric (Silhouette or RankMe) instead of val split =========
import argparse 
import os
import warnings
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np
from util.task_util import accuracy
from util.base_util import (
    seed_everything,
    load_dataset,
    cal_class_learning_status,
    get_num_classes,
    PL_Ncontrast,
    label_propagation,
    calculate_cosine_similarity_matrix,
    preprocess_adj_matrix,
    calculate_class_wise_reliability
)
from model import GCN
 
import hdbscan
from hdbscan.validity import validity_index
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch.nn.functional as Fnn
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()

# experimental environment setup
 
parser.add_argument('--hop', type=int, default=1) # r
parser.add_argument('--seed', type=int, default=4880)
parser.add_argument('--root', type=str, default='data/root')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default="Cora") # Cora, PubMed, CS, Physics, ogbn-arxiv
parser.add_argument('--partition', type=str, default="Louvain")
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--num_clients', type=int, default=20)
parser.add_argument('--num_rounds', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--num_dims', type=int, default=64)
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--tau', type=float, default=0.95)
parser.add_argument('--beta', type=float, default=4)
parser.add_argument('--lam', type=float, default=0.25)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--I', type=int, default=1)  
parser.add_argument('--tau_lp', type=float, default=0.75)
parser.add_argument('--R_switch', type=int, default=4)
parser.add_argument('--labeling_ratio', type=float, default=0.01)
parser.add_argument('--neg_ratio', type=float, default=5.0)
args = parser.parse_args()

 
def get_Z_for_rankme(model, data):
    """
    Obtain node representations for proxy metrics.
    We call model.forward(data, return_embed=True) so that it returns (logits, embed),
    and we use embed (N x d) as Z. If not available, we fallback to logits.
    """
    model.eval()
    with torch.no_grad():
        out = model.forward(data, return_embed=True)  # (logits, embed)
        if isinstance(out, tuple) and len(out) == 2:
            logits, embed = out
            Z = embed
        else:
            # Fallback to logits if something unexpected happens
            logits = model.forward(data)
            Z = logits
    return Z


def rankme_effective_rank(Z, center=True, l2norm=True, eps=1e-2):
    """
    Effective rank (RankMe) on Z \in R^{N x d}.
    """
    if l2norm:
        Z = Fnn.normalize(Z, dim=1)
    # Covariance-like matrix + ridge
    C = (Z.T @ Z) / (Z.size(0) + 1e-12)
    C = C + eps * torch.eye(C.size(0), device=C.device, dtype=C.dtype)
    # Symmetric eigendecomposition
    evals = torch.linalg.eigvalsh(C).clamp_min(1e-12)
    p = evals / (evals.sum() + 1e-12)
    H = -(p * (p + 1e-12).log()).sum()
    return torch.exp(H)  # effective rank


def cluster_and_score_hdbscan(Z, random_state=42):
    """
    HDBSCAN(+PCA) clustering then compute silhouette on non-noise points.
    Returns: (labels, silhouette_score or NaN)
    """
    Z = Fnn.normalize(Z, dim=1).detach().cpu().numpy()
    N, d = Z.shape
    m = max(2, min(50, N - 1, d))
    Zr = PCA(n_components=m, random_state=random_state).fit_transform(Z)

    min_cluster_size = max(5, int(0.02 * N))
    min_samples = max(5, int(0.01 * N))

    clt = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                          min_samples=min_samples,
                          metric='euclidean',
                          cluster_selection_method='eom')
    labels = clt.fit_predict(Zr)  # -1 is noise
    sil = float('nan')
    valid_mask = labels >= 0
    if valid_mask.sum() >= 10 and len(set(labels[valid_mask])) >= 2:
        sil = silhouette_score(Zr[valid_mask], labels[valid_mask])
    return labels, float(sil)


# ===================== k-hop contrastive sampling (case2, neg_ratio) =====================
@torch.no_grad()
def _ensure_hhop_cache(subgraph, hop: int):
    """
    Build and cache:
      - all directed k-hop positive candidates (src_all, dst_all), where dst in <=hop neighborhood of src
      - hhop_hash for fast membership test: hash = src * N + dst (sorted unique)
    Computed ONCE per (subgraph, hop).
    """
    if not hasattr(subgraph, "_hhop_cache"):
        subgraph._hhop_cache = {}
    if hop in subgraph._hhop_cache:
        return

    device = subgraph.x.device
    N = int(subgraph.num_nodes)

    # undirected, no self-loops, coalesced
    ei = subgraph.edge_index
    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei, num_nodes=N)
    ei, _ = coalesce(ei, None, N, N)

    if hop <= 1:
        src_all = ei[0].detach().cpu()
        dst_all = ei[1].detach().cpu()
    else:
        # build CSR on CPU for BFS expansion (hop <= 3 typical; done once)
        row = ei[0].detach().cpu().numpy()
        col = ei[1].detach().cpu().numpy()
        order = np.argsort(row, kind='mergesort')
        row = row[order]
        col = col[order]
        deg = np.bincount(row, minlength=N).astype(np.int64)
        ptr = np.zeros(N + 1, dtype=np.int64)
        ptr[1:] = np.cumsum(deg)

        src_chunks = []
        dst_chunks = []

        for i in range(N):
            # 1-hop
            n1 = col[ptr[i]:ptr[i+1]]
            if n1.size == 0:
                continue
            visited = np.zeros(N, dtype=bool)
            visited[i] = True
            visited[n1] = True

            all_nbrs = np.unique(n1).astype(np.int64)
            frontier = all_nbrs.copy()

            # 2..hop
            for _depth in range(2, hop + 1):
                if frontier.size == 0:
                    break
                neigh_list = [col[ptr[u]:ptr[u+1]] for u in frontier]
                if len(neigh_list) == 0:
                    break
                cand = np.concatenate(neigh_list)
                if cand.size == 0:
                    frontier = np.empty(0, dtype=np.int64)
                    continue
                cand = np.unique(cand).astype(np.int64)
                cand = cand[~visited[cand]]
                if cand.size == 0:
                    frontier = np.empty(0, dtype=np.int64)
                    continue
                visited[cand] = True
                all_nbrs = np.unique(np.concatenate([all_nbrs, cand])).astype(np.int64)
                frontier = cand

            # exclude self just in case
            all_nbrs = all_nbrs[all_nbrs != i]
            if all_nbrs.size == 0:
                continue
            src_chunks.append(np.full(all_nbrs.size, i, dtype=np.int64))
            dst_chunks.append(all_nbrs)

        if len(src_chunks) == 0:
            src_all = torch.empty((0,), dtype=torch.long)
            dst_all = torch.empty((0,), dtype=torch.long)
        else:
            src_all = torch.from_numpy(np.concatenate(src_chunks)).long()
            dst_all = torch.from_numpy(np.concatenate(dst_chunks)).long()

    # move to device + build hash for O(log M) membership queries
    src_all = src_all.to(device=device, dtype=torch.long)
    dst_all = dst_all.to(device=device, dtype=torch.long)
    hhop_hash = torch.unique(src_all * N + dst_all)  # sorted unique
    subgraph._hhop_cache[hop] = {
        "src_all": src_all,
        "dst_all": dst_all,
        "hash": hhop_hash,
        "N": N,
    }


def _pair_in_hhop(src: torch.Tensor, dst: torch.Tensor, hhop_hash: torch.Tensor, N: int) -> torch.Tensor:
    """
    Vectorized membership: whether (src, dst) is within <=hop neighborhood.
    """
    h = src * N + dst
    if hhop_hash.numel() == 0:
        return torch.zeros_like(h, dtype=torch.bool)

    idx = torch.searchsorted(hhop_hash, h)
    in_range = idx < hhop_hash.numel()

    # Safe indexing: clamp indices before gather; then mask by in_range.
    idx_safe = idx.clamp(max=hhop_hash.numel() - 1)
    found = hhop_hash[idx_safe] == h
    return in_range & found


def _get_label_state(subgraph):
    """
    Returns:
      has_label: bool mask [N]
      labels: long [N] (valid only where has_label==True, but safe to index)
    """
    if hasattr(subgraph, "idx_train_with_pseudo") and hasattr(subgraph, "y_with_pseudo"):
        has_label = subgraph.idx_train_with_pseudo.bool()
        labels = subgraph.y_with_pseudo.long()
    else:
        has_label = subgraph.train_idx.bool()
        labels = subgraph.y.long()
    return has_label, labels


def _contrastive_multi_pos(node_emb, src_pos, dst_pos, src_neg, dst_neg, tau: float):
    """
    Multi-positive InfoNCE:
      L_i = -log( sum_{p in P(i)} exp(tau * sim(i,p)) / (sum_{p in P(i)} exp(...) + sum_{n in N(i)} exp(...)) )
    """
    device = node_emb.device
    N = node_emb.size(0)
    eps = 1e-8

    z = Fnn.normalize(node_emb, dim=1)

    if src_pos.numel() == 0:
        return torch.tensor(0.0, device=device)

    s_pos = tau * (z[src_pos] * z[dst_pos]).sum(dim=1)
    exp_pos = torch.exp(s_pos)

    sum_pos = torch.zeros(N, device=device, dtype=exp_pos.dtype)
    sum_pos.index_add_(0, src_pos, exp_pos)

    if src_neg.numel() > 0:
        s_neg = tau * (z[src_neg] * z[dst_neg]).sum(dim=1)
        exp_neg = torch.exp(s_neg)
        sum_neg = torch.zeros(N, device=device, dtype=exp_neg.dtype)
        sum_neg.index_add_(0, src_neg, exp_neg)
    else:
        sum_neg = torch.zeros(N, device=device, dtype=exp_pos.dtype)

    denom = sum_pos + sum_neg + eps
    valid = sum_pos > 0
    loss = (torch.log(denom[valid]) - torch.log(sum_pos[valid] + eps)).mean()
    return loss


def _sample_pairs_hhop(subgraph, hop: int, neg_ratio: float):
    """
    case2 (pseudo-label-aware):
    positives:
      - from <=hop neighborhood
      - labeled/pseudo-labeled anchor: prefer same-label positives; if none, fall back to all <=hop neighbors
      - unlabeled anchor: take all <=hop neighbors
    negatives:
      - from outside <=hop neighborhood
      - count = (#pos for anchor) * neg_ratio
      - labeled/pseudo anchor: first try different-label & labeled negatives; if not found after retries, relax to random outside neighborhood
      - unlabeled anchor: random outside neighborhood
    """
    _ensure_hhop_cache(subgraph, hop)
    cache = subgraph._hhop_cache[hop]
    src_all = cache["src_all"]
    dst_all = cache["dst_all"]
    hhop_hash = cache["hash"]
    N = cache["N"]
    device = src_all.device

    # -------- positives --------
    has_label, labels = _get_label_state(subgraph)
    has_src = has_label[src_all]
    has_dst = has_label[dst_all]
    same = has_src & has_dst & (labels[src_all] == labels[dst_all])

    # fallback for labeled anchors with no same-label positives
    cnt_same = torch.bincount(src_all[same], minlength=N)
    no_same = has_label & (cnt_same == 0)

    keep = (~has_src) | same | no_same[src_all]
    src_pos = src_all[keep]
    dst_pos = dst_all[keep]

    # -------- negatives --------
    neg_mult = max(1, int(neg_ratio))
    if src_pos.numel() == 0:
        return src_pos, dst_pos, torch.empty((0,), device=device, dtype=torch.long), torch.empty((0,), device=device, dtype=torch.long)

    pos_cnt = torch.bincount(src_pos, minlength=N)
    neg_cnt = pos_cnt * neg_mult
    anchors = torch.arange(N, device=device, dtype=torch.long)
    mask = neg_cnt > 0

    if mask.sum().item() == 0:
        src_neg = torch.empty((0,), device=device, dtype=torch.long)
        dst_neg = torch.empty((0,), device=device, dtype=torch.long)
        return src_pos, dst_pos, src_neg, dst_neg

    src_neg = torch.repeat_interleave(anchors[mask], neg_cnt[mask])
    dst_neg = torch.randint(0, N, (src_neg.numel(),), device=device, dtype=torch.long)

    # stage 1: enforce outside-hhop + not-self + (for labeled anchors) prefer labeled & different label
    for _ in range(3):
        invalid = (dst_neg == src_neg) | _pair_in_hhop(src_neg, dst_neg, hhop_hash, N)
        src_labeled = has_label[src_neg]
        invalid = invalid | (src_labeled & (~has_label[dst_neg]))
        invalid = invalid | (src_labeled & (labels[dst_neg] == labels[src_neg]))
        if not invalid.any():
            break
        dst_neg[invalid] = torch.randint(0, N, (int(invalid.sum().item()),), device=device, dtype=torch.long)

    # stage 2: relax label constraint (still outside-hhop + not-self)
    for _ in range(3):
        invalid2 = (dst_neg == src_neg) | _pair_in_hhop(src_neg, dst_neg, hhop_hash, N)
        if not invalid2.any():
            break
        dst_neg[invalid2] = torch.randint(0, N, (int(invalid2.sum().item()),), device=device, dtype=torch.long)

    # stage 3: last resort (only not-self) to avoid pathological infinite resampling in tiny graphs
    invalid3 = (dst_neg == src_neg)
    if invalid3.any():
        dst_neg[invalid3] = (dst_neg[invalid3] + 1) % N

    return src_pos, dst_pos, src_neg, dst_neg


def Ncontrast(node_emb: torch.Tensor, subgraph, hop: int, tau: float = 1.0, neg_ratio: float = 5.0) -> torch.Tensor:
    src_pos, dst_pos, src_neg, dst_neg = _sample_pairs_hhop(
        subgraph=subgraph,
        hop=hop,
        neg_ratio=neg_ratio,
    )
    return _contrastive_multi_pos(node_emb, src_pos, dst_pos, src_neg, dst_neg, tau=tau)
# ==============================================================================================
 
USE_SIL = args.dataset in { "Cora","CiteSeer", "PubMed"}
USE_RANKME = args.dataset in { "CS", "Physics", "ogbn-arxiv"}

num_classes = get_num_classes(args.dataset)

if __name__ == "__main__":
    seed_everything(seed=args.seed)
    dataset = load_dataset(args, args.labeling_ratio, args.num_clients)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{args.gpu_id}")
    subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]
    round_results = []

    for client_id in range(args.num_clients):
        preprocess_adj_matrix(subgraphs[client_id], args.dataset, args.seed, client_id, args.num_clients, 50)

    feature_dim = subgraphs[0].x.size(1)
    num_classes = dataset.num_classes

    local_models = [GCN(feat_dim=subgraphs[client_id].x.shape[1],
                        hid_dim=args.hid_dim,
                        out_dim=dataset.num_classes,
                        dropout=args.dropout).to(device)
                    for client_id in range(args.num_clients)]

    local_optimizers = [Adam(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay) for client_id in range(args.num_clients)]
    global_model = GCN(feat_dim=subgraphs[0].x.shape[1],hid_dim=args.hid_dim,out_dim=dataset.num_classes,dropout=args.dropout).to(device)

    # === CHANGED: we keep best_server_val name but it now tracks best proxy metric, not val acc
    best_server_val = -1e9   # proxy metric (sil or rankme), larger is better
    best_server_test = 0
    no_improvement_count = 0

    for client_id in range(args.num_clients):
        F, max_probs = label_propagation( subgraphs[client_id],args.alpha,args.I,num_classes,args.dataset,args.seed,args.num_clients,client_id,args.tau_lp,pre=True)
        propagated_labels = F.argmax(dim=1)
        train_idx = subgraphs[client_id].train_idx
        non_train_mask = ~train_idx
        confident_mask = (max_probs >= args.tau_lp) & non_train_mask
        idx_with_pseudo = train_idx.clone()
        y_with_pseudo = subgraphs[client_id].y.clone()
        idx_with_pseudo[confident_mask] = True
        y_with_pseudo[confident_mask] = propagated_labels[confident_mask]
        subgraphs[client_id].y_with_pseudo = y_with_pseudo
        subgraphs[client_id].idx_train_with_pseudo = idx_with_pseudo
        mask2d = idx_with_pseudo.unsqueeze(0) & idx_with_pseudo.unsqueeze(1)
        y_eq = (y_with_pseudo.unsqueeze(0) == y_with_pseudo.unsqueeze(1)).float()
        indicate_matrix = torch.zeros(
            (idx_with_pseudo.size(0), idx_with_pseudo.size(0)),
            device=idx_with_pseudo.device
        )
        indicate_matrix[mask2d] = y_eq[mask2d] * 2 - 1
        subgraphs[client_id].indicate_matrix = indicate_matrix

    l_glb_acc_test = []
    cal_class_learning_status_list = [None] * args.num_clients

    ###################################################### training start
    for round_id in range(args.num_rounds):
        global_model.eval()
        # === CHANGED: 'global_acc_val' now holds proxy metric (sil/rankme) instead of val accuracy
        global_proxy_metric = 0.0
        global_acc_test = 0.0

        ####################################################### local train
        for client_id in range(args.num_clients):
            for epoch_id in range(args.num_epochs):
                #### local GNN Update
                local_models[client_id].train()
                local_optimizers[client_id].zero_grad()
                logits, node_emb = local_models[client_id].forward(subgraphs[client_id], return_embed=True)
                adj_label = subgraphs[client_id].adj_label
                ce_loss = loss_fn(logits[subgraphs[client_id].idx_train_with_pseudo],subgraphs[client_id].y_with_pseudo[subgraphs[client_id].idx_train_with_pseudo])
                loss_train = ce_loss
                Ncontrast_loss = Ncontrast(node_emb, subgraphs[client_id], args.hop, tau=args.beta, neg_ratio=args.neg_ratio)
                loss_train += Ncontrast_loss * args.lam
                total_loss = loss_train
                total_loss.backward()
                local_optimizers[client_id].step()

                #### LABEL-AUGMENTATION
                if round_id >= args.R_switch:
                    #### Adaptive Class-wise Pseudo-labeling
                    local_models[client_id].eval()
                    with torch.no_grad():
                        logits = local_models[client_id].forward(subgraphs[client_id])
                        probs = torch.softmax(logits, dim=-1)
                        max_probs, pseudo_labels = torch.max(probs, dim=-1)
                        cal_class_learning_status_list[client_id] = cal_class_learning_status(logits)
                        thresholds = args.tau * cal_class_learning_status_list[client_id].to(pseudo_labels.device)[pseudo_labels]
                        high_conf_mask = max_probs > thresholds
                        high_conf_indices = torch.nonzero(high_conf_mask).squeeze(1)
                        high_conf_labels = pseudo_labels[high_conf_mask]
                        existing_train_mask = torch.zeros_like(max_probs, dtype=torch.bool)
                        existing_train_mask[subgraphs[client_id].train_idx] = True
                        high_conf_mask_filtered = high_conf_mask & ~existing_train_mask
                        high_conf_indices_filtered = torch.nonzero(high_conf_mask_filtered).squeeze(1)
                        high_conf_labels_filtered = pseudo_labels[high_conf_mask_filtered]
                        idx_train_with_pseudo = subgraphs[client_id].train_idx.clone()
                        y_with_pseudo = subgraphs[client_id].y.clone()
                        pseudo_label_mask = torch.zeros_like(idx_train_with_pseudo, dtype=torch.bool)
                        pseudo_label_mask[high_conf_indices_filtered] = True
                        idx_train_with_pseudo[high_conf_indices_filtered] = 1
                        y_with_pseudo[high_conf_indices_filtered] = high_conf_labels_filtered
                        subgraphs[client_id].idx_train_with_pseudo = idx_train_with_pseudo
                        subgraphs[client_id].y_with_pseudo = y_with_pseudo
                        subgraphs[client_id].pseudo_label_mask = pseudo_label_mask

                    #### Label propagation
                    F, max_probs = label_propagation(subgraphs[client_id], args.alpha, args.I, num_classes,args.dataset,args.seed,args.num_clients,client_id,args.tau_lp)
                    propagated_labels = F.argmax(1)
                    non_train_mask = ~subgraphs[client_id].train_idx
                    confident_mask = (max_probs >= args.tau_lp) & non_train_mask
                    proportion_confident = confident_mask.sum().item() / len(confident_mask)
                    propagated_pseudo_labels = propagated_labels[confident_mask]

                    if round_id < args.R_switch:
                        idx_train_with_pseudo = subgraphs[client_id].train_idx.clone()
                        y_with_pseudo = subgraphs[client_id].y.clone()
                        pseudo_label_mask = torch.zeros_like(subgraphs[client_id].train_idx,dtype=torch.bool)
                    else:
                        idx_train_with_pseudo = subgraphs[client_id].idx_train_with_pseudo.clone()
                        y_with_pseudo = subgraphs[client_id].y_with_pseudo.clone()
                        pseudo_label_mask = subgraphs[client_id].pseudo_label_mask.clone()

                    idx_train_with_pseudo[confident_mask] = 1
                    y_with_pseudo[confident_mask] = propagated_labels[confident_mask]
                    pseudo_label_mask[confident_mask] = True
                    mask = idx_train_with_pseudo.unsqueeze(0) & idx_train_with_pseudo.unsqueeze(1)
                    y_equal_matrix = (y_with_pseudo.unsqueeze(0) == y_with_pseudo.unsqueeze(1)).float()
                    confident_mask = torch.nonzero(confident_mask).squeeze(1)
                    subgraphs[client_id].y_with_pseudo = y_with_pseudo

                    #### indicate_matrix update
                    mask = idx_train_with_pseudo.unsqueeze(0) & idx_train_with_pseudo.unsqueeze(1)
                    y_equal_matrix = (y_with_pseudo.unsqueeze(0) == y_with_pseudo.unsqueeze(1)).float()
                    indicate_matrix = torch.zeros((idx_train_with_pseudo.size(0), idx_train_with_pseudo.size(0)),device=idx_train_with_pseudo.device)
                    indicate_matrix[mask] = y_equal_matrix[mask] * 2 - 1
                    subgraphs[client_id].idx_train_with_pseudo = idx_train_with_pseudo
                    subgraphs[client_id].y_with_pseudo = y_with_pseudo
                    subgraphs[client_id].indicate_matrix = indicate_matrix

        # Class-wise-reliability Computation
        ckr = calculate_class_wise_reliability(subgraphs, args.num_clients, dataset.num_classes)
        zero_cols = (ckr.sum(0) == 0).nonzero(as_tuple=True)[0]
        normalized_ckr = ckr / ckr.sum(0)
        normalized_ckr[:, zero_cols] = 0
        if torch.isnan(normalized_ckr).any():
            raise ValueError("Error: normalized_ckr contains NaN values!")
        ckr_similarity_matrix = calculate_cosine_similarity_matrix(normalized_ckr, args.dataset)

        # Class-wise-reliability guided aggregation
        with torch.no_grad():
            aggregated_models = [copy.deepcopy(global_model) for _ in range(args.num_clients)]

            for client_id in range(args.num_clients):
                similarity_weights = ckr_similarity_matrix[client_id]

                for other_client_id in range(args.num_clients):
                    weight = similarity_weights[other_client_id]
                    for (local_state, global_state) in zip(local_models[other_client_id].parameters(),
                                                           aggregated_models[client_id].parameters()):
                        if other_client_id == 0:
                            global_state.data = weight * local_state
                        else:
                            global_state.data += weight * local_state

            for client_id in range(args.num_clients):
                local_models[client_id].load_state_dict(aggregated_models[client_id].state_dict())

 
        per_client_proxy = []
        per_client_weights = []

        for client_id in range(args.num_clients):
            local_models[client_id].eval()
            logits = local_models[client_id].forward(subgraphs[client_id])
            acc_test = accuracy(logits[subgraphs[client_id].test_idx],
                                subgraphs[client_id].y[subgraphs[client_id].test_idx])
            global_acc_test += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_test

            # === ADDED: proxy metric
            Z = get_Z_for_rankme(local_models[client_id], subgraphs[client_id])  # [N, d]
            if USE_SIL:
                _, sil = cluster_and_score_hdbscan(Z)
                metric_value = torch.tensor(sil, device=Z.device, dtype=torch.float32)
            else:
                rm = rankme_effective_rank(Z, center=True)
                metric_value = rm if isinstance(rm, torch.Tensor) else torch.tensor(rm, device=Z.device, dtype=torch.float32)

            per_client_proxy.append(metric_value)
            per_client_weights.append(subgraphs[client_id].x.shape[0])

        # weighted mean of proxy
        weights = torch.tensor(per_client_weights, device=device, dtype=torch.float32)
        proxy_stack = torch.stack(per_client_proxy)
        mask = torch.isfinite(proxy_stack)
        if mask.any():
            global_proxy_metric = (proxy_stack[mask] * (weights[mask] / weights[mask].sum())).sum().item()
        else:
            global_proxy_metric = float('-inf')  # fallback if all NaN

        # === CHANGED: early stopping / best tracking uses proxy metric
        if global_proxy_metric > best_server_val and round_id >5:
            best_server_val = global_proxy_metric
            best_server_test = global_acc_test
            best_round = round_id
            no_improvement_count = 0
            # print("-" * 50)
            tag = "Silhouette" if USE_SIL else "RankMe"
            print(f"[server]: new best round: {best_round}\tbest {tag}: {best_server_val:.4f}   test: {best_server_test:.2f}")
        else:
            no_improvement_count += 1
            tag = "Silhouette" if USE_SIL else "RankMe"
            print(f"Current {tag}: {global_proxy_metric:.4f}  \t  test: {global_acc_test:.2f}")
            if no_improvement_count == 30:
                # print(f" best round: {best_round}\tbest test: {best_server_test:.2f}")
                break

        l_glb_acc_test.append(global_acc_test)


print(f" Method : FedLAG best round: {best_round}\tbest test: {best_server_test:.2f}")

# === keep the result saving unchanged ===
results = {
    'BestGlobalAccTest': best_server_test,
    'best_round': best_round,
    'labeling_ratio': args.labeling_ratio,
    'Method': "FedLAG",
}

results_df = pd.DataFrame([results])
excel_path = f"train_{args.labeling_ratio}_{args.dataset}_clients_{args.num_clients}.xlsx"

if os.path.exists(excel_path):
    existing_df = pd.read_excel(excel_path)
    updated_df = pd.concat([existing_df, results_df], ignore_index=True)
else:
    updated_df = results_df

updated_df.to_excel(excel_path, index=False)
print(f"Results saved to {excel_path}")
