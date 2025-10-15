import argparse
import os
import warnings
import copy
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.optim import Adam
import hdbscan
from hdbscan.validity import validity_index
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
from util.task_util import (
    cal_topo_emb,
    accuracy,
    construct_graph,
    DiversityLoss,
)
from util.base_util import (
    seed_everything,
    load_dataset,
    get_num_classes,
)
from model import GCN, FedTAD_ConGenerator

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()

# experimental environment setup
parser.add_argument('--add_gaussian_noise', type=float, default=0)
 
parser.add_argument('--seed', type=int, default=8044)
parser.add_argument('--root', type=str, default='data/root')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default="CiteSeer")
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

parser.add_argument('--glb_epochs', type=int, default=1)
parser.add_argument('--it_g', type=int, default=1)
parser.add_argument('--it_d', type=int, default=1)
parser.add_argument('--lr_g', type=float, default=3e-2)
parser.add_argument('--lr_d', type=float, default=3e-2)
parser.add_argument('--topk', type=float, default=5)
parser.add_argument('--fedtad_mode', type=str, default='raw_distill', choices=['raw_distill', 'rep_distill'])
parser.add_argument('--num_gen', type=int, default=80)
parser.add_argument('--lam1', type=float, default=2)
parser.add_argument('--lam2', type=float, default=1)
parser.add_argument('--labeling_ratio', type=float, default=0.01)
args = parser.parse_args()

def get_Z_for_rankme(model, data):
    """
    Obtain node representations for proxy metrics.
    We call model.forward(data, return_embed=True) so that it returns (logits, x_dis, x_hid),
    and we use x_hid (N x d) as Z. If not available, we fallback to logits.
    """
    model.eval()
    with torch.no_grad():
        out = model.forward(data, return_embed=True)  # (logits, x_dis, x_hid)
        if isinstance(out, tuple) and len(out) == 3:
            logits, _, x_hid = out
            Z = x_hid
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
 
USE_SIL = args.dataset in {"Cora", "CiteSeer", "PubMed"}
USE_RANKME = args.dataset in {"CS", "Physics", "ogbn-arxiv"}

num_classes = get_num_classes(args.dataset)


if __name__ == "__main__":
    seed_everything(seed=args.seed)
    dataset = load_dataset(args, args.labeling_ratio, args.num_clients)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{args.gpu_id}")
    subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]
    round_results = []
    feature_dim = subgraphs[0].x.size(1)
    num_classes = dataset.num_classes

    local_models = [GCN(feat_dim=subgraphs[client_id].x.shape[1],
                        hid_dim=args.hid_dim,
                        out_dim=dataset.num_classes,
                        dropout=args.dropout).to(device)
                    for client_id in range(args.num_clients)]

    local_optimizers = [Adam(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay) for
                        client_id in range(args.num_clients)]
    global_model = GCN(feat_dim=subgraphs[0].x.shape[1],
                       hid_dim=args.hid_dim,
                       out_dim=dataset.num_classes,
                       dropout=args.dropout).to(device)
    global_optimizer = Adam(global_model.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)
    generator = FedTAD_ConGenerator(noise_dim=64, feat_dim=args.hid_dim if args.fedtad_mode == 'rep_distill' else
    subgraphs[0].x.shape[1], out_dim=dataset.num_classes, dropout=0).to(device)
    global_optimizer = Adam(global_model.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)
    generator_optimizer = Adam(generator.parameters(), lr=args.lr_g, weight_decay=args.weight_decay)

    best_server_val = 0
    best_server_test = 0
    no_improvement_count = 0
    ft_emb_list = []


    
    ckr = torch.zeros((args.num_clients, dataset.num_classes)).to(device)
    for client_id in range(args.num_clients):
        data = subgraphs[client_id]

        graph_emb = cal_topo_emb(edge_index=data.edge_index, num_nodes=data.x.shape[0], max_walk_length=5).to(device)
        ft_emb = torch.cat((data.x, graph_emb), dim=1).to(device)

        ft_emb_list.append(ft_emb)
        subgraphs[client_id].ft_emb = ft_emb

        if len(data.train_idx.nonzero()) == 1:
            for train_i in data.train_idx.nonzero().flatten():
                neighbor = data.edge_index[1, :][data.edge_index[0, :] == train_i]
                node_all = 0
                for neighbor_j in neighbor:
                    node_kr = torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
                    node_all += node_kr
                node_all += 1
                node_all /= (neighbor.shape[0] + 1)

                label = data.y[train_i]
                ckr[client_id, label] += node_all

                std = torch.clamp(args.add_gaussian_noise * ckr, min=1e-6)
                noise = torch.normal(mean=torch.zeros_like(ckr), std=std)
                ckr += noise
        else:
            for train_i in data.train_idx.nonzero().squeeze():
                neighbor = data.edge_index[1, :][data.edge_index[0, :] == train_i]
                node_all = 0
                for neighbor_j in neighbor:
                    node_kr = torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
                    node_all += node_kr
                node_all += 1
                node_all /= (neighbor.shape[0] + 1)

                label = data.y[train_i]
                ckr[client_id, label] += node_all

                std = torch.clamp(args.add_gaussian_noise * ckr, min=1e-6)
                noise = torch.normal(mean=torch.zeros_like(ckr), std=std)
                ckr += noise



    normalized_ckr = ckr / ckr.sum(0)


    l_glb_acc_test = []
    cal_class_learning_status_list = [None] * args.num_clients

    ###################################################### traing start
    for round_id in range(args.num_rounds):
        global_model.eval()
        # global eval
        global_acc_val = 0
        global_acc_test = 0

        ####################################################### local train
        for client_id in range(args.num_clients):
            ####################################################### epoch start
            ####################################################### forwad and backward
            for epoch_id in range(args.num_epochs):
                local_models[client_id].train()
                local_optimizers[client_id].zero_grad()
                logits, _ = local_models[client_id].forward(subgraphs[client_id])
                loss_train = loss_fn(logits[subgraphs[client_id].train_idx],subgraphs[client_id].y[subgraphs[client_id].train_idx])
                loss_train.backward()
                local_optimizers[client_id].step()


        # global aggregation

        with torch.no_grad():
            for client_id in range(args.num_clients):
                weight = subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0]

                for (local_state, global_state) in zip(local_models[client_id].parameters(),
                                                       global_model.parameters()):
                    if client_id == 0:
                        global_state.data = weight * local_state
                    else:
                        global_state.data += weight * local_state


        num_gen = args.num_gen
        c_cnt = [0] * dataset.num_classes
        for class_i in range(dataset.num_classes):
            c_cnt[class_i] = int(num_gen * 1 / dataset.num_classes)
        c_cnt[-1] += num_gen - sum(c_cnt)

        # print(f"pseudo label distribution: {c_cnt}")
        c = torch.zeros(num_gen).to(device).long()
        ptr = 0
        for class_i in range(dataset.num_classes):
            for _ in range(c_cnt[class_i]):
                c[ptr] = class_i
                ptr += 1

        each_class_idx = {}
        for class_i in range(dataset.num_classes):
            each_class_idx[class_i] = c == class_i
            each_class_idx[class_i] = each_class_idx[class_i].to(device)

        for client_id in range(args.num_clients):
            local_models[client_id].eval()

        for _ in range(args.glb_epochs):

            ############ sampling noise ##############
            z = torch.randn((num_gen, 64)).to(device)

            ############ train generator ##############
            generator.train()
            global_model.eval()
            for it_g in range(args.it_g):
                loss_sem = 0
                loss_diverg = 0
                loss_div = 0

                generator_optimizer.zero_grad()
                for client_id in range(args.num_clients):
                    ######  generator forward  ########
                    node_logits = generator.forward(z=z, c=c)
                    node_norm = Fnn.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=args.topk)

                    ##### local & global model -> forward #########
                    if args.fedtad_mode == 'rep_distill':
                        local_pred, _ = local_models[client_id].rep_forward(
                            data=pseudo_graph)
                        global_pred, _ = global_model.rep_forward(
                            data=pseudo_graph)
                    else:
                        local_pred = local_models[client_id].forward(
                            data=pseudo_graph)
                        global_pred = global_model.forward(
                            data=pseudo_graph)

                    ##########  semantic loss  #############
                    acc_list = [0] * dataset.num_classes
                    for class_i in range(dataset.num_classes):
                        loss_sem += normalized_ckr[client_id][class_i] * nn.CrossEntropyLoss()(
                            local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
                        acc = accuracy(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
                        acc_list[class_i] = f"{acc:.2f}"
                    acc_tot = float(accuracy(local_pred, c))
                    # print(f"[client {client_id}] accuracy on each class for pseudo_graph: {acc_list}, on all classes: {acc_tot:.2f}")

                    ############  diversity loss  ##############
                    loss_div += DiversityLoss(metric='l1').to(device)(z.view(z.shape[0], -1), node_logits)

                    ############  divergence loss  ############
                    for class_i in range(dataset.num_classes):
                        loss_diverg += - normalized_ckr[client_id][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_idx[class_i]] - local_pred[
                                each_class_idx[class_i]].detach()), dim=1))

                ############ generator loss #############
                loss_G = args.lam1 * loss_sem + loss_diverg + args.lam2 * loss_div
                loss_G.backward()
                generator_optimizer.step()

                ########### train global model ###########
                generator.eval()
                global_model.train()

                ######  generator forward  ########
                node_logits = generator.forward(z=z, c=c)
                node_norm = Fnn.normalize(node_logits, p=2, dim=1)
                adj_logits = torch.mm(node_norm, node_norm.t())
                pseudo_graph = construct_graph(node_logits.detach(), adj_logits.detach(), k=args.topk)

                for it_d in range(args.it_d):
                    global_optimizer.zero_grad()
                    loss_D = 0

                    for client_id in range(args.num_clients):
                        #######  local & global model -> forward  #######
                        if args.fedtad_mode == 'rep_distill':
                            local_pred, _ = local_models[client_id].rep_forward(
                                data=pseudo_graph)
                            global_pred, _ = global_model.rep_forward(
                                data=pseudo_graph)
                        else:
                            local_pred = local_models[client_id].forward(
                                data=pseudo_graph)
                            global_pred, _ = global_model.forward(
                                data=pseudo_graph)

                        ############  divergence loss  ############
                        for class_i in range(dataset.num_classes):
                            loss_D += normalized_ckr[client_id][class_i] * torch.mean(torch.mean(
                                torch.abs(global_pred[each_class_idx[class_i]]
                                          - local_pred[each_class_idx[class_i]].detach()), dim=1))

                    loss_D.backward()
                    global_optimizer.step()

        # global model broadcast
        for client_id in range(args.num_clients):
            local_models[client_id].load_state_dict(global_model.state_dict())
        

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
        if global_proxy_metric > best_server_val:
            best_server_val = global_proxy_metric
            best_server_test = global_acc_test
            best_round = round_id
            no_improvement_count = 0
            # print("-" * 50)
            tag = "Silhouette" if USE_SIL else "RankMe"
            # print(f"[server]: new best round: {best_round}\tbest {tag}: {best_server_val:.4f}   test: {best_server_test:.2f}")
        else:
            no_improvement_count += 1
            tag = "Silhouette" if USE_SIL else "RankMe"
            # print(f"Current {tag}: {global_proxy_metric:.4f}  \t  test: {global_acc_test:.2f}")
            if no_improvement_count == 30:
                # print(f" best round: {best_round}\tbest test: {best_server_test:.2f}")
                break

        l_glb_acc_test.append(global_acc_test)

print(f" Method : FedTAD, best round: {best_round}\tbest test: {best_server_test:.2f}")

results = {
    'Seed': args.seed,
    'BestGlobalAccTest': best_server_test,
    'Method': "FedTAD",

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








