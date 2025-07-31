import argparse
import os
import warnings
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
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

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()

# experimental environment setup
parser.add_argument('--add_gaussian_noise', type=float, default=0)
parser.add_argument('--hop', type=int, default=3) # r
parser.add_argument('--seed', type=int, default=4621)
parser.add_argument('--root', type=str, default='/home/ai2/work/fedtad/dataset')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default="CiteSeer") # Cora, PubMed
parser.add_argument('--partition', type=str, default="Louvain")
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--num_rounds', type=int, default=100)
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
parser.add_argument('--I', type=int, default=1) # Must bigger than 1
parser.add_argument('--tau_lp', type=float, default=0.5)
parser.add_argument('--R_switch', type=int, default=5)
parser.add_argument('--labeling_ratio', type=float, default=0.01)
args = parser.parse_args()



num_classes = get_num_classes(args.dataset)

if __name__ == "__main__":
    seed_everything(seed=args.seed)
    dataset = load_dataset(args, args.labeling_ratio, args.num_clients)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{args.gpu_id}")
    subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]
    round_results = []

    for client_id in range(args.num_clients):
        preprocess_adj_matrix(subgraphs[client_id], args.dataset, args.seed, client_id,args.num_clients, 50)

    feature_dim = subgraphs[0].x.size(1)
    num_classes = dataset.num_classes

    local_models = [GCN(feat_dim=subgraphs[client_id].x.shape[1],
                        hid_dim=args.hid_dim,
                        out_dim=dataset.num_classes,
                        dropout=args.dropout).to(device)
                    for client_id in range(args.num_clients)]

    local_optimizers = [Adam(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay) for client_id in range(args.num_clients)]
    global_model = GCN(feat_dim=subgraphs[0].x.shape[1],hid_dim=args.hid_dim,out_dim=dataset.num_classes,dropout=args.dropout).to(device)

    best_server_val = 0
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

    ###################################################### traing start
    for round_id in range(args.num_rounds):
        global_model.eval()
        global_acc_val = 0
        global_acc_test = 0

        ####################################################### local train
        for client_id in range(args.num_clients):
            ####################################################### epoch start
            ####################################################### forwad and backward
            for epoch_id in range(args.num_epochs):
                #### local GNN Update
                local_models[client_id].train()
                local_optimizers[client_id].zero_grad()
                logits, x_dis = local_models[client_id].forward(subgraphs[client_id])
                adj_label = subgraphs[client_id].adj_label
                subgraphs[client_id].x_dis = x_dis
                ce_loss = loss_fn(logits[subgraphs[client_id].idx_train_with_pseudo],subgraphs[client_id].y_with_pseudo[subgraphs[client_id].idx_train_with_pseudo])
                loss_train = ce_loss
                indicate_matrix = subgraphs[client_id].indicate_matrix
                PL_Ncontrast_loss = PL_Ncontrast(x_dis, indicate_matrix,  args.hop, args.beta,args.dataset,args.seed,args.num_clients,client_id)
                loss_train += PL_Ncontrast_loss * args.lam
                total_loss = loss_train
                total_loss.backward()
                local_optimizers[client_id].step()

                #### LABEL-AUGMENTATION
                if round_id >= args.R_switch:
                    #### Adaptive Class-wise Pseudo-labeling
                    local_models[client_id].eval()
                    with torch.no_grad():
                        logits, x_dis = local_models[client_id].forward(subgraphs[client_id], return_x_dis=True)
                        subgraphs[client_id].x_dis = x_dis
                        probs = torch.softmax(logits, dim=-1)  # Calculate probabilities for all nodes
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
                        pseudo_label_mask = pseudo_label_mask = torch.zeros_like(subgraphs[client_id].train_idx,dtype=torch.bool)
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

                    #### indicate_matrix for positive/nagative sampling in contrastive learning
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

        for client_id in range(args.num_clients):
            local_models[client_id].eval()
            logits = local_models[client_id].forward(subgraphs[client_id])
            test_idx = subgraphs[client_id].test_idx
            val_idx = subgraphs[client_id].val_idx
            acc_test = accuracy(logits[subgraphs[client_id].test_idx],
                                subgraphs[client_id].y[subgraphs[client_id].test_idx])
            acc_val = accuracy(logits[subgraphs[client_id].val_idx],
                               subgraphs[client_id].y[subgraphs[client_id].val_idx])
            global_acc_test += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_test
            global_acc_val += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_val

        if global_acc_val > best_server_val:
            best_server_val = global_acc_val
            best_server_test = global_acc_test
            best_round = round_id
            no_improvement_count = 0
            print("-" * 50)
            print(f"[server]: new best round: {best_round}\tbest val acc: {best_server_val}   test: {best_server_test:.2f}")
        else:
            no_improvement_count += 1
            print(f"Current: {global_acc_val}  \t  test: {global_acc_test:.2f}")
            if no_improvement_count == 30:
                print(f" best round: {best_round}\tbest test: {best_server_test:.2f}")
                break

        l_glb_acc_test.append(global_acc_test)

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








