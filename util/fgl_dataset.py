import os
import os.path as osp
import re
import pickle
import time
import warnings
import torch
from torch_geometric.data import Dataset
from util.base_data_util import data_partition
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class FGLDataset(Dataset):
    def __init__(
        self,
        args,
        root,
        name,
        num_clients,
        partition,
        train,
        val,
        test,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        part_delta=20
    ):
        start = time.time()
        self.args = args
        self.root = root
        self.name = name
        self.num_clients = num_clients
        self.partition = partition
        self.train = train
        self.val = val
        self.test = test
        self.part_delta = part_delta

        super(FGLDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.load_data()

        end = time.time()
        print(f"load FGL dataset {name} done ({end-start:.2f} sec)")

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        fmt_name = re.sub("-", "_", self.name)
        return osp.join(
            self.raw_dir, fmt_name, "Client{}".format(
                self.num_clients), self.partition
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        files_names = ["data{}.pt".format(i) for i in range(self.num_clients)]
        return files_names

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, "data{}.pt".format(idx)))
        return data

    def process(self):
        self.load_global_graph()

        if osp.exists(self.processed_dir):
            for file in os.listdir(self.processed_dir):
                os.remove(osp.join(self.processed_dir, file))
        else:
            os.makedirs(self.processed_dir)

        base_filename = f"{self.name}_seed{self.args.seed}_train{self.train}_clients{self.num_clients}"

        subgraph_list, node_dict = data_partition(
            G=self.global_data,
            num_clients=self.num_clients,
            train=self.train,
            val=self.val,
            test=self.test,
            partition=self.partition,
            part_delta=self.part_delta,
            filename=base_filename
        )


        for i in range(self.num_clients):
            data = subgraph_list[i]
            G_nx = to_networkx(data, to_undirected=True)
            num_nodes = len(G_nx.nodes)

            labels = np.array([data.y[node] for node in G_nx.nodes])


            adj_matrix = nx.adjacency_matrix(G_nx)
            adj_2hop = adj_matrix.dot(adj_matrix)
            adj_2hop.setdiag(0)

            adj_label = torch.tensor(adj_matrix.todense()).float()
            client_adj_2hop_no_diag = torch.tensor(adj_2hop.todense()).float()

            if torch.cuda.is_available():
                adj_label = adj_label.cuda()
                client_adj_2hop_no_diag = client_adj_2hop_no_diag.cuda()

            data.adj_label = adj_label
            data.client_adj_2hop_no_diag = client_adj_2hop_no_diag

            torch.save(data, osp.join(self.processed_dir, f"data{i}.pt"))



    

    def load_global_graph(self):
        if self.name in ["Cora", "CiteSeer", "PubMed"]:
            from torch_geometric.datasets import Planetoid
            self.global_dataset = Planetoid(root=self.raw_dir, name=self.name)
        elif self.name in ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]:
            from ogb.nodeproppred import PygNodePropPredDataset
            self.global_dataset = PygNodePropPredDataset(
                root=self.raw_dir, name=self.name
            )
        elif self.name in ["CS", "Physics"]:
            from torch_geometric.datasets import Coauthor
            self.global_dataset = Coauthor(root=self.raw_dir, name=self.name)
        elif self.name in ["Computers", "Photo"]:
            from torch_geometric.datasets import Amazon
            self.global_dataset = Amazon(
                root=self.raw_dir, name=self.name.lower())
        elif self.name in ["NELL"]:
            from torch_geometric.datasets import NELL
            self.global_dataset = NELL(
                root=os.path.join(self.raw_dir, name="NELL"))
        elif self.name in ["Reddit"]:
            from torch_geometric.datasets import Reddit
            self.global_dataset = Reddit(
                root=os.path.join(self.raw_dir, name="Reddit"))
        elif self.name in ["Flickr"]:
            from torch_geometric.datasets import Flickr
            self.global_dataset = Flickr(
                root=os.path.join(self.raw_dir, name="Flickr"))
        else:
            raise ValueError(
                "Not supported for this dataset, please check root file path and dataset name"
            )
        self.global_data = self.global_dataset.data
        self.global_data.num_classes = self.global_dataset.num_classes

    def load_data(self):
        print("loading graph...")
        self.load_global_graph()
        self.feat_dim = self.global_dataset.num_features
        self.out_dim = self.global_dataset.num_classes
        self.global_data = self.global_dataset.data

        # 直接调用 process 方法重新划分并加载数据
        self.process()
        self.subgraphs = [self.get(i) for i in range(self.num_clients)]
        for i in range(len(self.subgraphs)):
            self.subgraphs[i].feat_dim = self.global_dataset.num_features
            self.subgraphs[i].out_dim = self.out_dim
            if self.name in ["ogbn-arxiv", "ogbn-products"]:
                for i in range(self.num_clients):
                    self.subgraphs[i].y = self.subgraphs[i].y.squeeze()
        