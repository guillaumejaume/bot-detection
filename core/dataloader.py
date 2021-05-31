"""Twitter Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import dgl 
from networkx.classes.function import density


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'


def set_graph_on_cuda(graph):
    cuda_graph = dgl.DGLGraph()
    cuda_graph.add_nodes(graph.number_of_nodes())
    cuda_graph.add_edges(graph.edges()[0], graph.edges()[1])
    for key_graph, val_graph in graph.ndata.items():
        tmp = graph.ndata[key_graph].clone()
        cuda_graph.ndata[key_graph] = tmp.cuda()
    for key_graph, val_graph in graph.edata.items():
        cuda_graph.edata[key_graph] = graph.edata[key_graph].clone().cuda()
    return cuda_graph



class TwitterDataset(Dataset):
    """Twitter dataset."""

    def __init__(
            self,
            data_path: str,
            load_in_ram: bool = False,
            topology_only: bool = False,
            with_name: bool = False
    ):
        """
        Twitter dataset constructor.
        Args:
            data_path (str): Graph path to a given split (eg, cell_graphs/test/).
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
            topology_only (bool, optional): Use only the graph topology - if True, 
                                            will initialize the nodes features w/ 
                                            one-hot encoding of the node in-degree 
        """
        super(TwitterDataset, self).__init__()
        self.data_path = data_path
        self.load_in_ram = load_in_ram
        self.topology_only = topology_only
        self.with_name = with_name
        self._load_graphs()
        if topology_only and load_in_ram:
            self._set_node_degree()
        # self._compute_graph_properties()

    def _set_node_degree(self):
        for g in self.graphs:
            g.ndata['degree'] = g.in_degrees()
        # node_degrees = [g.in_degrees() for g in self.graphs]
        # for x in node_degrees:
        #     print('node', x.shape)
        # unique_node_degrees = torch.cat(node_degrees, axis=0)
        # unique_node_degrees = torch.unique(unique_node_degrees)
        # print('Unique:', unique_node_degrees)

    def _compute_graph_properties(self):

        # 1. Number of node and edges in total 
        avg_num_nodes = []
        avg_num_edges = []
        avg_density = []
        for g in self.graphs:
            avg_num_nodes.append(g.number_of_nodes())
            avg_num_edges.append(g.number_of_edges())
            avg_density.append(density(g.to_networkx()))

        print("Number of samples:", len(avg_num_nodes)) 
        print('Average number of nodes:', sum(avg_num_nodes) / len(avg_num_nodes))
        print('Average number of edges:', sum(avg_num_edges) / len(avg_num_edges))
        print('Average density:', sum(avg_density) / len(avg_density))

        def extract_sublist(array, target):
            sub = [val for val, label  in zip(array, self.graph_labels) if label == target]
            return sub

        for label in [0, 1]:
            print("Number of samples of label:", label, sum(np.array(self.graph_labels) == label)) 
            print('Average number of nodes of label:', label, sum(extract_sublist(avg_num_nodes, label)) / sum(np.array(self.graph_labels) == label))
            print('Average number of edges of label:', label, sum(extract_sublist(avg_num_edges, label)) / sum(np.array(self.graph_labels) == label))
            print('Average density of label:', label, sum(extract_sublist(avg_density, label)) / sum(np.array(self.graph_labels) == label))

    @staticmethod
    def get_index(fname, g):
        user_id = int(fname.split('/')[-1].replace('.bin', ''))
        idx = list(g.ndata['user_id'].numpy()).index(user_id)
        return idx

    def _load_graphs(self):
        """
        Load graphs
        """
        self.graph_fnames = glob(os.path.join(self.data_path, '*.bin'))
        self.graph_fnames.sort()
        self.num_graphs = len(self.graph_fnames)
        if self.load_in_ram:
            graphs = [load_graphs(os.path.join(self.data_path, fname)) for fname in self.graph_fnames]
            self.graphs = [entry[0][0] for entry in graphs]
            self.graph_labels = [entry[1]['label'].item() for entry in graphs]
            self.node_ids = [self.get_index(fname, g) for fname, g in zip(self.graph_fnames, self.graphs)]

    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """

        if self.load_in_ram:
            g = self.graphs[index]
            label = self.graph_labels[index]
            idx = self.node_ids[index]
        else:
            g, label = load_graphs(self.graph_fnames[index])
            label = label['label'].item()
            g = g[0]
            if self.topology_only:
                g.ndata['degree'] = g.in_degrees()
            idx = self.get_index(self.graph_fnames[index], g)
        g.ndata['feats'] = g.ndata['feats'].float()
        g = set_graph_on_cuda(g) if IS_CUDA else g

        if self.with_name:
            user_id = self.graph_fnames[index].split('/')[-1].split('.')[0]
            return g, idx, label, user_id

        return g, idx, label

    def __len__(self):
        """Return the number of samples in the BRACS dataset."""
        return self.num_graphs


def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        graphs: (dgl.DGLGraph)
        labels: (torch.LongTensor)
    """
    graphs = dgl.batch([example[0] for example in batch])
    node_ids = [example[1] for example in batch]
    labels = torch.LongTensor([example[2] for example in batch]).to(DEVICE)
    try:
        user_ids = [example[3] for example in batch]
        return graphs, node_ids, labels, user_ids
    except:
        return graphs, node_ids, labels


def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):
    """
    Create a data loader.
    """

    dataset = TwitterDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )

    return dataloader
