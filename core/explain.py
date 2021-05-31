"""Explain graphs using the GraphGradCAM explainer."""

import numpy as np
import torch 
import yaml
from tqdm import tqdm  
import glob 
import os
import numpy as np
import argparse
from dgl.data.utils import load_graphs
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
import h5py
import warnings
import mlflow.pytorch

from gradcam import GraphGradCAMExplainer
from dataloader import make_data_loader


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 800


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='path to the graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '--concat',
        help='if concat intermediate representations.',
        action='store_true',
    )
    parser.add_argument(
        '--topology_only',
        help='if train w/ topology only.',
        action='store_true',
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path to where the output explanations are saved.',
        default='../output',
        required=False
    )
    return parser.parse_args()


def check_io(save_path):
    if not os.path.isdir(save_path):
        print('Could not find save path, creating it at: {}'.format(save_path))
        os.mkdir(save_path) 
    if not os.path.isdir(os.path.join(save_path, 'graphgradcam')):
            os.mkdir(os.path.join(save_path, 'graphgradcam')) 


def explain_cell_graphs(explainer, data_path, out_path, in_ram, topology_only):

    dataloader = make_data_loader(
        data_path=os.path.join(data_path),
        batch_size=1,
        load_in_ram=in_ram,
        shuffle=False,
        topology_only=topology_only,
        with_name=True
    )

    all_labels = []
    all_predictions = []
    all_misclassifications = []

    for graphs, node_ids, labels, user_ids in tqdm(dataloader, desc='Testing', unit='batch'):

        # 1.. explain the graph & automatically save it in h5 file
        importance_score, logits = explainer.process(graphs, node_ids)

        # 2. performance analysis
        pred = np.argmax(logits.squeeze(), axis=0)
        all_predictions.append(pred)
        label = labels.item()
        all_labels.append(label)

        # 4. save importance scores 
        out_fname = os.path.join(out_path, user_ids[0] + '.h5')
        with h5py.File(out_fname, "w") as f:
            f.create_dataset(
                "importance_score",
                data=importance_score,
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                "correct",
                data=np.array([label == pred]),
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                "node_ids",
                data=np.array(node_ids),
                compression="gzip",
                compression_opts=9,
            )

    print('Weighted F1 score:', f1_score(np.array(all_labels), np.array(all_predictions)))#, average='weighted'))
    print('Accuracy score:', accuracy_score(np.array(all_labels), np.array(all_predictions)))
    print('Precision score:', precision_score(np.array(all_labels), np.array(all_predictions)))#, average='weighted'))
    print('Recall score:', recall_score(np.array(all_labels), np.array(all_predictions)))#, average='weighted'))

def main(args):

    # 1. check io directories 
    check_io(args.out_path)

    # w/o concat & w/ node features
    model = mlflow.pytorch.load_model('s3://mlflow/af43adbc6c1c402eb5f1bdbb7c790de8/artifacts/model_best_val_weighted_f1_score').to(DEVICE)

    # 3. explain the cell graphs
    explainer = GraphGradCAMExplainer(model=model)

    explain_cell_graphs(
        explainer=explainer,
        data_path=args.data_path,
        out_path=os.path.join(args.out_path, 'graphgradcam'),
        in_ram=args.in_ram,
        topology_only=args.topology_only
    )


if __name__ == "__main__":
    main(args=parse_arguments())
