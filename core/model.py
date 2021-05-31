import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ReLU, Tanh, Sigmoid, ELU, LeakyReLU, PReLU, Sequential, Linear, Embedding
import dgl
import numpy as np
import importlib


ACTIVATIONS = {
    'relu': ReLU(),
    'tanh': Tanh(),
    'sigmoid': Sigmoid(),
    'elu': ELU(),
    'PReLU': PReLU(),
    'leaky_relu': LeakyReLU()
}


GNN_MSG = 'gnn_msg'
GNN_NODE_FEAT_IN = 'feats'
GNN_NODE_FEAT_OUT = 'gnn_node_feat_out'
GNN_AGG_MSG = 'gnn_agg_msg'

MAX_NODE_DEGREE = 729  # computed offline 

READOUT_TYPES = {
    'sum': dgl.sum_nodes,
    'mean': dgl.mean_nodes,
}


REDUCE_TYPES = {
    'sum': torch.sum,
    'mean': torch.mean,
}

EPS = 1e-5



class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=1,
        act="relu",
        use_bn=False,
        bias=True,
        verbose=False,
    ):
        """
        MLP Constructor
        :param in_dim: (int) input dimension
        :param hidden_dim: (int) hidden dimension(s), if type(h_dim) is int => all the hidden have the same dimensions
                                                 if type(h_dim) is list => hidden use val in list as dimension
        :param out_dim: (int) output_dimension
        :param num_layers: (int) number of layers
        :param act: (str) activation function to use, last layer without activation!
        :param use_bn: (bool) il layers should have batch norm
        :param bias: is Linear should have bias term, if type(h_dim) is bool => all the hidden have bias terms
                                                      if type(h_dim) is list of bool => hidden use val in list as bias
        :param verbose: (bool) verbosity level
        """
        super(MLP, self).__init__()

        # optional argument
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # set activations
        self._set_activations(act)

        # set mlp dimensions
        self._set_mlp_dimensions(in_dim, hidden_dim, out_dim, num_layers)

        # set batch norm
        self._set_batch_norm(use_bn, num_layers)

        # set bias terms
        self._set_biases(bias, num_layers)

        # build MLP layers
        self.mlp = nn.ModuleList()
        if num_layers == 1:
            self.mlp = self._build_layer(0, act=False)
        elif num_layers > 1:
            # append hidden layers
            for layer_id in range(num_layers):
                self.mlp.append(
                    self._build_layer(
                        layer_id, act=layer_id != (
                            num_layers - 1)))
        else:
            raise ValueError('The number of layers must be greater than 1.')

        if verbose:
            for layer_id, layer in enumerate(self.mlp):
                print('MLP layer {} has params {}'.format(layer_id, layer))

    def _build_layer(self, layer_id, act=True):
        """
        Build layer
        :param layer_id: (int)
        :return: layer (Sequential)
        """

        layer = Sequential()
        layer.add_module("fc",
                         Linear(self.dims[layer_id],
                                self.dims[layer_id + 1],
                                bias=self.bias[layer_id]))
        if self.use_bn[0]:
            bn = nn.BatchNorm1d(self.dims[layer_id + 1])
            layer.add_module("bn", bn)

        if act:
            layer.add_module(self.act, self.activation)
        return layer

    def _set_biases(self, bias, num_layers):
        """
        Set and control bias input
        """
        if isinstance(bias, bool):
            self.bias = num_layers * [bias]
        elif isinstance(bias, list):
            assert len(
                bias
            ) == num_layers, "Length of bias should match the number of layers."
            self.bias = bias
        else:
            raise ValueError(
                "Unsupported type for bias. Needs to be of type bool or list.")

    def _set_batch_norm(self, use_bn, num_layers):
        """
        Set and control batch norm param
        """
        if isinstance(use_bn, bool):
            self.use_bn = num_layers * [use_bn]
        else:
            raise ValueError(
                "Unsupported type for batch norm. Needs to be of type bool.")

    def _set_mlp_dimensions(self, in_dim, h_dim, out_dim, num_layers):
        """
        Set and control mlp dimensions
        """
        if isinstance(h_dim, int):
            self.dims = [in_dim] + (num_layers - 1) * [h_dim] + [out_dim]
        elif isinstance(h_dim, list):
            assert len(h_dim) == (
                num_layers - 1
            ), "Length of h_dim should match the number of hidden layers."
            self.dims = [in_dim] + h_dim + [out_dim]
        else:
            raise ValueError(
                "Unsupported type for h_dim. Needs to be int or list."
            )

    def _set_activations(self, act):
        """
        Set and control activations
        """
        if act in ACTIVATIONS.keys():
            self.act = act
            self.activation = ACTIVATIONS[act]
        else:
            raise ValueError(
                'Unsupported type of activation function: {}. Choose among {}'.
                format(act, list(ACTIVATIONS.keys()))
            )

    def forward(self, feats):
        """
        MLP forward
        :param feats: (FloatTensor) features to pass through MLP
        :return: out: MLP output
        """
        out = feats
        for layer in self.mlp:
            out = layer(out)
        return out


class GINLayer(nn.Module):

    def __init__(
            self,
            node_dim: int,
            out_dim: int,
            act: str = 'relu',
            agg_type: str = 'mean',
            hidden_dim: int = 32,
            batch_norm: bool = True,
            verbose: bool = False) -> None:
        """
        GIN Layer constructor
        Args:
            node_dim (int): Input dimension of each node.
            out_dim (int): Output dimension of each node.
            act (str): Activation function of the update function.
            agg_type (str): Aggregation function. Default to 'mean'.
            hidden_dim (int): Hidden dimension of the GIN MLP. Default to 32.
            batch_norm (bool): If we should use batch normalization. Default to True.
            verbose (bool): Verbosity. Default to False.
        """
        super().__init__()

        if verbose:
            print('Instantiating new GNN layer.')

        self.node_dim = node_dim
        self.out_dim = out_dim
        self.act = act
        self.agg_type = agg_type
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.mlp = MLP(
            node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            verbose=verbose,
        )

    def reduce_fn(self, nodes):
        """
        For each node, aggregate the nodes using a reduce function.
        Current supported functions are sum and mean.
        """
        accum = REDUCE_TYPES[self.agg_type](
            (nodes.mailbox[GNN_MSG]), dim=1)
        return {GNN_AGG_MSG: accum}

    def msg_fn(self, edges):
        """
        Message of each node
        """
        msg = edges.src[GNN_NODE_FEAT_IN]
        return {GNN_MSG: msg}

    def node_update_fn(self, nodes):
        """
        Node update function
        """
        h = nodes.data[GNN_NODE_FEAT_IN]
        h = self.mlp(h)
        h = F.relu(h)
        return {GNN_NODE_FEAT_OUT: h}

    def forward(self, g, h):
        """
        Forward-pass of a GIN layer.
        :param g: (DGLGraph) graph to process.
        :param h: (FloatTensor) node features
        """

        g.ndata[GNN_NODE_FEAT_IN] = h

        g.update_all(self.msg_fn, self.reduce_fn)

        if GNN_AGG_MSG in g.ndata.keys():
            g.ndata[GNN_NODE_FEAT_IN] = g.ndata[GNN_AGG_MSG] + \
                g.ndata[GNN_NODE_FEAT_IN]
        else:
            g.ndata[GNN_NODE_FEAT_IN] = g.ndata[GNN_NODE_FEAT_IN]

        g.apply_nodes(func=self.node_update_fn)

        # apply graph norm and batch norm
        h = g.ndata[GNN_NODE_FEAT_OUT]
        if self.batch_norm:
            h = self.batchnorm_h(h)

        return h



class MultiLayerGNN(nn.Module):
    """
    MultiLayer network that concatenates several gnn layers.
    """

    def __init__(
        self,
        input_dim=None,
        output_dim=32,
        num_layers=3,
        concat=True,
        **kwargs
    ) -> None:
        """
        MultiLayer GNN constructor.
        Args:
            layer_type (str): GNN layer type. Default to "gin_layer".
            input_dim (int): Input dimension of the node features. Default to None.
            output_dim (int): Output dimension of the node embeddings. Default to 32.
            num_layers (int): Number of GNN layers. Default to 3.
        """

        assert input_dim is not None, "Please provide input node dimensions."

        super(MultiLayerGNN, self).__init__()

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.concat = concat

        # input layer
        self.layers.append(
            GINLayer(
                node_dim=input_dim,
                out_dim=output_dim,
                **kwargs
            )
        )
        # hidden layers
        for i in range(1, num_layers - 1):
            self.layers.append(
                GINLayer(
                    node_dim=output_dim,
                    out_dim=output_dim,
                    **kwargs
                )
            )
        # output layer
        self.layers.append(
            GINLayer(
                node_dim=output_dim,
                out_dim=output_dim,
                **kwargs
            )
        )

    def forward(self, g, h):
        """
        Forward pass.
        :param g: (DGLGraph)
        :param h: (FloatTensor)
        """
        h_concat = []
        for layer in self.layers:
            h = layer(g, h)
            h_concat.append(h)

        if self.concat:
            concat_emb = torch.cat(h_concat, dim=-1)
            concat_emb = torch.reshape(concat_emb, (concat_emb.shape[0], -1))  # #nodes x (#layersx#node_dim)
            g.ndata[GNN_NODE_FEAT_OUT] = concat_emb
        else:
            g.ndata[GNN_NODE_FEAT_OUT] = h 
        return g


class BotDetector(nn.Module):
    """
    BotDetector network.
    """

    def __init__(
        self,   
        input_dim,
        concat=True, 
        topology_only=False,
        **kwargs
    ) -> None:
        """
        Bot Detector constructor.
        """
        super(BotDetector, self).__init__()

        self.topology_only = topology_only
        if self.topology_only:
            input_dim = 16 
            self.node_embeddings = Embedding(MAX_NODE_DEGREE+1, input_dim)

        self.gnn = MultiLayerGNN(input_dim=input_dim, concat=concat, **kwargs)

        self.classifier = MLP(
            in_dim=32*3 if concat else 32,
            hidden_dim=32,
            out_dim=2,
            num_layers=2
        )

    def forward(self, g, node_ids):
        """
        Forward pass
        """

        # if self.topology_only:
        #     g.ndata[GNN_NODE_FEAT_IN] = self.node_embeddings(g.ndata['degree'])
            
        h = g.ndata[GNN_NODE_FEAT_IN]
        g = self.gnn(g, h)
        g_list = dgl.unbatch(g)
        node_embeddings = [g.ndata[GNN_NODE_FEAT_OUT][user_id, :] for g, user_id in zip(g_list, node_ids)]
        node_embeddings = torch.stack(node_embeddings, dim=0)
        logits = self.classifier(node_embeddings)
        return logits

