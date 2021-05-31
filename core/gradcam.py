from typing import List, Optional, Tuple, Union

import dgl
import numpy as np
import torch
import torch.nn.functional as F


EPS = 10e-7


def copy_graph(x):
    graph_copy = dgl.DGLGraph(graph_data=x)
    for k, v in x.ndata.items():
        graph_copy.ndata[k] = v.clone()
    for k, v in x.edata.items():
        graph_copy.edata[k] = v.clone()
    return graph_copy


class GradCAM(object):

    def __init__(self, model: torch.nn.Module, conv_layers: List[str]) -> None:
        """
        BaseCAM constructor.
        Args:
            model (torch.nn.Module): Input model.
            conv_layer (List[str]): List of tensor names to compute activations on.
        """

        self.model = model
        self.forward_hook = list()
        self.backward_hook = list()
        self.hook_handles = list()

        # Forward hooks
        for conv_layer in conv_layers:
            if not hasattr(model, conv_layer):
                raise ValueError(
                    f"Unable to find submodule {conv_layers} in the model")
            self.hook_handles.append(
                self.model._modules.get(conv_layer).register_forward_hook(
                    self._set_forward_hook))
        # Backward hooks
        for conv_layer in conv_layers:
            self.hook_handles.append(
                self.model._modules.get(conv_layer).register_backward_hook(
                    self._set_backward_hook))
        # Enable hooks
        self._hooks_enabled = True
        # Should ReLU be used before normalization
        self._relu = True
        # Model output is used by the extractor
        self._score_used = False

    def _set_forward_hook(self, module, input, output):
        """Hook activations (forward hook)."""
        if self._hooks_enabled:
            self.forward_hook.append(output.data)

    def _set_backward_hook(self, module, input, output):
        """Hook gradient (backward hook)."""
        if self._hooks_enabled:
            self.backward_hook.append(output[0].data)

    def clear_hooks(self):
        """Clear model hooks."""
        for handle in self.hook_handles:
            handle.remove()

    @staticmethod
    def _normalize(cams):
        """CAM normalization."""
        cams -= cams.min(0).values
        cams /= cams.max(0).values + EPS
        return cams

    def _get_weights(self, class_idx, scores):
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self._backprop(scores, class_idx)
        grads = torch.stack(list(reversed(self.backward_hook)), dim=2)
        return grads.mean(axis=0)

    def _precheck(self, class_idx, scores):
        """Check for invalid computation cases"""

        # Check that forward has already occurred
        if not self.forward_hook:
            raise AssertionError(
                "Inputs need to be forwarded in the model for the conv features to be hooked"
            )

        # Check class_idx value
        if class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

        # Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError(
                "Model output scores is required to be passed to compute CAMs"
            )

    def __call__(self, class_idx, scores=None, normalized=True):
        """
        Compute CAM for a specified class
        Args:
            class_idx (int): Output class index of the target class whose CAM will be computed.
            scores (torch.Tensor[1, K], optional): Forward output scores of the hooked model, ie logits.
            normalized (bool, optional): Whether the CAM should be normalized.
        Returns:
            torch.Tensor[M, N]: Class activation map of hooked conv layer.
        """

        # Integrity check
        self.backward_hook = list()
        self._precheck(class_idx, scores)

        # Get map weight
        weights = self._get_weights(class_idx, scores)
        is_cuda = weights.is_cuda

        # Perform the weighted combination to get the CAM
        forwards = torch.stack(self.forward_hook, dim=2)
        num_nodes = forwards.squeeze(0).shape[0]
        batch_cams = (
            weights.unsqueeze(0).repeat(num_nodes, 1, 1) * forwards.squeeze(0)
        ).sum(dim=1)

        if is_cuda:
            batch_cams = batch_cams.cuda()

        if self._relu:
            batch_cams = F.relu(batch_cams, inplace=True)

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        # Average out the different weights of the layers
        batch_cams = batch_cams.mean(dim=1)

        return batch_cams

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def _backprop(self, scores, class_idx):
        """Backpropagate the loss for a specific output class"""

        if self.forward_hook is None:
            raise TypeError(
                "Apply forward path before calling backward hook."
            )

        # Backpropagate to get the gradients on the hooked layer
        loss = scores[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)


class GraphGradCAMExplainer:
    def __init__(
        self,
        model, 
        gnn_layer_name: List[str] = None,
        gnn_layer_ids: List[str] = None,
        **kwargs
    ) -> None:
        """
        GraphGradCAMExplainer explainer constructor.
        Args:
            gnn_layer_name (List[str]): List of reference layers to use for computing CAM
                                        Default to None. If None tries to automatically infer
                                        from the model.
            gnn_layer_ids: (List[str]): List of reference layer IDs to use for computing CAM
                                        Default to None. If None tries to automatically infer
                                        from the model.
        """

        # look for GPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

        # set model
        self.model = model
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.zero_grad()

        if gnn_layer_name is None and gnn_layer_ids is None:
            all_param_names = [
                name for name,
                _ in self.model.named_parameters()]
            self.gnn_layer_ids = list(filter(lambda x: x.isdigit(), set(
                [p.split(".")[2] for p in all_param_names])))
            self.gnn_layer_name = all_param_names[0].split(".")[0]
        else:
            self.gnn_layer_ids = gnn_layer_ids
            self.gnn_layer_name = gnn_layer_name

        assert self.gnn_layer_ids is not None
        assert self.gnn_layer_name is not None

    def process(
        self,
        graph: dgl.DGLGraph,
        node_idx
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node importances for a single class
        Args:
            graph (dgl.DGLGraph): Graph to explain.
            class_idx (Union[None, int, List[int]]): Class indices (index) to explain. If None results in using the winning class.
                                                     If a list is provided, explainer all the class indices provided.
                                                     Defaults to None.
        Returns:
            node_importance (np.ndarray): Node-level importance scores.
            logits (np.ndarray): Prediction logits.
        """

        # 1. get winning class 
        self.extractor = self._get_extractor()
        graph_copy = copy_graph(graph)
        original_logits = self.model(graph_copy, node_idx)
        class_idx = original_logits.argmax().item()

        # 2. explain prediction w/ winning class 
        node_importance = self.extractor(class_idx, original_logits, normalized=True).cpu()
        self.extractor.clear_hooks()
        logits = original_logits.cpu().detach().numpy()
        node_importance = node_importance.cpu().detach().squeeze(dim=0).numpy()
        return node_importance, logits

    def _get_extractor(self):
        return GradCAM(
            getattr(self.model, self.gnn_layer_name).layers, self.gnn_layer_ids
        )

