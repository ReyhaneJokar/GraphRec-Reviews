from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList, Sequential, Linear, ReLU, Sigmoid
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index


class ReFINe_plus(torch.nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, num_layers: int, num_users: int, num_items: int, alpha: Optional[Union[float, Tensor]] = None, **kwargs):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = 600

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.user_encoder = Sequential(
            Linear(num_items, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, embedding_dim))
        self.user_decoder = Sequential(
            Linear(embedding_dim, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, num_items),
            Sigmoid())
        
        self.item_encoder = Sequential(
            Linear(num_users, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, embedding_dim))
        self.item_decoder = Sequential(
            Linear(embedding_dim, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, num_users),
            Sigmoid())

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()
    
        for layer in self.user_encoder:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        for layer in self.user_decoder:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        for layer in self.item_encoder:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        for layer in self.item_decoder:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def get_embedding(self, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(self, edge_index: Adj, edge_label_index: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)

    def autoencoder_forward(self, neg_edge_label_index_ae: Tensor, device: str):
        ae_input_user = torch.zeros((self.num_users, self.num_items), device=device)
        ae_input_user[neg_edge_label_index_ae[0], neg_edge_label_index_ae[1] - self.num_users] = 1

        ae_input_item = torch.zeros((self.num_items, self.num_users), device=device)
        ae_input_item[neg_edge_label_index_ae[1] - self.num_users, neg_edge_label_index_ae[0]] = 1

        user_latent = self.user_encoder(ae_input_user)
        user_reconstructed = self.user_decoder(user_latent)

        item_latent = self.item_encoder(ae_input_item)
        item_reconstructed = self.item_decoder(item_latent)

        return user_latent, item_latent, user_reconstructed, item_reconstructed

    def predict_link(self, edge_index: Adj, edge_label_index: OptTensor = None, edge_weight: OptTensor = None, prob: bool = False) -> Tensor:
        pred = self(edge_index, edge_label_index, edge_weight).sigmoid()
        return pred if prob else pred.round()

    def recommend(self, edge_index: Adj, edge_weight: OptTensor = None, src_index: OptTensor = None, dst_index: OptTensor = None, k: int = 1, sorted: bool = True) -> Tensor:
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1, sorted=sorted).indices

        if dst_index is not None:
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor, **kwargs) -> Tensor:
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def recommendation_loss(self, pos_edge_rank: Tensor, neg_edge_rank: Tensor, node_id: Optional[Tensor] = None, lambda_reg: float = 1e-4, **kwargs) -> Tensor:
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        emb = self.embedding.weight
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

    def compute_ae_loss(self, neg_edge_label_index_ae: Tensor, device: str, lambda_reg: float = 1e-5):
        user_latent, item_latent, user_reconstructed, item_reconstructed = self.autoencoder_forward(neg_edge_label_index_ae, device)

        ae_input_user = torch.zeros((self.num_users, self.num_items), device=device)
        ae_input_user[neg_edge_label_index_ae[0], neg_edge_label_index_ae[1] - self.num_users] = 1

        ae_input_item = torch.zeros((self.num_items, self.num_users), device=device)
        ae_input_item[neg_edge_label_index_ae[1] - self.num_users, neg_edge_label_index_ae[0]] = 1

        user_ae_loss = F.binary_cross_entropy(user_reconstructed, ae_input_user)
        item_ae_loss = F.binary_cross_entropy(item_reconstructed, ae_input_item)

        reg_loss = sum(p.norm(2).pow(2) for p in self.user_encoder.parameters())
        reg_loss += sum(p.norm(2).pow(2) for p in self.user_decoder.parameters())
        reg_loss += sum(p.norm(2).pow(2) for p in self.item_encoder.parameters())
        reg_loss += sum(p.norm(2).pow(2) for p in self.item_decoder.parameters())
        reg_loss = (lambda_reg / 2) * reg_loss

        ae_loss = user_ae_loss + item_ae_loss + reg_loss

        return ae_loss, user_latent, item_latent

    def compute_align_loss(self, edge_index: Adj, user_latent: Tensor, item_latent: Tensor):
        emb = self.get_embedding(edge_index)
        user_emb, item_emb = emb[:self.num_users], emb[self.num_users:]
        align_loss = F.mse_loss(user_emb, user_latent) + F.mse_loss(item_emb, item_latent)
        return align_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')


class BPRLoss(_Loss):
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor, parameters: Tensor = None) -> Tensor:
        log_prob = F.logsigmoid(positives - negatives).mean()

        regularization = 0
        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)
            regularization = regularization / positives.size(0)

        return -log_prob + regularization