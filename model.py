from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList, Sequential, Linear, ReLU, Sigmoid
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index

class EdgeAwareLGConv(MessagePassing):
    """
    LightGCN propagation with an additive content term. `edge_weight` is
    the standard structural normalization (D^-1/2 A D^-1/2) and scales the
    neighbor embedding exactly as vanilla LightGCN does. `edge_feat` -- a
    projected vector from the review's text+aspect features -- is ADDED to
    that scaled message, letting content inject new information into the
    message instead of only rescaling existing structural signal (which is
    what the scalar gate did, and which repeatedly collapsed to a constant).
    """
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, edge_feat: OptTensor = None) -> Tensor:
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_feat=edge_feat)

    def message(self, x_j: Tensor, edge_weight: OptTensor, edge_feat: OptTensor) -> Tensor:
        msg = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        if edge_feat is not None:
            msg = msg + edge_feat
        return msg

class ReFINe_plus(torch.nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, num_layers: int, num_users: int, num_items: int, edge_attr_dim: int = 0, alpha: Optional[Union[float, Tensor]] = None, **kwargs):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = 600

        if alpha is None:
            alpha = torch.ones(num_layers + 1, dtype=torch.float)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
            alpha = alpha.float()

        else:
            alpha = torch.tensor([alpha] * (num_layers + 1), dtype=torch.float)
        self.alpha = torch.nn.Parameter(alpha)
        
        self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList([EdgeAwareLGConv() for _ in range(num_layers)])

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
        
        self.edge_attr_proj = None
        self.edge_feat_scale = None
        if edge_attr_dim and edge_attr_dim > 0:
            proj_hidden_dim = embedding_dim * 2
            self.edge_attr_proj = Sequential(
                torch.nn.LayerNorm(edge_attr_dim),
                Linear(edge_attr_dim, proj_hidden_dim),
                torch.nn.LayerNorm(proj_hidden_dim),
                ReLU(),
                torch.nn.Dropout(0.1),
                Linear(proj_hidden_dim, embedding_dim),
            )
            self.edge_feat_scale = torch.nn.Parameter(torch.tensor(0.0))

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
                
        if self.edge_attr_proj is not None:
            for layer in self.edge_attr_proj:
                if isinstance(layer, Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
                        
    def get_embedding(self, edge_index: Adj, edge_weight: OptTensor = None, edge_attr: OptTensor = None) -> Tensor:
        edge_feat = None
        if edge_weight is None:
            edge_index, edge_weight = gcn_norm(
                edge_index, None, self.num_nodes,
                add_self_loops=False, dtype=self.embedding.weight.dtype,
            )
        if edge_attr is not None and self.edge_attr_proj is not None:
            edge_feat = self.compute_edge_feat(edge_attr)       
             
        alpha = torch.softmax(self.alpha, dim=0)
        x = self.embedding.weight
        out = x * alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight, edge_feat=edge_feat)
            out = out + x * alpha[i + 1]

        return out

    def compute_edge_feat(self, edge_attr: Tensor) -> Tensor:
        if self.edge_attr_proj is None:
            return None
        raw = self.edge_attr_proj(edge_attr)
        raw = F.normalize(raw, p=2, dim=-1)
        return self.edge_feat_scale * raw

    def forward(self, edge_index: Adj, edge_label_index: OptTensor = None, edge_weight: OptTensor = None, edge_attr: OptTensor = None) -> Tensor:
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight, edge_attr=edge_attr)

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

    def predict_link(self, edge_index: Adj, edge_label_index: OptTensor = None, edge_weight: OptTensor = None, edge_attr: OptTensor = None, prob: bool = False) -> Tensor:
        pred = self(edge_index, edge_label_index, edge_weight=edge_weight, edge_attr=edge_attr).sigmoid()
        return pred if prob else pred.round()

    def recommend(self, edge_index: Adj, edge_weight: OptTensor = None, edge_attr: OptTensor = None, src_index: OptTensor = None, dst_index: OptTensor = None, k: int = 1, sorted: bool = True) -> Tensor:
        out_src = out_dst = self.get_embedding(edge_index, edge_weight=edge_weight, edge_attr=edge_attr)

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

    def compute_align_loss(self, emb: Tensor, user_latent: Tensor, item_latent: Tensor):
        user_emb, item_emb = emb[:self.num_users], emb[self.num_users:]
        align_loss = F.mse_loss(user_emb, user_latent) + F.mse_loss(item_emb, item_latent)
        return align_loss

    def edge_attr_to_weight(self, edge_attr: Tensor) -> Tensor:
        if self.edge_attr_proj is None:
            return None
        return torch.sigmoid(self.edge_attr_proj(edge_attr)).squeeze(-1)

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