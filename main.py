import torch
from model import ReFINe_plus
from torch.nn import MSELoss
from torch_geometric.utils import degree

from tqdm import tqdm
import argparse
import os
import time
import random
import numpy as np
import json
from pathlib import Path

import data_loader, utils

#############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=2024)
parser.add_argument('--gpu_id', type=int, default=2)
parser.add_argument('--dataset', type=str, default='ML-100K')
parser.add_argument('--dataset_augment', type=str, default='augment')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=8192)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--evaluation_step', type=int, default=1)
parser.add_argument('--early_stopping_step', type=int, default=50)
parser.add_argument('--project_dir', type=str, required=True)
parser.add_argument('--top_k', nargs='+', type=int, default=[5, 10, 15, 20])
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--real_neg_samp_prob', type=float, default=1.5, help='real_negative_sampling_probabilities')
parser.add_argument('--path_name', type=str, default='nothing')
args = parser.parse_args()
#############################################################################

seed = args.random_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
os.environ['PYTHONHASHSEED'] = str(seed)

gpu = 'cuda:'+str(args.gpu_id)
device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

print('#############################################################################')
print('ramdom_seed:', args.random_seed)
print('gpu_id:', args.gpu_id)
print('dataset:', args.dataset)
print('dataset_augment:', args.dataset_augment)
print('batch_size:', args.batch_size)
print('test_batch_size:', args.test_batch_size)
print('embedding_dim:', args.embedding_dim)
print('layers:', args.layers)
print('learning_rate:', args.learning_rate)
print('evaluation_step:', args.evaluation_step)
print('early_stopping_step:', args.early_stopping_step)
print('top_k:', args.top_k)
print('epochs:', args.epochs)
print('real_negative_sampling_probabilities:', args.real_neg_samp_prob)
print('path_name:', args.path_name)
print('#############################################################################\n')

print('#############################################################################')
print('data loading...')
if args.dataset_augment == 'original':
    print('train set: train_original')
else:
    print('train set: train_augment')
data, data_neg, data_neutral = data_loader.data_loading(project_dir=args.project_dir, load_val_or_test='val')
num_users, num_items = data['user'].num_nodes, data['item'].num_nodes
data = data.to_homogeneous().to(device)
data_neg = data_neg.to_homogeneous().to(device)
data_neutral = data_neutral.to_homogeneous().to(device)
print('done!')
print('#############################################################################')


batch_size = args.batch_size
test_batch_size = args.test_batch_size

mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, data.edge_index[0] < data.edge_index[1]]
train_loader = torch.utils.data.DataLoader(
    range(train_edge_label_index.size(1)),
    shuffle=True,
    batch_size=batch_size)

mask_neg = data_neg.edge_index[0] < data_neg.edge_index[1]
train_neg_edge_label_index = data_neg.edge_index[:, mask_neg]

mask_neutral = data_neutral.edge_index[0] < data_neutral.edge_index[1]
train_neutral_edge_label_index = data_neutral.edge_index[:, mask_neutral]

edge_attr_dim = 0
if hasattr(data, "edge_attr") and data.edge_attr is not None:
    edge_attr_dim = data.edge_attr.size(-1)


model = ReFINe_plus(
    num_nodes=data.num_nodes,
    embedding_dim=args.embedding_dim,
    num_layers=args.layers,
    num_users=num_users,
    num_items=num_items,
    edge_attr_dim=edge_attr_dim,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

negative_sampling_probabilities = torch.ones(num_users, num_items, device=device)
negative_sampling_probabilities[train_edge_label_index[0], train_edge_label_index[1]-num_users] = 0
negative_sampling_probabilities[train_neg_edge_label_index[0], train_neg_edge_label_index[1]-num_users] = args.real_neg_samp_prob
# negative_sampling_probabilities[train_neutral_edge_label_index[0], train_neutral_edge_label_index[1]-num_users] = 1.  # cj neutral

mse_loss = MSELoss()

def train():
    total_loss = total_examples = 0
    num_neg = num_items // 10

    for index in train_loader:
        out = model.get_embedding(
            data.edge_index,
            edge_attr=(data.edge_attr if hasattr(data, "edge_attr") else None),
        )

        pos_edge_label_index = train_edge_label_index[:, index]
        out_src = out[pos_edge_label_index[0]]
        out_dst = out[pos_edge_label_index[1]]

        neg_edge_label_index = torch.multinomial(
            negative_sampling_probabilities[pos_edge_label_index[0]],
            num_samples=num_neg,
            replacement=False
        ) + num_users
        out_dst_neg = out[neg_edge_label_index[1]]

        pos_rank = torch.mul(out_src, out_dst).sum(dim=1)
        neg_rank = torch.mul(out_src.unsqueeze(dim=1), out_dst_neg).sum(dim=-1)

        optimizer.zero_grad()
        loss = torch.log(1 + torch.exp(neg_rank - pos_rank.unsqueeze(dim=1)).sum(dim=1)).mean()
        lambda_reg = 1e-7
        reg_loss = model.embedding.weight.norm(p=2).pow(2)
        loss += (lambda_reg / 2) * reg_loss

        ae_loss, user_latent, item_latent = model.compute_ae_loss(train_neg_edge_label_index, device)
        align_loss = model.compute_align_loss(
            data.edge_index,
            user_latent,
            item_latent,
            edge_attr=(data.edge_attr if hasattr(data, "edge_attr") else None)
        )

        loss += (ae_loss + align_loss)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(ks: list):
    emb = model.get_embedding(
        data.edge_index,
        edge_attr=(data.edge_attr if hasattr(data, "edge_attr") else None)
    )
    user_emb, item_emb = emb[:num_users], emb[num_users:]

    results = list()
    for k in ks:
        precision = recall = ndcg = total_examples = 0
        for start in range(0, num_users, test_batch_size):
            end = start + test_batch_size
            logits = user_emb[start:end] @ item_emb.t()

            mask = ((train_edge_label_index[0] >= start) &
                    (train_edge_label_index[0] < end))
            logits[train_edge_label_index[0, mask] - start,
                   train_edge_label_index[1, mask] - num_users] = float('-inf')
            mask_neg = ((train_neg_edge_label_index[0] >= start) &
                        (train_neg_edge_label_index[0] < end))
            logits[train_neg_edge_label_index[0, mask_neg] - start,
                   train_neg_edge_label_index[1, mask_neg] - num_users] = float('-inf')
            mask_neutral = ((train_neutral_edge_label_index[0] >= start) &
                            (train_neutral_edge_label_index[0] < end))
            logits[train_neutral_edge_label_index[0, mask_neutral] - start,
                   train_neutral_edge_label_index[1, mask_neutral] - num_users] = float('-inf')

            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            mask = ((data.edge_label_index[0] >= start) &
                    (data.edge_label_index[0] < end))
            ground_truth[data.edge_label_index[0, mask] - start,
                         data.edge_label_index[1, mask] - num_users] = True
            node_count = degree(data.edge_label_index[0, mask] - start,
                                num_nodes=logits.size(0))

            topk_index = logits.topk(k, dim=-1).indices
            isin_mat = ground_truth.gather(1, topk_index)

            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())

            relevance_scores = ground_truth.float()
            ideal_relevance_scores = relevance_scores.sort(dim=1, descending=True).values[:, :k]
            log2_k = torch.log2(torch.arange(2, k + 2, device=logits.device, dtype=torch.float))
            dcg_scores = (relevance_scores.gather(1, topk_index) / log2_k).sum(dim=1)
            ideal_dcg_scores = (ideal_relevance_scores / log2_k).sum(dim=1)
            ndcg += float((dcg_scores / ideal_dcg_scores.clamp(1e-6)).sum())

            total_examples += int((node_count > 0).sum())

        results.append((precision / total_examples, recall / total_examples, ndcg / total_examples))

    return results


if not os.path.exists('result'):
    os.makedirs('result')
if not os.path.exists('result/' + args.dataset):
    os.makedirs('result/' + args.dataset)

path_name = 'result/' + args.dataset + '/' + args.path_name + '.pt'
early_stopping = utils.EarlyStopping(patience=args.early_stopping_step, verbose=True, path=path_name)

topks = args.top_k
start_time = time.time()
for epoch in range(1, args.epochs + 1):
    loss = train()

    if epoch % args.evaluation_step == 0:
        results = test(ks=topks)

        print(f'\nEpoch: {epoch:03d}, '
              f'Loss: {loss:.4f}')
        for k, (precision, recall, ndcg) in zip(topks, results):
            if k < 10:
                print(f'Precision@{k}: {precision:7.4f}, '
                      f'Recall@{k}: {recall:7.4f}, '
                      f'NDCG@{k}: {ndcg:7.4f}')
            else:
                print(f'Precision@{k}: {precision:.4f}, '
                      f'Recall@{k}: {recall:.4f}, '
                      f'NDCG@{k}: {ndcg:.4f}')

        early_stopping(epoch, results, model)

        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {early_stopping.best_epoch}. Best Validation Results:")
            for k, (precision, recall, ndcg) in zip(topks, early_stopping.best_metrics):
                if k < 10:
                    print(f'Precision@{k}: {precision:7.4f}, '
                          f'Recall@{k}: {recall:7.4f}, '
                          f'NDCG@{k}: {ndcg:7.4f}')
                else:
                    print(f'Precision@{k}: {precision:.4f}, '
                          f'Recall@{k}: {recall:.4f}, '
                          f'NDCG@{k}: {ndcg:.4f}')
            break

end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining completed in {training_time} seconds.")

if not os.path.exists(path_name):
    raise FileNotFoundError(f"Best checkpoint not found: {path_name}")

if not early_stopping.early_stop:
    print("Early stopping does not trigger. Loading the best checkpoint saved during training.")

result_dir = Path("result") / args.dataset
result_dir.mkdir(parents=True, exist_ok=True)

best_ckpt_path = result_dir / f"{args.path_name}.pt"
test_metrics_path = result_dir / f"{args.path_name}_test_metrics.json"
val_metrics_path = result_dir / f"{args.path_name}_best_val_metrics.json"

if best_ckpt_path.exists():
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
else:
    torch.save(model.state_dict(), best_ckpt_path)

data, _, _ = data_loader.data_loading(project_dir=args.project_dir, load_val_or_test='test')
num_users, num_items = data['user'].num_nodes, data['item'].num_nodes
data = data.to_homogeneous().to(device)

model.eval()
print('\n#############################################################################')
print('Final Test Results')
results = test(ks=topks)

metrics_dict = {}
for k, (precision, recall, ndcg) in zip(topks, results):
    metrics_dict[f"Precision@{k}"] = float(precision)
    metrics_dict[f"Recall@{k}"] = float(recall)
    metrics_dict[f"NDCG@{k}"] = float(ndcg)

    if k < 10:
        print(f'Precision@{k}: {precision:7.4f}, '
              f'Recall@{k}: {recall:7.4f}, '
              f'NDCG@{k}: {ndcg:7.4f}')
    else:
        print(f'Precision@{k}: {precision:.4f}, '
              f'Recall@{k}: {recall:.4f}, '
              f'NDCG@{k}: {ndcg:.4f}')

print('#############################################################################')

with open(test_metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

print(f"Saved checkpoint: {best_ckpt_path}")
print(f"Saved test metrics: {test_metrics_path}")

exit()
