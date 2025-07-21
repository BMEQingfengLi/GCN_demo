import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 2025.7.18 using simulated FC matrix for subject-wise label classification

class BrainDataset(Dataset):
    def __init__(self, num_subjects=300, num_regions=87):
        # generate FC matrix of 300 subjects (300,87,87)
        adj = np.random.rand(num_subjects, num_regions, num_regions)
        adj = (adj + adj.transpose(0, 2, 1)) / 2
        for i in range(adj.shape[0]):
            np.fill_diagonal(adj[i], 1)  # self-connection (according to ICLR 2017 and Liu et al. TNNLS 2024)

        # node feature(300,87,3)
        self.features = torch.FloatTensor(np.random.randn(num_subjects, num_regions, 3))

        # adjacency matrix symmetric normalization
        degree = torch.sum(adj, dim=2, keepdim=True)
        self.adj = torch.FloatTensor(adj) / (torch.sqrt(degree) * torch.sqrt(degree.transpose(1, 2) + 1e-8))

        # set labels (150 patients: 1, 150 HC: 0)
        self.labels = torch.LongTensor([1] * 150 + [0] * 150)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.adj[item], self.labels[item]


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, norm_adj):
        x = torch.bmm(norm_adj, x)  # batch-wise matrix multiplication
        return self.linear(x)


class BrainGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(3, 32)
        self.gcn2 = GCNLayer(32, 16)
        self.gcn3 = GCNLayer(16, 2)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = F.relu(self.gcn2(x, adj))
        return F.log_softmax(self.gcn3(x, adj).mean(dim=1), dim=1)


def train():
    features, adj, labels = generate_brain_data()
    train_idx = torch.randperm(300)[:210]  # 70% for training

    model = BrainGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(400):
        batch_idx = train_idx[epoch % 10 * 20: (epoch % 10 + 1) * 20]  # batch=20
        output = model(features[batch_idx], adj[batch_idx])
        loss = F.nll_loss(output, labels[batch_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            test_output = model(features[~train_idx], adj[~train_idx])  # 30% for testing
            acc = (test_output.argmax(1) == labels[~train_idx]).float().mean()
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')


if __name__ == '__main__':
    train()
