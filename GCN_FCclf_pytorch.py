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
        degree = np.zeros_like(adj)
        for i in range(adj.shape[0]):
            degree[i] = np.diag(np.sum(adj[i], axis=1))
        adj = torch.FloatTensor(adj)
        degree = torch.FloatTensor(degree)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)  # add 1e-8 to avoid value divided by 0
        degree_inv_sqrt[degree_inv_sqrt == float(torch.inf)] = 0  # in case there are isolated nodes

        self.adj = degree_inv_sqrt * adj * degree_inv_sqrt

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
        x = torch.matmul(norm_adj, x)  # batch-wise matrix multiplication
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
    dataset = BrainDataset()
    train_size = int(0.7 * len(dataset))

    train_set, test_set = torch.utils.data.random_split(dataset,
                                                        [train_size, len(dataset) - train_size])  # 70% for training
    train_loader = DataLoader(train_set,
                              batch_size=40,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=10)

    model = BrainGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(400):
        model.train()
        for batch_features, batch_adj, batch_labels in train_loader:
            optimizer.zero_grad()
            output = model(batch_features, batch_adj)
            loss = F.nll_loss(output, batch_labels)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            correct = 0
            for batch_features, batch_adj, batch_labels in test_loader:
                output = model(batch_features, batch_adj)
                correct += (output.argmax(1) == batch_labels).sum().item()
            print(f'Epoch {epoch}, Test ACC: {correct / len(test_set):.4f}')


if __name__ == '__main__':
    train()
