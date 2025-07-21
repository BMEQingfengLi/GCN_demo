import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 2025.7.17 node classification (batchsize = 1)

def generate_data(num_nodes=100):
    adj = np.random.randint(0, 2, (num_nodes, num_nodes))
    adj = np.triu(adj)
    adj = adj + adj.T - np.diag(adj.diagonal())
    np.fill_diagonal(adj, 1)  # node with self-connections (according to the ICLR2017 paper)

    features = np.random.randn(num_nodes, 5)

    labels = np.random.randint(0, 2, num_nodes)

    return torch.FloatTensor(features), torch.FloatTensor(adj), torch.LongTensor(labels)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        degree = torch.diag(torch.sum(adj, dim=1))
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0

        norm_adj = torch.mm(torch.mm(degree_inv_sqrt, adj), degree_inv_sqrt)

        x = torch.mm(norm_adj, x)  # node features are first "warped" by normalized adjacency matrix
        x = self.linear(x)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        return F.log_softmax(x, dim=1)


def train_test():
    features, adj, labels = generate_data(100)

    train_mask = torch.zeros(100, dtype=torch.bool)
    train_mask[:80] = 1
    test_mask = ~train_mask

    model = GCN(input_dim=5, hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = criterion(output[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(features, adj).argmax(dim=1)
                correct = (pred[test_mask] == labels[test_mask]).sum().item()
                acc = correct / test_mask.sum().item()
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')


if __name__ == '__main__':
    train_test()
