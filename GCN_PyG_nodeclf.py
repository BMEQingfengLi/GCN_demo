import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import torch

# 2025.7.18 node classification (batchsize = 1)

def generate_pyg_data(num_nodes=100):
    adj = np.random.randint(0, 2, (num_nodes, num_nodes))
    adj = np.triu(adj)
    adj = adj + adj.T - np.diag(adj.diagonal())

    edge_index = torch.tensor(np.array(np.where(adj > 0)), dtype=torch.long)

    x = torch.randn(num_nodes, 5, dtype=torch.float)

    y = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:80] = 1
    test_mask = ~train_mask

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)


class GCN_PyG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_PyG, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_test_pyg():
    data = generate_pyg_data()

    model = GCN_PyG(input_dim=5, hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data).argmax(dim=1)
                correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
                acc = correct / data.test_mask.sum().item()
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')


if __name__ == '__main__':
    train_test_pyg()