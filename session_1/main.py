import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


X_train = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Testing Data
X_test = torch.tensor(
    [[0.1, 0.0], [0.0, 0.9], [1.1, 0.0], [0.9, 1.0]], dtype=torch.float32
)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2,4)
        self.layer2 = nn.Linear(4,1)
        
    def forward(self, input):
        l1 = torch.sigmoid(self.layer1(input))
        l2 = self.layer2(l1)
        return l2
    
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.1)
for epoch in range(10000):
    out = model(X_train)
    loss = criterion(out, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        
with torch.no_grad():
    test_outputs = model(X_test)
    print(f"Test Outputs: {test_outputs}")
