"""
UM-SJTU JI Deep Learning Hands-on
Session 4 - Long Short-Term Memory (LSTM) Networks on Time Series Data
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =============================================================================
# Step 1 & 2: Generate Time Series Data
# =============================================================================

# Generate sine wave data
seq_length = 20
num_samples = 1000

time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data = data[:-1].reshape(-1, seq_length)

# Duplicate and stack the sequence
X = np.repeat(data, num_samples, axis=0)
y = np.sin(time_steps[-1] * np.ones((num_samples, 1)))

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X).unsqueeze(2)  # Add an extra dimension for the single feature
y_tensor = torch.FloatTensor(y)


# =============================================================================
# Step 3: Define the Model
# =============================================================================

class TimeSeries_LSTM(nn.Module):
    def __init__(self):
        super(TimeSeries_LSTM, self).__init__()
        self.lstm = nn.LSTM(1, 50)  # 1 input feature, 50 hidden units
        self.fc = nn.Linear(50, 1)  # 50 input features, 1 output feature

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(1), 50)  # Initial hidden state
        c_0 = torch.zeros(1, x.size(1), 50)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


# =============================================================================
# Step 4: Initialize the Model, Loss, and Optimizer
# =============================================================================

model = TimeSeries_LSTM()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer


# =============================================================================
# Step 5: Train the Model
# =============================================================================

print("Training LSTM...")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')


# =============================================================================
# Step 6: Evaluate the Model
# =============================================================================

num_test_samples = 100

# Generate new sine wave data
test_time_steps = np.linspace(0, np.pi, seq_length + 1)
test_data = np.sin(test_time_steps)
test_data = test_data[:-1].reshape(-1, seq_length)

# Duplicate and stack the sequence
test_X = np.repeat(test_data, num_test_samples, axis=0)
test_y = np.sin(test_time_steps[-1] * np.ones((num_test_samples, 1)))

# Convert to PyTorch tensors
test_X_tensor = torch.FloatTensor(test_X).unsqueeze(2)  # Add an extra dimension for the single feature
test_y_tensor = torch.FloatTensor(test_y)

# Predict and Compare
with torch.no_grad():
    predicted = model(test_X_tensor)

    # Calculate the Mean Squared Error
    mse = criterion(predicted, test_y_tensor).item()
    print(f'\nMean Squared Error on test data: {mse:.6f}')

# Convert predictions to numpy array
predicted_np = predicted.numpy()

# Visualize a sample test sequence and prediction
plt.figure(figsize=(10, 5))
plt.plot(test_time_steps[:-1], test_X[0], label='Input Sequence', linewidth=2)
plt.scatter([test_time_steps[-1]], [test_y[0]], label='Actual Future Value', c='r', s=100, zorder=5)
plt.scatter([test_time_steps[-1]], [predicted_np[0]], label='Predicted Future Value', c='g', s=100, zorder=5)
plt.xlabel('Time step')
plt.ylabel('sin(t)')
plt.title('LSTM Time Series Prediction')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('lstm_prediction.png', dpi=150)
plt.show()
print("\nPlot saved as 'lstm_prediction.png'")
