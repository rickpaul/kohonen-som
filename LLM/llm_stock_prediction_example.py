"""
Source:
   + https://medium.com/geekculture/develop-ai-models-to-predict-stock-price-using-ai-openai-chatgpt-efa5ad6675ae
"""

# %%


# %%

import pandas as pd
import pandas_datareader as pdr
import yfinance
import torch

# %%

# Define the number of previous days to use for prediction
num_prev_days = 30

# Use pandas_datareader to obtain the MSFT stock data
yfinance.pdr_override()
data = pdr.data.get_data_yahoo('MSFT', "2022-10-01", "2022-11-30")
data2 = pdr.data.get_data_yahoo('MSFT', "2022-11-30", "2022-12-30")

# %%

# Create a list of the close prices
close_prices = data['Close'].tolist()

# Create a list of tuples containing the previous num_prev_days close prices and the current close price
price_data = [(close_prices[i-num_prev_days:i], close_prices[i]) for i in range(num_prev_days, len(close_prices))]

# Split the data into input and target tensors
inputs = torch.tensor([i[0] for i in price_data], dtype=torch.float)
targets = torch.tensor([i[1] for i in price_data], dtype=torch.float)
# Define the neural network
class LLM(torch.nn.Module):
    def __init__(self, num_prev_days):
        super(LLM, self).__init__()
        self.linear1 = torch.nn.Linear(num_prev_days, 10)
        self.linear2 = torch.nn.Linear(10, 1)
        
    def forward(self, input):
        output = self.linear1(input)
        output = torch.relu(output)
        output = self.linear2(output)
        return output

# Instantiate the LLM model
model = LLM(num_prev_days)

# Define the loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(500):
    # Forward pass
    predictions = model(inputs)
    
    # Compute loss
    loss = loss_fn(predictions, targets)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')
# %%
# Use the trained model to make predictions for the next month
predictions = []
last_prices = close_prices[-num_prev_days:]

for i in range(1,31):
    input = torch.tensor(last_prices, dtype=torch.float)
    prediction = model(input).item()
    predictions.append(prediction)
    last_prices.pop(0)
    last_prices.append(prediction)

# Print the predictions
print(predictions)  