import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Function to get stock data and store it in a CSV file
def get_stock_data(symbol, end_date):
    end_date_str = end_date.strftime('%Y-%m-%d')
    stock_data = yf.download(symbol, start="2024-01-01", end=end_date_str)
    stock_data.to_csv(f"D:/Python/stock_prediction_website/{symbol}_data.csv")

# Function to train the LSTM model
def train_model(symbol):
    dataset = np.loadtxt(f"D:/Python/stock_prediction_website/{symbol}_data.csv", delimiter=",", skiprows=1, usecols=4)
    dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset_normalized = scaler.fit_transform(dataset).reshape(-1)

    # Define hyperparameters
    input_size = 1
    hidden_layer_size = 100
    output_size = 1
    learning_rate = 0.001
    epochs = 150

    model = LSTM(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=output_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        for seq, labels in generate_sequence_data(dataset_normalized):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    
    # Save the model
    torch.save(model.state_dict(), f"D:/Python/stock_prediction_website/{symbol}_model.pth")

# Function to generate sequence data for training
def generate_sequence_data(data):
    sequence_length = 10
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        labels = data[i+sequence_length:i+sequence_length+1]
        yield torch.FloatTensor(seq), torch.FloatTensor(labels)

# Function to predict future stock prices
def predict_future_prices(symbol):
    model = LSTM()
    model.load_state_dict(torch.load(f"D:/Python/stock_prediction_website/{symbol}_model.pth"))
    model.eval()

    dataset = np.loadtxt(f"D:/Python/stock_prediction_website/{symbol}_data.csv", delimiter=",", skiprows=1, usecols=4)
    dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset_normalized = scaler.fit_transform(dataset).reshape(-1)

    with torch.no_grad():
        test_seq = torch.FloatTensor(dataset_normalized[-10:])
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        future_prices = []
        for i in range(30):
            future_price = model(test_seq)
            future_prices.append(future_price.item())
            test_seq = torch.cat((test_seq[1:], future_price))

    return scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

# Function to plot actual and predicted stock prices
def plot_actual_vs_predicted(symbol, actual_prices, predicted_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(range(len(actual_prices), len(actual_prices) + len(predicted_prices)), predicted_prices, label='Predicted Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title(f'Actual vs Predicted Prices for {symbol}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    symbol = input("Enter the stock symbol: ").upper()
    end_date = datetime.now()

    get_stock_data(symbol, end_date)
    train_model(symbol)
    future_prices = predict_future_prices(symbol)

    dataset = np.loadtxt(f"D:/Python/stock_prediction_website/{symbol}_data.csv", delimiter=",", skiprows=1, usecols=4)
    actual_prices = dataset[-30:]

    plot_actual_vs_predicted(symbol, actual_prices, future_prices)

