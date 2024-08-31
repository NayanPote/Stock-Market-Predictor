from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import torch
from model import LSTM, train_model, predict_future_prices
from sklearn.preprocessing import MinMaxScaler
from model import predict_future_prices

app = Flask(__name__)

# Load the LSTM model
def load_lstm_model(symbol):
    model = LSTM()
    model.load_state_dict(torch.load(f"D:/Python/stock_prediction_website/{symbol}_model.pth"))
    model.eval()
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    stock_symbol = request.json['stock_symbol']
    
    # Load the trained LSTM model
    model = load_lstm_model(stock_symbol)

    # Make prediction
    future_prices = predict_future_prices(stock_symbol, model)

    # Redirect to the prediction result route with the prediction result as a query parameter
    return jsonify({'future_prices': future_prices.tolist()})

@app.route('/prediction_result')
def prediction_result():
    # Get future prices from query parameters
    future_prices = request.args.get('future_prices').split(',')

    # Render the prediction result template with the prediction
    return render_template('prediction_result.html', future_prices=future_prices)



if __name__ == '__main__':
    app.run(debug=True)


