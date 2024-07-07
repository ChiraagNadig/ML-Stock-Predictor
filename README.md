# ML-Stock-Predictor
Web application with ML model to predict stock market price of stock using historical stock data

Used Tensorflow, Yfinance, MatPlotlib, Numpy, Pandas, Streamlit, Keras


Video demo:
https://drive.google.com/file/d/1-XvIs-lNgCTJm982GyFUm81u1LZMzKIg/view?usp=sharing

Web application contains:
- Text box: To enter ticker symbol of stock
- Table: To display stock data of selected stock from 2012 to 2024
- Graph image: To display 50-day moving average vs daily closing price of stock
- Graph image: To display 50-day moving average vs 100-day moving average vs daily closing price of stock
- Graph image: To display 50-day moving average vs 100-day moving average vs 200-day moving average vs daily closing price of stock

The 50-day, 100-day, and 200-day moving average are used by ML model for predictions
ML output (Graph image showing Original price vs Predicted price of stock) takes longer time to load, hence it was not shown in the demo video

ML output for GOOG stock:
<img width="1440" alt="Screenshot 2024-07-07 at 12 52 12 PM" src="https://github.com/ChiraagNadig/ML-Stock-Predictor/assets/79017920/f3ea1dd0-0068-402a-ab30-a27b797a05e8">

ML output for AAPL stock:
<img width="1440" alt="Screenshot 2024-07-07 at 1 32 12 PM" src="https://github.com/ChiraagNadig/ML-Stock-Predictor/assets/79017920/eeab974d-3d0b-4075-ae87-37be8b8a8204">
