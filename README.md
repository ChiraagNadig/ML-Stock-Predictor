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
<img width="850" alt="Screenshot 2024-07-07 at 12 52 12 PM" src="https://github.com/ChiraagNadig/ML-Stock-Predictor/assets/79017920/57c5bba2-a53f-4dd0-991d-e22337a4108f">




ML output for AAPL stock:
<img width="842" alt="Screenshot 2024-07-07 at 1 32 12 PM" src="https://github.com/ChiraagNadig/ML-Stock-Predictor/assets/79017920/d9a5af53-d118-47f9-b230-60eb65a0ef7e">

