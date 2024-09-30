# Investra - Smart Investment Portfolio Optimization Platform

Investra is a comprehensive platform designed to help individuals and retail investors manage and optimize their investment portfolios. It offers personalized portfolio analysis, predictive stock models, real-time news sentiment analysis, and interactive visualizations to help users make data-driven investment decisions. With Investra, portfolio management becomes accessible and smarter for everyone.

## Features

- **Personalized Portfolio Analysis**: Analyze your portfolio's diversification, asset allocation, and risk exposure with detailed insights and recommendations.
- **Predictive Models**: Leverage advanced Long Short-Term Memory (LSTM) models to get short-term stock price predictions based on historical data.
- **News Sentiment Analysis**: Get real-time sentiment analysis of news articles related to your portfolio’s stocks using Natural Language Processing (NLP).
- **Interactive Visualizations**: Visualize portfolio performance, risk, and rebalancing suggestions with easy-to-understand charts and graphs.
- **Stock Terms Chatbot**: Understand investment terms with the help of a built-in chatbot that explains key financial concepts. It also real time stock prices.
  
## Installation

Follow these steps to set up Investra on your local machine.

### Prerequisites

Make sure you have the following installed:
- **Python 3.8+**
- **PostgreSQL**
- **Flask**
- **Virtual environment (optional but recommended)**

### Step-by-Step Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/Lis7212/INVESTRA.git
   cd INVESTRA
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the PostgreSQL database:
   - Create a new PostgreSQL database.
   - Update your `config.py` with the correct database URI.

5. Initialize the database:
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

6. Run the Flask application:
   ```bash
   flask run
   ```

7. Access the app by navigating to `http://127.0.0.1:5000/` in your web browser.

## Technologies Used

- **Frontend**: HTML, CSS, Plotly for interactive charts and visualizations.
- **Backend**: Flask (Python), PostgreSQL for database management.
- **Machine Learning**: Long Short-Term Memory (LSTM) models for stock predictions.
- **Natural Language Processing (NLP)**: Sentiment analysis using `distilbert-base-uncased`.
- **APIs**: Integration with news APIs for fetching stock-related news articles.

## Usage

1. **Portfolio Management**: Input your stock portfolio and receive personalized analysis and recommendations.
2. **Predict Stock Movements**: Get short-term predictions for the stocks in your portfolio using the LSTM model.
3. **Sentiment Analysis**: View real-time sentiment analysis based on news articles related to your stocks.
4. **Rebalancing Suggestions**: Get actionable insights into improving your portfolio based on risk tolerance and market trends.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue if you have suggestions for improving the platform.
