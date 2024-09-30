from flask import Flask, render_template, redirect, url_for, flash, session, request, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField, FloatField, DateField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import psycopg2
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
import re
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import subprocess
import json
import model_1  # Importing your model_1.py
import model_2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'f6e35d4c88a36cd5b94e6ab1023661c5c1beb9ef65beb8b9'

# News Analysis Model and Tokenizer Setup
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Load intents from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Create TF-IDF vectorizer
vectorizer = model_1.create_vectorizer(intents)

# Database connection
conn = psycopg2.connect(
    host="localhost",
    database="project",
    user="postgres",
    password="vava@1502"
)
cursor = conn.cursor()

class SignupForm(FlaskForm):
    full_name = StringField('Full Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email Address', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    phone = StringField('Phone Number', validators=[Length(max=15)], render_kw={"placeholder": "Optional"})
    date_of_birth = DateField('Date of Birth', validators=[DataRequired()], format='%Y-%m-%d')
    address = StringField('Address', validators=[Length(max=200)], render_kw={"placeholder": "Optional"})
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email Address', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class PortfolioForm(FlaskForm):
    num_stocks = IntegerField('Number of Stock Symbols', validators=[DataRequired()], render_kw={"min": "1", "max": "3"})
    stock_symbol1 = StringField('Stock Ticker Symbol 1', validators=[Optional()])
    number_of_shares1 = IntegerField('Number of Shares 1', validators=[Optional()])
    purchase_price1 = FloatField('Purchase Price 1', validators=[Optional()])
    current_price1 = FloatField('Current Price 1', validators=[Optional()])
    purchase_date1 = DateField('Purchase Date 1', validators=[Optional()], format='%Y-%m-%d')
    
    stock_symbol2 = StringField('Stock Ticker Symbol 2', validators=[Optional()])
    number_of_shares2 = IntegerField('Number of Shares 2', validators=[Optional()])
    purchase_price2 = FloatField('Purchase Price 2', validators=[Optional()])
    current_price2 = FloatField('Current Price 2', validators=[Optional()])
    purchase_date2 = DateField('Purchase Date 2', validators=[Optional()], format='%Y-%m-%d')
    
    stock_symbol3 = StringField('Stock Ticker Symbol 3', validators=[Optional()])
    number_of_shares3 = IntegerField('Number of Shares 3', validators=[Optional()])
    purchase_price3 = FloatField('Purchase Price 3', validators=[Optional()])
    current_price3 = FloatField('Current Price 3', validators=[Optional()])
    purchase_date3 = DateField('Purchase Date 3', validators=[Optional()], format='%Y-%m-%d')

    submit = SubmitField('Save Portfolio')

class RiskToleranceForm(FlaskForm):
    investment_goals = SelectField('Investment Goals', choices=[
        ('retirement', 'Retirement'), 
        ('education', 'Education'), 
        ('wealth_accumulation', 'Wealth Accumulation')
    ], validators=[DataRequired()])
    investment_horizon = SelectField('Investment Horizon', choices=[
        ('short_term', 'Short-term'), 
        ('medium_term', 'Medium-term'), 
        ('long_term', 'Long-term')
    ], validators=[DataRequired()])
    risk_tolerance_level = SelectField('Risk Tolerance Level', choices=[
        ('conservative', 'Conservative'), 
        ('moderate', 'Moderate'), 
        ('aggressive', 'Aggressive')
    ], validators=[DataRequired()])
    income_stability = SelectField('Income Stability', choices=[
        ('stable', 'Stable'), 
        ('variable', 'Variable')
    ], validators=[DataRequired()])
    financial_situation = SelectField('Current Financial Situation', choices=[
        ('surplus', 'Surplus'), 
        ('balanced', 'Balanced'), 
        ('deficit', 'Deficit')
    ], validators=[DataRequired()])
    investment_experience = SelectField('Investment Experience', choices=[
        ('beginner', 'Beginner'), 
        ('intermediate', 'Intermediate'), 
        ('advanced', 'Advanced')
    ], validators=[DataRequired()])
    submit = SubmitField('Save Risk Tolerance')

def calculate_metrics(df):
    df['Daily Return'] = df['close'].pct_change()
    df['Moving Average 30'] = df['close'].rolling(window=30).mean()
    df['Volatility'] = df['Daily Return'].rolling(window=30).std()
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod() - 1
    return df.dropna()

def recommend_rebalancing(risk_tolerance_level, stock_metrics, investment_goals, investment_horizon, income_stability, financial_situation, investment_experience):
    recommendations = {}
    
    for stock, metrics in stock_metrics.items():
        avg_return = metrics['Daily Return'].mean()
        volatility = metrics['Volatility'].mean()

        # Adjust target allocation based on additional risk tolerance fields
        if risk_tolerance_level == 'moderate':
            target_allocation = 1 / volatility
        elif risk_tolerance_level == 'aggressive':
            target_allocation = 1 / (volatility * 0.8)
        else:
            target_allocation = 1 / (volatility * 1.2)  # Conservative risk tolerance

        # Further adjustments based on other risk tolerance factors
        if investment_goals == 'retirement':
            target_allocation *= 1.1  # Increase allocation for long-term goals
        if investment_horizon == 'short_term':
            target_allocation *= 0.8  # Decrease allocation for short-term horizons
        if income_stability == 'variable':
            target_allocation *= 0.9  # Decrease allocation for variable income
        if financial_situation == 'deficit':
            target_allocation *= 0.7  # Decrease allocation for deficit financial situations
        if investment_experience == 'beginner':
            target_allocation *= 0.85  # Decrease allocation for beginners
        
        recommendations[stock] = target_allocation
    
    # Normalize allocations
    total_allocation = np.sum(list(recommendations.values()))
    for stock in recommendations:
        recommendations[stock] /= total_allocation
    
    return recommendations


def generate_recommendations(recommendations, current_value, portfolio):
    recommended_allocation = pd.DataFrame({
        'Stock': list(recommendations.keys()),
        'Current Allocation': [1 / len(recommendations)] * len(recommendations),  # Assuming equal initial allocation
        'Recommended Allocation': list(recommendations.values())
    })
    
    for stock in recommended_allocation['Stock']:
        recommended_allocation.loc[recommended_allocation['Stock'] == stock, 'Recommended Value'] = current_value * recommendations[stock]
        
        # Calculate the number of shares to buy or sell
        current_stock = next((item for item in portfolio if item['symbol'] == stock), None)
        if current_stock:
            current_value_per_stock = current_stock['number_of_shares'] * current_stock['current_price']
            recommended_allocation.loc[recommended_allocation['Stock'] == stock, 'Current Value'] = current_value_per_stock
            recommended_allocation.loc[recommended_allocation['Stock'] == stock, 'Shares to Buy/Sell'] = (recommended_allocation.loc[recommended_allocation['Stock'] == stock, 'Recommended Value'] - current_value_per_stock) / current_stock['current_price']
        else:
            recommended_allocation.loc[recommended_allocation['Stock'] == stock, 'Current Value'] = 0
            recommended_allocation.loc[recommended_allocation['Stock'] == stock, 'Shares to Buy/Sell'] = recommended_allocation.loc[recommended_allocation['Stock'] == stock, 'Recommended Value'] / current_stock['current_price']
    
    return recommended_allocation


def plot_combined_performance(stock_df, stocks):
    fig = go.Figure()

    for stock in stocks:
        stock_data = stock_df[stock_df['Company'] == stock]
        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Cumulative Return'],
                                 mode='lines', name=f'{stock} Cumulative Return'))

        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Moving Average 30'],
                                 mode='lines', name=f'{stock} Moving Average 30', line=dict(dash='dash')))

    fig.update_layout(title='Historical Performance Comparison',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Return',
                      legend_title='Legend',
                      margin=dict(l=40, r=20, t=40, b=40))  # Adjust margins as needed

    return fig.to_json()


def plot_recommendations(recommendations_df):
    fig = px.bar(recommendations_df, x='Stock', y=['Current Allocation', 'Recommended Allocation'],
                 title='Current vs Recommended Allocation',
                 labels={'value': 'Allocation', 'Stock': 'Stock'},
                 height=400)
    fig.show()

# News Analysis Functions
def fetch_news(api_key, company_name):
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={company_name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        
        articles = []
        # Iterate over results to get the required data
        for article in news_data.get('results', []):
            if 'title' in article and 'link' in article:  # Ensure both title and link are available
                articles.append({
                    'title': article['title'],  # Article title
                    'url': article['link'],     # URL for the full article
                })
        return articles
    
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news data: {e}")
        return []

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Analyze sentiment
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = model(inputs['input_ids'])[0]
    logits = outputs.numpy()
    sentiment = np.argmax(logits, axis=1).item()
    return sentiment

# Map sentiment to investment decision
def sentiment_to_decision(sentiment_score):
    if sentiment_score == 2:
        return 'Buy'
    elif sentiment_score == 0:
        return 'Sell'
    else:
        return 'Hold'

# Calculate decision percentages
def calculate_percentages(sentiment_decisions):
    total = len(sentiment_decisions)
    decision_counts = {decision: sentiment_decisions.count(decision) for decision in ['Buy', 'Sell', 'Hold']}
    decision_percentages = {decision: (count / total) * 100 for decision, count in decision_counts.items()}
    return decision_counts, decision_percentages

# Visualize sentiment and decision
def visualize_decision_and_sentiment(decision_counts, decision_percentages, sentiments):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    categories = list(decision_counts.keys())
    counts = list(decision_counts.values())
    percentages = list(decision_percentages.values())

    blue_palette = sns.color_palette(["#c6a4d9", "#9c5eab", "#5A0DA6"])  # Light to dark blue
    red_palette = sns.color_palette(["#f8d7da", "#f5a4a1", "#d9534f"])
    # Decision visualization (Left plot)
    sns.barplot(ax=axes[0], x=categories, y=counts, palette=blue_palette)
    axes[0].set_title("Decision Percentages Based on Sentiment Analysis", fontsize=16)
    axes[0].set_xlabel("Decision", fontsize=14)
    axes[0].set_ylabel("Number of Articles", fontsize=14)

    # Add percentage annotations for decision bars
    for p in axes[0].patches:
        height = p.get_height()
        category = p.get_x() + p.get_width() / 2
        category_label = categories[int(category)]
        label = f'{height} ({percentages[categories.index(category_label)]:.1f}%)'
        axes[0].annotate(label,
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='center', xytext=(0, 10),
                         textcoords='offset points')

    # Sentiment visualization (Right plot)
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    sentiment_counts = [sentiments.count(0), sentiments.count(1), sentiments.count(2)]
    sns.barplot(ax=axes[1], x=sentiment_labels, y=sentiment_counts, palette=red_palette)
    axes[1].set_title("Sentiment Distribution", fontsize=16 )
    axes[1].set_xlabel("Sentiment", fontsize=14)
    axes[1].set_ylabel("Number of Articles", fontsize=14)

    plt.tight_layout()
    
    # Save plot to a BytesIO object and encode as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    plt.close()
    
    return plot_url

def summarize_decisions(stock_symbol, decision_counts, decision_percentages):
    # Generate a summary for a given stock's decision counts and percentages
    summary = f"For {stock_symbol}: "
    
    if not decision_counts:
        return f"No sentiment data available for {stock_symbol}."
    
    total_articles = sum(decision_counts.values())
    summary += f"A total of {total_articles} articles were analyzed. "
    
    for decision, count in decision_counts.items():
        percentage = decision_percentages.get(decision, 0)
        summary += f"{percentage:.2f}% of the articles were classified as '{decision}' ({count} articles). "
    
    # You can modify this to add more insights if needed.
    return summary

def generate_recommendations_advice(investment_goals, investment_horizon, 
                                    risk_tolerance_level, income_stability, 
                                    financial_situation, investment_experience):
    recommendations = {}

    # Basic recommendations based on investment goals
    if investment_goals == 'retirement':
        recommendations['goal_advice'] = "Consider focusing on long-term growth stocks and retirement accounts."
    elif investment_goals == 'education':
        recommendations['goal_advice'] = "Look into tax-advantaged savings plans such as 529 plans."
    elif investment_goals == 'wealth_accumulation':
        recommendations['goal_advice'] = "Diversify your portfolio with a mix of stocks and ETFs."

    # Recommendations based on investment horizon
    if investment_horizon == 'short_term':
        recommendations['horizon_advice'] = "Consider safer investments such as bonds or savings accounts."
    elif investment_horizon == 'medium_term':
        recommendations['horizon_advice'] = "A balanced mix of stocks and bonds may suit your needs."
    elif investment_horizon == 'long_term':
        recommendations['horizon_advice'] = "Focus on growth stocks and possibly some high-yield investments."

    # Recommendations based on risk tolerance level
    if risk_tolerance_level == 'conservative':
        recommendations['risk_advice'] = "Opt for lower-risk investments such as bonds and stable blue-chip stocks."
    elif risk_tolerance_level == 'moderate':
        recommendations['risk_advice'] = "Consider a balanced portfolio with a mix of stocks and bonds."
    elif risk_tolerance_level == 'aggressive':
        recommendations['risk_advice'] = "Invest in high-growth stocks and sector-specific ETFs."

    # Recommendations based on income stability
    if income_stability == 'stable':
        recommendations['income_advice'] = "You can afford to take more risks with your investments."
    elif income_stability == 'variable':
        recommendations['income_advice'] = "Focus on safer investments and maintain an emergency fund."

    # Recommendations based on financial situation
    if financial_situation == 'surplus':
        recommendations['financial_advice'] = "Invest surplus funds in growth-oriented assets."
    elif financial_situation == 'balanced':
        recommendations['financial_advice'] = "Maintain a balanced portfolio to sustain your financial health."
    elif financial_situation == 'deficit':
        recommendations['financial_advice'] = "Prioritize paying off debts before making significant investments."

    # Recommendations based on investment experience
    if investment_experience == 'beginner':
        recommendations['experience_advice'] = "Start with low-cost index funds or mutual funds."
    elif investment_experience == 'intermediate':
        recommendations['experience_advice'] = "Consider actively managed funds and sector ETFs."
    elif investment_experience == 'advanced':
        recommendations['experience_advice'] = "You may want to explore options trading or direct stock investments."

    # Suggested portfolio allocation
    if risk_tolerance_level == 'conservative':
        recommendations['allocation'] = "70% Bonds, 20% Blue-Chip Stocks, 10% Cash"
    elif risk_tolerance_level == 'moderate':
        recommendations['allocation'] = "50% Stocks, 40% Bonds, 10% Cash"
    elif risk_tolerance_level == 'aggressive':
        recommendations['allocation'] = "80% Stocks, 15% High-Yield Bonds, 5% Cash"

    return recommendations





@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/signup-login', methods=['GET', 'POST'])
def signup_login():
    signup_form = SignupForm()
    login_form = LoginForm()

    if signup_form.validate_on_submit():
        hashed_password = generate_password_hash(signup_form.password.data)
        cursor.execute("""
            INSERT INTO users (full_name, email, password, phone, date_of_birth, address)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            signup_form.full_name.data,
            signup_form.email.data,
            hashed_password,
            signup_form.phone.data,
            signup_form.date_of_birth.data,
            signup_form.address.data
        ))
        conn.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('signup_login'))

    if login_form.validate_on_submit():
        cursor.execute("SELECT * FROM users WHERE email = %s", (login_form.email.data,))
        user = cursor.fetchone()
        if user and check_password_hash(user[3], login_form.password.data):
            session['user_id'] = user[0]
            flash('Login successful!', 'success')
            return redirect(url_for('index'))  # Redirect to the portfolio page or another page
        else:
            flash('Login failed. Please check your email and password.', 'danger')

    return render_template('signup_login.html', signup_form=signup_form, login_form=login_form)

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    form = PortfolioForm()
    if form.validate_on_submit():
        stock_data = []
        for i in range(1, 4):  # Assuming max 3 stocks
            if getattr(form, f'stock_symbol{i}').data:
                stock_data.append({
                    'symbol': getattr(form, f'stock_symbol{i}').data,
                    'number_of_shares': getattr(form, f'number_of_shares{i}').data,
                    'purchase_price': getattr(form, f'purchase_price{i}').data,
                    'current_price': getattr(form, f'current_price{i}').data
                })

        # Save stock data to session or database
        session['portfolio'] = stock_data

        flash('Portfolio saved successfully!', 'success')
        return redirect(url_for('risk_tolerance'))

    return render_template('portfolio.html', form=form)

@app.route('/risk_tolerance', methods=['GET', 'POST'])
def risk_tolerance():
    form = RiskToleranceForm()
    if form.validate_on_submit():
        # Save risk tolerance data to the database
        cursor.execute("""
            INSERT INTO risk_tolerance (user_id, investment_goals, investment_horizon, risk_tolerance_level, income_stability, financial_situation, investment_experience)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            session['user_id'],
            form.investment_goals.data,
            form.investment_horizon.data,
            form.risk_tolerance_level.data,
            form.income_stability.data,
            form.financial_situation.data,
            form.investment_experience.data
        ))
        conn.commit()

        # Store entire risk tolerance data in session
        session['risk_tolerance'] = {
            'investment_goals': form.investment_goals.data,
            'investment_horizon': form.investment_horizon.data,
            'risk_tolerance_level': form.risk_tolerance_level.data,
            'income_stability': form.income_stability.data,
            'financial_situation': form.financial_situation.data,
            'investment_experience': form.investment_experience.data
        }

        flash('Risk Tolerance saved successfully!', 'success')
        return redirect(url_for('confirm_risk_tolerance'))

    return render_template('risk_tolerance.html', form=form)


@app.route('/confirm_risk_tolerance', methods=['GET', 'POST'])
def confirm_risk_tolerance():
    # Retrieve risk tolerance data from session
    portfolio = session.get('portfolio', [])
    risk_tolerance = session.get('risk_tolerance', {})

    # If confirmed, redirect to portfolio analysis
    if request.method == 'POST':
        return redirect(url_for('portfolio_analysis'))

    # Render confirmation page with portfolio and risk tolerance data
    return render_template('confirm_risk_tolerance.html', 
                           portfolio=portfolio, 
                           risk_tolerance=risk_tolerance)


import json


@app.route('/portfolio_analysis')
def portfolio_analysis():
    # Retrieve portfolio and risk tolerance from session
    portfolio = session.get('portfolio', [])
    risk_tolerance = session.get('risk_tolerance', 'moderate')

    # Load the stock data from your CSV file
    df = pd.read_csv('last3years_stock_data_updated.csv')
    stocks = [stock['symbol'] for stock in portfolio]
    stock_df = df[df['Company'].isin(stocks)]
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = calculate_metrics(stock_df)

    stock_metrics = {stock: stock_df[stock_df['Company'] == stock] for stock in stocks}
    cursor.execute("SELECT investment_goals, investment_horizon, risk_tolerance_level, income_stability, financial_situation, investment_experience FROM risk_tolerance WHERE user_id = %s", (session['user_id'],))
    risk_tolerance_data = cursor.fetchone()
    investment_goals, investment_horizon, risk_tolerance_level, income_stability, financial_situation, investment_experience = risk_tolerance_data
    recommendations = recommend_rebalancing(risk_tolerance_level, stock_metrics, investment_goals, investment_horizon, income_stability, financial_situation, investment_experience)
    total_current_value = sum(stock['number_of_shares'] * stock['current_price'] for stock in portfolio)
    recommended_allocation_df = generate_recommendations(recommendations, total_current_value, portfolio)

    # Create Plotly figures
    performance_fig = go.Figure()
    for stock in stocks:
        stock_data = stock_df[stock_df['Company'] == stock]
        performance_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Cumulative Return'],
                                             mode='lines', name=f'{stock} Cumulative Return'))
        performance_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Moving Average 30'],
                                             mode='lines', name=f'{stock} Moving Average 30', line=dict(dash='dash')))

    performance_fig.update_layout(title='Historical Performance Comparison',
                                  xaxis_title='Date',
                                  yaxis_title='Cumulative Return',
                                  legend_title='Legend')

    allocation_fig = px.bar(recommended_allocation_df, x='Stock', y=['Current Allocation', 'Recommended Allocation'],
                           title='Current vs Recommended Allocation',
                           labels={'value': 'Allocation', 'Stock': 'Stock'},
                           height=400)

    # Convert figures to JSON
    performance_plot_json = json.dumps(performance_fig, cls=plotly.utils.PlotlyJSONEncoder)
    allocation_plot_json = json.dumps(allocation_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('portfolio_analysis.html', 
                           recommendations=recommended_allocation_df.to_dict('records'),
                           combined_performance_plot=performance_plot_json,
                           allocation_plot=allocation_plot_json)


@app.route('/news_analysis')
def news_analysis():
    portfolio = session.get('portfolio', [])
    stock_symbols = [stock['symbol'] for stock in portfolio]
    all_decision_counts = []
    all_decision_percentages = []
    all_sentiments = []
    all_plot_urls = []
    all_articles = []
    summaries = [] 
    
    for stock_symbol in stock_symbols:
        news_articles = fetch_news(api_key="pub_522433f767e7753da50bc8b7ec8aec2a02545", company_name=stock_symbol)
        
        if not news_articles:
            print(f"No news articles found for {stock_symbol}")
            all_decision_counts.append({})
            all_decision_percentages.append({})
            all_sentiments.append([])
            all_plot_urls.append(None)
            all_articles.append([])
            summaries.append(f"No news articles or sentiment analysis data for {stock_symbol}.")
            continue

        # Get the top 5 news articles
        top_articles = news_articles[:5]
        all_articles.append(top_articles)
        
        cleaned_articles = [
            preprocess_text(
                (article.get('title') or '') + ' ' + (article.get('description') or '')
            ) for article in news_articles
        ]
        print(f"Cleaned Articles for {stock_symbol}: {cleaned_articles}")
        
        sentiments = [analyze_sentiment(article) for article in cleaned_articles]
        print(f"Sentiments for {stock_symbol}: {sentiments}")
        
        decisions = [sentiment_to_decision(sentiment) for sentiment in sentiments]
        print(f"Decisions for {stock_symbol}: {decisions}")
        
        decision_counts, decision_percentages = calculate_percentages(decisions)
        print(f"Decision Counts for {stock_symbol}: {decision_counts}")
        print(f"Decision Percentages for {stock_symbol}: {decision_percentages}")
        
        plot_url = visualize_decision_and_sentiment(decision_counts, decision_percentages, sentiments)
        print(f"Plot URL for {stock_symbol}: {plot_url}")
        
        all_decision_counts.append(decision_counts)
        all_decision_percentages.append(decision_percentages)
        all_sentiments.append(sentiments)
        all_plot_urls.append(plot_url)

        summary = summarize_decisions(stock_symbol, decision_counts, decision_percentages)
        summaries.append(summary)
    
    # Ensure all lists are of the same length before rendering the template
    if len(stock_symbols) == len(all_decision_counts) == len(all_decision_percentages) == len(all_plot_urls):
        return render_template('news_analysis.html',
                               stock_symbols=stock_symbols,
                               all_decision_counts=all_decision_counts,
                               all_decision_percentages=all_decision_percentages,
                               all_plot_urls=all_plot_urls,
                               all_articles=all_articles,
                               summaries=summaries)
    else:
        return "Data mismatch error: One of the lists is shorter than expected."

@app.route('/stock-prediction')
def stock_prediction():
    # Just return a link to the already running Streamlit app
    return '<a href="http://localhost:8501" target="_blank">Go to Stock Prediction</a>'

@app.route('/recommendations', methods=['GET'])
def recommendation_page():
    # Fetch risk tolerance data from session
    risk_tolerance = session.get('risk_tolerance')

    # Check if risk_tolerance data exists
    if not risk_tolerance:
        flash('No risk tolerance data found. Please complete the risk tolerance form first.', 'warning')
        return redirect(url_for('risk_tolerance'))

    # Generate recommendations based on risk tolerance data
    recommendations = generate_recommendations_advice(
        investment_goals=risk_tolerance['investment_goals'],
        investment_horizon=risk_tolerance['investment_horizon'],
        risk_tolerance_level=risk_tolerance['risk_tolerance_level'],
        income_stability=risk_tolerance['income_stability'],
        financial_situation=risk_tolerance['financial_situation'],
        investment_experience=risk_tolerance['investment_experience']
    )

    # Pass both risk_tolerance and recommendations to the template
    return render_template('recommendation.html', recommendations=recommendations, risk_tolerance=risk_tolerance)


@app.route('/logout')
def logout():
    # Remove user session data to log out the user
    session.pop('user_id', None)
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message')
    mode = request.form.get('mode', 'stock_terms')  # Default mode is stock_terms

    if mode == 'stock_terms':
        # Get chatbot response
        response = model_1.chatbot_response(user_input)
        
        if "Multiple options found:" in response:
            # Send multiple options to the chatbot UI
            return jsonify({'response': response, 'type': 'options'})
        else:
            # Send single response
            return jsonify({'response': response, 'type': 'response'})
    elif mode == 'stock_price':
        # Get stock price response
        response = model_2.get_stock_price(user_input)
        if response is not None:
            response = f"The stock price for {user_input} is: {response}"
        else:
            response = "Sorry, I couldn't fetch the stock price."
        return jsonify({'response': response, 'type': 'response'})

@app.route('/select_option', methods=['POST'])
def select_option():
    user_input = request.form.get('message')
    selected_number = int(request.form.get('selected_number', 1))

    # Handle user selection
    response = model_1.handle_user_selection(user_input, selected_number)
    return jsonify({'response': response, 'type': 'response'})


if __name__ == '__main__':
    app.run(debug=True)
