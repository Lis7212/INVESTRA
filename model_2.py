import requests
from bs4 import BeautifulSoup
import time

# Dictionary mapping company names to ticker symbols
company_to_ticker = {
    'reliance': 'RELIANCE',
    'tcs': 'TCS',
    'infosys': 'INFY',
    'hdfc bank': 'HDFCBANK',
    'icici bank': 'ICICIBANK',
    'kotak mahindra bank': 'KOTAKBANK',
    'sbi': 'SBIN',
    'itc': 'ITC',
    'bajaj finance': 'BAJFINANCE',
    'adani enterprises': 'ADANIENT'
}

def get_stock_price(company_name):
    # Get the ticker symbol for the company
    ticker = company_to_ticker.get(company_name.lower())
    if not ticker:
        print("Company not found.")
        return
    
    url = f"https://www.google.com/finance/quote/{ticker}:NSE?hl=en"
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        class1 = "YMlKec fxKbKc"
        price_str = soup.find(class_=class1).text.strip()
        price = float(price_str[1:].replace(",", ""))
        return price
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# # User input
# company_name = input("Enter company name: ")
# price = get_stock_price(company_name)
# if price is not None:
#     print(f"The stock price for {company_name} is: {price}")

