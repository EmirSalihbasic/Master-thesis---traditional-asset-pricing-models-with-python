import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt

ticker_bank = 'JPM'  # Ticker of the bank JPMorgan Chase
ticker_market = '^GSPC'  # Ticker of the index S&P 500 index

# Initialize lists to store beta and expected return values
years = ['2019', '2020', '2021', '2022', '2023']
expected_returns = []

try:
    # Function to fetch 10-year Treasury yield from Yahoo Finance
    def get_10yr_yield(year):
        data = yf.download('^TNX', start=f'{year}-01-01', end=f'{year}-12-31')['Adj Close']
        return data.dropna().mean() / 100.0  # Convert to decimal from percentage

    for year in years:
        # Download historical data for JPMorgan Chase and S&P 500 for each year
        bank_data = yf.download(ticker_bank, start=f'{year}-01-01', end=f'{year}-12-31')['Adj Close']
        market_data = yf.download(ticker_market, start=f'{year}-01-01', end=f'{year}-12-31')['Adj Close']

        # Calculate daily returns for the year
        returns_bank = bank_data.pct_change().dropna()
        returns_market = market_data.pct_change().dropna()

        # Calculate beta using linear regression
        beta, _, _, _, _ = stats.linregress(returns_market.values, returns_bank.values)

        # Assume risk-free rate (e.g., 10-year Treasury yield)
        risk_free_rate = get_10yr_yield(year)

        # Calculate expected return using CAPM
        expected_return = risk_free_rate + beta * (returns_market.mean() - risk_free_rate)
        expected_returns.append(expected_return)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(years, [100*x for x in expected_returns], marker='o', linestyle='-', color='b', label='Expected Return')

    plt.xlabel('Year')
    plt.ylabel('Expected Return (%)')
    plt.title('CAPM-derived Expected Return for JPMorgan Chase')
    plt.grid(False)  # Remove gridlines inside the plot

    # Display values on the plot
    for i, v in enumerate(expected_returns):
        plt.text(years[i], 100*v, f'{100*v:.2f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.ylim(min(expected_returns) * 100 - 1, max(expected_returns) * 100 + 1)  # Adjust ylim to include negative values
    plt.axhline(y=0, color='black', linestyle='--')  # Add horizontal line at y=0
    plt.show()

except Exception as e:
    print(f"Error occurred: {e}")









