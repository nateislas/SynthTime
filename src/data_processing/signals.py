import yfinance as yf
import pandas as pd

def get_sp500_market_regime(start_date="2000-01-01", end_date="2024-12-31"):
    """
    Fetches historical S&P 500 index data from Yahoo Finance and returns a simplified DataFrame.

    Args:
        start_date (str): Start date for fetching data (YYYY-MM-DD).
        end_date (str): End date for fetching data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame with columns [Date, Market Regime, Monthly Return]
    """
    # Fetch S&P 500 data (symbol: ^GSPC)
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", auto_adjust=True)

    # Ensure a single-level index
    sp500 = sp500.reset_index()

    # Convert Date to datetime format
    sp500["Date"] = pd.to_datetime(sp500["Date"])


    # Calculate monthly returns
    sp500["Monthly Return"] = sp500["Close"].pct_change(periods=21)  # ~21 trading days in a month

    # Define Market Regime Based on Monthly Returns
    def classify_regime(return_value):
        if return_value > 0.05:  # More than +5% in a month
            return "Bull"
        elif return_value < -0.05:  # Less than -5% in a month
            return "Bear"
        else:
            return "Neutral"

    # Apply classification
    sp500["Market Regime"] = sp500["Monthly Return"].apply(classify_regime)

    # Ensure there are no multi-index levels
    sp500.columns = sp500.columns.get_level_values(0)  # Flatten multi-index if it exists
    
    sp500["Date"] = sp500["Date"].dt.strftime('%Y-%m-%d')

    # Return only necessary columns
    return sp500[["Date", "Market Regime", "Monthly Return"]]
