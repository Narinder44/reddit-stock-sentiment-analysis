import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime

# Set page configuration
st.set_page_config(
    page_title="Reddit & Stock Price Analyzer",
    page_icon="📈",
    layout="wide",
)

# Title and introduction
st.title("Hype vs. Reality: Can Reddit Predict Stock Prices?")
st.markdown("""
This dashboard analyzes the relationship between Reddit's WallStreetBets sentiment and stock price movements.
Select a stock and date range to explore if social media sentiment can predict market behavior.
""")

# Sidebar for inputs
st.sidebar.header("Settings")

# Add a debug toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Function to load and process Reddit data
@st.cache_data
def load_reddit_data(file_path):
    """Load and process the Reddit WallStreetBets dataset"""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Identify timestamp column
        timestamp_col = None
        for col in ['timestamp', 'created_utc', 'created', 'date']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            st.warning("Could not find a timestamp column in the dataset")
            # Create a placeholder date column
            df['date'] = pd.to_datetime('today')
            return df
        
        # Convert timestamp to datetime based on format
        try:
            if timestamp_col == 'created_utc':
                # Unix timestamp in seconds
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s', errors='coerce')
            else:
                # Standard datetime format
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            # Create consistent date column (just the date part, no time)
            df['date'] = pd.to_datetime(df[timestamp_col]).dt.date
            df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        except Exception as e:
            if debug_mode:
                st.error(f"Error processing dates: {e}")
            # Create fallback date columns
            df['date'] = pd.to_datetime('today')
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df
    except Exception as e:
        st.error(f"Error loading Reddit data: {e}")
        return None

# Function to perform sentiment analysis
@st.cache_data
def analyze_sentiment(df, text_column, stocks=['GME', 'AMC', 'TSLA']):
    """Perform sentiment analysis on Reddit posts"""
    analyzer = SentimentIntensityAnalyzer()
    
    # Process each stock and create a list of daily sentiment
    daily_sentiment_list = []
    
    for stock in stocks:
        try:
            # Filter posts that mention the stock
            stock_posts = df[df[text_column].str.contains(stock, case=False, na=False)].copy()
            
            if stock_posts.empty:
                if debug_mode:
                    st.warning(f"No posts found mentioning {stock}")
                continue
                
            # Apply sentiment analysis
            stock_posts['sentiment'] = stock_posts[text_column].apply(
                lambda x: analyzer.polarity_scores(str(x))['compound'] if pd.notna(x) else np.nan
            )
            
            # Group by date and calculate average sentiment
            daily = stock_posts.groupby('date_str').agg(
                post_count=('sentiment', 'count'),
                avg_sentiment=('sentiment', 'mean')
            ).reset_index()
            
            daily['stock'] = stock
            daily_sentiment_list.append(daily)
                
        except Exception as e:
            st.error(f"Error processing sentiment for {stock}: {e}")
    
    # Combine all stocks
    if daily_sentiment_list:
        return pd.concat(daily_sentiment_list, ignore_index=True)
    else:
        # Create empty DataFrame with correct structure
        return pd.DataFrame(columns=['date_str', 'stock', 'post_count', 'avg_sentiment'])

# Function to get stock data
@st.cache_data
def get_stock_data(stock_symbol, start_date, end_date):
    """Get historical stock price data from Yahoo Finance"""
    try:
        # Add a few days buffer to calculate returns
        buffer_end = pd.to_datetime(end_date) + pd.Timedelta(days=5)
        
        # Download stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=buffer_end)
        
        if stock_data.empty:
            st.warning(f"No stock data found for {stock_symbol} in the selected date range")
            return pd.DataFrame()
        
        # Calculate daily returns
        stock_data['daily_return'] = stock_data['Close'].pct_change() * 100
        stock_data['next_day_return'] = stock_data['daily_return'].shift(-1)
        
        # Reset index to make date a column and format as string for merging
        stock_data = stock_data.reset_index()
        stock_data['date_str'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
        
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {stock_symbol}: {e}")
        return pd.DataFrame()

# Function to manually calculate lag correlations
def calculate_lag_correlations(sentiment_df, stock_df, max_lag=5):
    """Calculate correlations between lagged sentiment and returns"""
    try:
        if sentiment_df.empty or stock_df.empty:
            return []
            
        # Extract date strings and values into simple lists for easier processing
        sentiment_dates = sentiment_df['date_str'].tolist()
        sentiment_values = sentiment_df['avg_sentiment'].tolist()
        
        stock_dates = stock_df['date_str'].tolist()
        stock_returns = stock_df['next_day_return'].tolist()
        
        # Create dictionaries for easier lookups
        sentiment_dict = dict(zip(sentiment_dates, sentiment_values))
        returns_dict = dict(zip(stock_dates, stock_returns))
        
        # Get all dates and sort them
        all_dates = sorted(list(set(sentiment_dates + stock_dates)))
        
        # Create a list of data points (only where we have both sentiment and returns)
        data_points = []
        for date in all_dates:
            if date in sentiment_dict and date in returns_dict:
                data_points.append({
                    'date': date,
                    'sentiment': sentiment_dict[date],
                    'return': returns_dict[date]
                })
        
        # Convert to dataframe and sort by date
        df = pd.DataFrame(data_points)
        
        if len(df) < 2:
            return []
            
        # Calculate correlations for different lags
        correlations = []
        for lag in range(1, min(max_lag + 1, len(df))):
            # Create lagged sentiment
            df['sentiment_lag'] = df['sentiment'].shift(lag)
            
            # Calculate correlation
            valid_data = df.dropna()
            if len(valid_data) >= 2:
                corr = np.corrcoef(valid_data['sentiment_lag'], valid_data['return'])[0, 1]
                correlations.append((lag, corr))
            else:
                correlations.append((lag, 0))
                
        return correlations
    except Exception as e:
        if debug_mode:
            st.error(f"Error in lag correlation: {e}")
        return []

# Main function
def main():
    # File upload for Reddit data
    uploaded_file = st.sidebar.file_uploader("Upload Reddit WallStreetBets Data (CSV)", type=['csv'])
    
    # If no file is uploaded, show instructions
    if not uploaded_file:
        st.info("Please upload the Reddit WallStreetBets dataset to begin analysis.")
        st.markdown("""
        ### How to get the data:
        1. Download the Reddit WallStreetBets dataset from [Kaggle](https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts)
        2. Upload the CSV file using the file uploader in the sidebar
        """)
        return
    
    # Process the uploaded file
    with st.spinner("Processing Reddit data..."):
        reddit_df = load_reddit_data(uploaded_file)
    
    if reddit_df is None or reddit_df.empty:
        st.error("Failed to load Reddit data. Please check the file format.")
        return
    
    # Display basic info about the data
    st.sidebar.success(f"Loaded {len(reddit_df)} Reddit posts")
    
    # Determine text column for sentiment analysis
    text_column = None
    potential_columns = ['title', 'body', 'selftext', 'text', 'content']
    for col in potential_columns:
        if col in reddit_df.columns:
            text_column = col
            break
    
    if text_column is None:
        if 'title' in reddit_df.columns and 'selftext' in reddit_df.columns:
            # Combine title and selftext for sentiment analysis
            reddit_df['combined_text'] = reddit_df['title'].fillna('') + ' ' + reddit_df['selftext'].fillna('')
            text_column = 'combined_text'
        else:
            st.error("Could not identify a text column for sentiment analysis")
            return
    
    # Try to determine date range from the data
    try:
        min_date = pd.to_datetime(reddit_df['date'].min())
        max_date = pd.to_datetime(reddit_df['date'].max())
        
        # Format dates for display
        min_date_str = min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date)
        max_date_str = max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)
        
        st.sidebar.write(f"Data range: {min_date_str} to {max_date_str}")
    except Exception as e:
        if debug_mode:
            st.sidebar.error(f"Error determining date range: {e}")
        min_date = datetime.date(2021, 1, 1)
        max_date = datetime.date(2021, 3, 31)
    
    # Allow user to select a date range
    default_start = min_date if isinstance(min_date, (datetime.date, pd.Timestamp)) else datetime.date(2021, 1, 1)
    default_end = max_date if isinstance(max_date, (datetime.date, pd.Timestamp)) else datetime.date(2021, 3, 31)
    
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", default_end)
    
    # Stock selection
    available_stocks = ['GME', 'AMC', 'TSLA']
    selected_stock = st.sidebar.selectbox("Select Stock", available_stocks)
    
    # Sentiment analysis
    with st.spinner("Performing sentiment analysis..."):
        # Create a filtered dataframe for the date range
        date_mask = (pd.to_datetime(reddit_df['date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(reddit_df['date']) <= pd.to_datetime(end_date))
        filtered_df = reddit_df[date_mask]
        
        if debug_mode:
            st.write(f"Filtered to {len(filtered_df)} posts within date range")
        
        if filtered_df.empty:
            st.warning("No posts found in the selected date range")
            return
            
        sentiment_df = analyze_sentiment(filtered_df, text_column, stocks=[selected_stock])
    
    # Get stock data 
    with st.spinner("Fetching stock data..."):
        stock_data = get_stock_data(selected_stock, start_date, end_date)
    
    # Filter sentiment data for the selected stock
    stock_sentiment = sentiment_df[sentiment_df['stock'] == selected_stock]
    
    if stock_sentiment.empty:
        st.warning(f"No sentiment data found for {selected_stock} in the selected date range")
        return
        
    if stock_data.empty:
        st.warning(f"No stock price data found for {selected_stock} in the selected date range")
        return
    
    # Display header with date range
    st.header(f"{selected_stock} Analysis ({start_date} to {end_date})")
    
    # Calculate metrics safely
    try:
        total_posts = int(stock_sentiment['post_count'].sum())
        post_text = f"{total_posts:,}" if total_posts > 0 else "N/A"
    except:
        post_text = "N/A"
        
    try:
        avg_sentiment = float(stock_sentiment['avg_sentiment'].mean())
        sentiment_text = f"{avg_sentiment:.2f}"
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
    except:
        sentiment_text = "N/A"
        sentiment_label = "Unknown"
        
    try:
        # Simple price change calculation
        first_price = float(stock_data.iloc[0]['Close'])
        last_price = float(stock_data.iloc[-1]['Close'])
        price_change = ((last_price / first_price) - 1) * 100
        price_text = f"{price_change:.1f}%"
    except:
        price_text = "N/A"
        
    # Calculate correlation
    try:
        # Create a simple merged dataset for analysis
        analysis_data = []
        for idx, sentiment_row in stock_sentiment.iterrows():
            date_str = sentiment_row['date_str']
            
            # Find the next day's stock data
            next_day_mask = stock_data['date_str'] > date_str
            if next_day_mask.any():
                next_day_idx = next_day_mask.idxmax()
                next_day_return = stock_data.loc[next_day_idx, 'daily_return']
                
                analysis_data.append({
                    'date': date_str,
                    'sentiment': sentiment_row['avg_sentiment'],
                    'next_day_return': next_day_return
                })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        if len(analysis_df) >= 2:
            correlation = analysis_df['sentiment'].corr(analysis_df['next_day_return'])
            corr_text = f"{correlation:.2f}"
        else:
            corr_text = "N/A"
    except:
        corr_text = "N/A"
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", post_text)
    col2.metric("Average Sentiment", f"{sentiment_text} ({sentiment_label})")
    col3.metric("Price Change", price_text)
    col4.metric("Sentiment-Return Correlation", corr_text)
    
    # Create plots
    st.subheader("Reddit Sentiment vs. Stock Price")
    
    # Set up the figure with a dark background for a better look
    plt.style.use('dark_background')
    
    # Create the figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 2, 1]})
    
    # Plot 1: Stock Price
    stock_dates = pd.to_datetime(stock_data['Date'])
    axs[0].plot(stock_dates, stock_data['Close'], color='#1F77B4', linewidth=2)
    axs[0].set_ylabel('Price ($)')
    axs[0].set_title(f'{selected_stock} Stock Price')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Sentiment
    sentiment_dates = pd.to_datetime(stock_sentiment['date_str'])
    axs[1].plot(sentiment_dates, stock_sentiment['avg_sentiment'], color='#FF7F0E', linewidth=2)
    axs[1].set_ylabel('Sentiment')
    axs[1].set_title('Reddit Sentiment')
    axs[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Post Volume
    axs[2].bar(sentiment_dates, stock_sentiment['post_count'], color='#2CA02C', alpha=0.7)
    axs[2].set_ylabel('Post Count')
    axs[2].set_title('Daily Post Volume')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    # Format dates
    fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    st.pyplot(fig)
    
    # Calculate lag correlation (completely rewritten to avoid all errors)
    lag_results = calculate_lag_correlations(stock_sentiment, stock_data)
    
    if lag_results:
        st.subheader("Sentiment Lag Analysis")
        
        # Create a new figure for the lag analysis
        fig_lag = plt.figure(figsize=(10, 6))
        ax = fig_lag.add_subplot(111)
        
        # Extract data for plotting
        lags = [lag[0] for lag in lag_results]
        corrs = [lag[1] for lag in lag_results]
        
        # Create bar chart
        bars = ax.bar(lags, corrs, color='#1E88E5')
        
        # Customize the chart
        ax.set_xlabel('Lag (Days)')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlation: Reddit Sentiment vs. Future Stock Returns')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set the x-axis to show integers only
        ax.set_xticks(lags)
        
        # Show plot
        st.pyplot(fig_lag)
        
        # Find best lag
        if lag_results:
            best_lag_idx = np.abs(corrs).argmax()
            best_lag = lags[best_lag_idx]
            best_corr = corrs[best_lag_idx]
        else:
            best_lag = "N/A"
            best_corr = 0
        
        # Show key findings
        st.subheader("Analysis Interpretation")
        
        st.markdown(f"""
        #### Key Findings:
        
        1. **Sentiment-Price Relationship**: The correlation between Reddit sentiment and next-day returns is **{corr_text}**.
        
        2. **Optimal Prediction Lag**: The strongest relationship occurs with a lag of **{best_lag} days** 
           (correlation: **{best_corr:.2f}**).
        
        3. **Post Volume**: There were **{post_text}** posts about {selected_stock} during this period.
        """)
        
        # Provide interpretation
        if abs(float(corr_text) if corr_text != "N/A" else 0) < 0.2:
            st.info("The relationship between Reddit sentiment and stock returns is weak for this time period.")
        elif abs(float(corr_text) if corr_text != "N/A" else 0) < 0.5:
            st.info("There appears to be a moderate relationship between Reddit sentiment and stock returns.")
        else:
            if float(corr_text) > 0:
                st.success("There appears to be a strong positive relationship between Reddit sentiment and future stock returns.")
            else:
                st.warning("There appears to be a strong negative relationship between Reddit sentiment and future stock returns.")
    else:
        # Even if we can't calculate lag correlations, provide an explanation
        st.subheader("Lag Analysis")
        st.info("""
        Lag analysis examines whether Reddit sentiment on one day predicts stock price movements in future days.
        For example, a 1-day lag correlation measures how today's sentiment relates to tomorrow's returns,
        while a 3-day lag measures how today's sentiment might predict returns three days later.
        
        There weren't enough matching data points to calculate meaningful lag correlations for this specific date range.
        Try selecting a wider date range or a stock with more consistent Reddit activity.
        """)
    
    # Sentiment vs Returns Scatter Plot
    st.subheader("Sentiment vs. Next-Day Returns")
    
    try:
        if len(analysis_df) >= 2:
            # Create scatter plot
            fig_scatter = plt.figure(figsize=(10, 6))
            ax = fig_scatter.add_subplot(111)
            
            # Plot scatter points
            ax.scatter(analysis_df['sentiment'], analysis_df['next_day_return'], 
                     color='#1E88E5', alpha=0.7, s=50)
            
            # Add a trend line using numpy's polyfit
            if len(analysis_df) > 1:
                x = analysis_df['sentiment']
                y = analysis_df['next_day_return']
                
                # Simple linear regression
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = m * x_line + b
                
                # Plot trend line
                ax.plot(x_line, y_line, color='#FF7F0E', linestyle='--', linewidth=2)
                
                # Add equation text
                equation = f"y = {m:.2f}x + {b:.2f}"
                ax.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
                          color='white', fontsize=12)
            
            # Customize chart
            ax.set_xlabel('Sentiment Score')
            ax.set_ylabel('Next-Day Return (%)')
            ax.set_title('Sentiment vs. Next-Day Returns')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Show plot
            st.pyplot(fig_scatter)
        else:
            st.warning("Not enough data points to create a scatter plot")
    except Exception as e:
        if debug_mode:
            st.error(f"Error creating scatter plot: {e}")
        st.warning("Could not create scatter plot due to data issues")
    
    # Show raw data
    with st.expander("View Raw Data"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Data")
            st.dataframe(stock_sentiment)
        
        with col2:
            st.subheader("Stock Price Data")
            st.dataframe(stock_data)
    
    # Answer about real stock data
    st.subheader("About the Data")
    st.markdown("""
    **Yes, this dashboard uses real stock data** fetched directly from Yahoo Finance through the yfinance library.
    The stock prices, returns, and volumes are actual historical market data.
    
    The sentiment analysis is performed on real Reddit posts from the WallStreetBets subreddit that you uploaded.
    The dashboard calculates the sentiment scores using the VADER (Valence Aware Dictionary and Sentiment Reasoner) algorithm,
    which is specifically tuned for social media content.
    
    ### What is Lag Correlation Analysis?
    
    Lag correlation analysis examines whether Reddit sentiment on one day predicts stock price movements on future days.
    For example:
    
    - A 1-day lag measures if today's sentiment predicts tomorrow's price change
    - A 3-day lag measures if today's sentiment predicts the price change three days later
    - A 5-day lag looks at whether sentiment predicts price movements in a trading week
    
    The bar chart above shows correlation coefficients for different lag periods. A stronger correlation (closer to +1 or -1)
    suggests Reddit sentiment might have predictive power for future stock returns.
    """)

if __name__ == "__main__":
    main()