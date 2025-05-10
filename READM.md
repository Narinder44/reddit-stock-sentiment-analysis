# Hype vs. Reality: Can Reddit Predict Stock Prices?

A Streamlit dashboard that analyzes the relationship between Reddit's WallStreetBets sentiment and stock price movements.


## Project Overview

This project explores whether sentiment from Reddit's WallStreetBets can predict short-term price changes for popular stocks like GME, AMC, and TSLA. By combining Natural Language Processing (NLP), time series analysis, and data visualization, this dashboard aims to uncover patterns that link online hype with financial reality.

### Research Questions

1. **Does Reddit sentiment correlate with next-day price changes?**
2. **Is there a lag effect between sentiment spikes and stock price movement?**
3. **Are certain platforms more predictive than others?** (Note: We were unable to analyze Twitter data due to API paywall and access limitations)

## Key Findings

Based on our analysis of WallStreetBets Reddit data:

1. **Reddit Sentiment and Next-Day Returns**: There is a weak negative correlation (approximately -0.11) between Reddit sentiment and next-day stock returns for GME. This suggests a contrarian relationship where positive sentiment often precedes slight price declines.

2. **Lag Effect Confirmed**: The analysis shows evidence of a lag effect, with the strongest relationship occurring at a 4-day lag (correlation: -0.06). This indicates Reddit sentiment may take several days to fully impact stock prices.

3. **Platform Comparison**: We were unable to compare Reddit with other platforms like Twitter due to data access limitations. Future research could address this question with proper API access.

The weak nature of these correlations suggests that while social media sentiment provides some signal about future stock movements, it's likely just one of many factors influencing price changes.

## Features

- **Sentiment Analysis**: Uses VADER (Valence Aware Dictionary and Sentiment Reasoner) to analyze Reddit post sentiment
- **Real-Time Stock Data**: Fetches historical stock data from Yahoo Finance
- **Correlation Analysis**: Measures the relationship between Reddit sentiment and stock returns
- **Lag Analysis**: Examines whether sentiment predicts price movements over different time periods (1-5 days)
- **Interactive Visualization**: Time series charts of stock prices, sentiment, and post volume
- **Customizable Date Ranges**: Select specific time periods for analysis
- **Multiple Stock Support**: Analyze GME, AMC, or TSLA

## Requirements

- Python 3.6+
- Required packages:
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - yfinance
  - vaderSentiment

## Installation and Usage

1. Clone the repository or download the script
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download the Reddit WallStreetBets dataset from [Kaggle](https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts)
4. Run the Streamlit app:

```bash
streamlit run sentiment_stock_analyzer.py
```

5. Use the sidebar to:
   - Upload the Reddit WallStreetBets CSV file
   - Select a date range for analysis
   - Choose a stock (GME, AMC, or TSLA)

## Dashboard Components

### 1. Key Metrics
- **Total Posts**: Number of Reddit posts mentioning the selected stock
- **Average Sentiment**: Mean sentiment score with a label (Positive/Neutral/Negative)
- **Price Change**: Percentage change in stock price during the selected period
- **Sentiment-Return Correlation**: Correlation coefficient between sentiment and next-day returns

### 2. Visualizations
- **Stock Price Chart**: Historical price movements for the selected stock
- **Sentiment Timeline**: Daily average sentiment score from Reddit posts
- **Post Volume Chart**: Number of Reddit posts per day
- **Lag Correlation Chart**: Correlation between sentiment and future returns over different time lags
- **Sentiment vs. Returns Scatter Plot**: Relationship between sentiment scores and next-day returns

## Methodology

Our research followed these methodological steps:

### 1. Data Collection
- **Reddit Data**: We used the WallStreetBets dataset from Kaggle containing over 50,000 Reddit posts
- **Stock Price Data**: We fetched historical stock price data using the Yahoo Finance API through the yfinance Python package
- **Timeframe**: We focused on posts between 2020-09-29 and 2021-08-16, a period that includes the January 2021 "meme stock" phenomenon

### 2. Sentiment Analysis
- **Algorithm**: We utilized VADER (Valence Aware Dictionary and Sentiment Reasoner), a lexicon and rule-based sentiment analysis tool specifically attuned to social media content
- **Sentiment Scoring**: Each post was assigned a compound sentiment score ranging from -1 (extremely negative) to +1 (extremely positive)
- **Aggregation**: We calculated daily average sentiment scores for each stock by averaging all posts mentioning the target ticker symbol on each day

### 3. Correlation Analysis
- **Next-Day Returns**: We calculated the correlation between daily sentiment scores and next-day price returns
- **Lag Analysis**: We examined correlations at various lag periods (1-5 days) to identify potential delayed effects
- **Statistical Methods**: We used Pearson correlation coefficients to quantify the strength and direction of relationships

## Challenges and Limitations

During our research, we encountered several significant challenges:

### 1. Data Access Limitations
- **Twitter Data Inaccessibility**: We were unable to obtain Twitter data as initially planned due to API access restrictions and the paywall for historical data
- **Platform Comparison**: Without Twitter data, we couldn't address our third research question about which platform is more predictive

### 2. Technical Challenges
- **Data Processing**: We encountered several technical errors during development, including merge errors between DataFrames, type errors when formatting metrics, and module dependency issues
- **Computational Limitations**: Processing large volumes of text data for sentiment analysis was computationally intensive

### 3. Methodological Limitations
- **Sentiment Analysis Accuracy**: VADER, while effective, may not capture all nuances of financial discussions
- **Causality vs. Correlation**: Our analysis identified correlations but cannot definitively establish causality
- **Confounding Variables**: Other factors beyond sentiment were not fully controlled for

## Understanding Lag Analysis

Lag correlation analysis examines whether Reddit sentiment on one day predicts stock price movements on future days:

- A 1-day lag measures if today's sentiment predicts tomorrow's price change
- A 3-day lag measures if today's sentiment predicts the price change three days later
- A 5-day lag looks at whether sentiment predicts price movements in a trading week

The bar chart shows correlation coefficients for different lag periods. A stronger correlation (closer to +1 or -1) suggests Reddit sentiment might have predictive power for future stock returns.

## Conclusion

Our analysis suggests that Reddit sentiment does have a weak correlation with stock price movements, particularly showing a contrarian relationship where positive sentiment often precedes negative returns. The lag effect indicates that sentiment may take several days to fully impact stock prices.

For traders and investors, Reddit sentiment should be considered as a supplementary indicator rather than a primary decision-making tool. The relationship, while statistically detectable, is not strong enough to be the sole basis for investment decisions.

Future research that incorporates multiple data sources and more sophisticated analytical techniques could provide deeper insights into how social media sentiment impacts stock price movements.

