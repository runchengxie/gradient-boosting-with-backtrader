# ==== Step 1: Understand the Problem ====

# This project aims to leverage the capabilities of gradient boosting to predict a company's stock price. 
# The methodology will involve a 7-step model building process and a funneling approach to feature selection


# ==== Step 2: Data Collection and Preparation ====

# ---- Import necessary libraries ----
import tushare as ts
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ---- Configuration ----
CACHE_DIR = 'downloaded_raw_data'
TICKER = '000300.SH'  # Major Chinese broad market index, CSI 300
START_DATE = '2004-12-31'
END_DATE = '2025-05-20'
TEST_SIZE = 0.2 # 20% for test set
RANDOM_STATE = 42
# Threshold for defining 'positive move'
# This needs to be adjusted based on the chosen ticker's volatility and characteristics.
POSITIVE_MOVE_THRESHOLD = 0.2


# ---- Data Collection ----
print(f"--- Fetching data for {TICKER} from {START_DATE} to {END_DATE} using Tushare ---")
try:
    raw_data = None

    # Create cache directory if it doesn't exist
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    cache_file_name = f"{TICKER}_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.parquet"
    cache_file_path = os.path.join(CACHE_DIR, cache_file_name)

    if os.path.exists(cache_file_path):
        print(f"Loading data from cache: {cache_file_path}")
        raw_data = pd.read_parquet(cache_file_path)
        print(f"Successfully loaded {len(raw_data)} data points from cache.")
    else:
        print(f"Cache not found at {cache_file_path}. Fetching from Tushare API.")
        api_key = os.environ.get('TUSHARE_API_KEY')
        if not api_key:
            raise ValueError("TUSHARE_API_KEY environment variable not set. Please ensure it is configured.")
        ts.set_token(api_key)
        pro = ts.pro_api()

        # Check the validity of the tushare api service
        print("--- Querying Tushare User Points ---")
        try:
            user_points_df = pro.user(token=api_key)
            print("Tushare User Points Information:")
            print(user_points_df)
        except Exception as e:
            print(f"Error querying Tushare user points: {e}")
        print("-"*50 + "\n")

        start_date_ts = pd.to_datetime(START_DATE).strftime('%Y%m%d')
        end_date_ts = pd.to_datetime(END_DATE).strftime('%Y%m%d')

        # Using idx_factor_pro to fetch data along with technical factors
        print(f"Attempting to fetch data and technical factors for index {TICKER} using pro.idx_factor_pro.")
        
        factor_fields = [
            'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_change', 'ma_bfq_5', 'ma_bfq_20', 'ma_bfq_60', 'ema_bfq_10', 'ema_bfq_60', 'macd_dif_bfq', 'macd_dea_bfq', 'macd_bfq', 'dmi_adx_bfq', 'dmi_pdi_bfq', 'dmi_mdi_bfq', 'roc_bfq', 'mtm_bfq', 'updays', 'downdays', 'atr_bfq', 'boll_upper_bfq', 'boll_lower_bfq', 'rsi_bfq_12', 'cci_bfq', 'wr_bfq', 'kdj_k_bfq', 'kdj_d_bfq', 'obv_bfq', 'mfi_bfq'
        ]
        fields_str = ",".join(factor_fields)

        data = pro.idx_factor_pro(ts_code=TICKER, start_date=start_date_ts, end_date=end_date_ts, fields=fields_str)

        if data.empty:
            raise ValueError(f"No data found for ticker {TICKER} (from {start_date_ts} to {end_date_ts}) using Tushare idx_factor_pro. Check ticker symbol, API limits, and ensure it's a supported index.")

        data['trade_date'] = pd.to_datetime(data['trade_date'])
        data.set_index('trade_date', inplace=True)
        data.sort_index(inplace=True)
        
        # Check for essential fields from idx_factor_pro
        required_columns = ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_change', 'ma_bfq_5']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns after Tushare idx_factor_pro fetch: {missing_cols}. Available columns: {data.columns.tolist()}")
        
        raw_data = data.copy()

        print(f"Successfully fetched {len(raw_data)} data points with technical factors for {TICKER} using Tushare.")
        # Save data to cache
        try:
            raw_data.to_parquet(cache_file_path)
            print(f"Data saved to cache: {cache_file_path}")
        except Exception as e_cache_save:
            print(f"Error saving data to cache {cache_file_path}: {e_cache_save}")
        print("-"*50 + "\n")
except ValueError as e:
    print(f"Error fetching data: {e}")
    exit()

print("--- Data collection phase completed successfully. ---")
print("-"*50 + "\n")
print("--- Data Basic Information ---")
print(f"Total number of samples: {len(raw_data)}")
start_date = raw_data.index.min()
end_date = raw_data.index.max()
years_covered = (end_date - start_date).days / 365.25
print(f"Date range: from {start_date} to {end_date}")
print(f"Total trading period covered: {years_covered:.2f} years")
print("-"*50 + "\n")


# ==== Step 3: Explore and Visualize Data ====

# ---- Basic Data Inspection ----
print("\n--- First Few Rows of the Data ---")
print(raw_data.head())
print("-"*50 + "\n")
print("\n--- Last Few Rows of the Data ---")
print(raw_data.tail())
print("-"*50 + "\n")
print("\n--- Data Types ---")
print(raw_data.dtypes)
print("-"*50 + "\n")
print("\n--- Missing Values ---")
print(raw_data.isnull().sum())
print("-"*50 + "\n")

# ---- Descriptive Statistics (for numerical features) ----
print("\n--- Descriptive Statistics for Numerical Features ---")
print(raw_data.describe())
print("-"*50 + "\n")

# ---- Initial Visualizations ----
plt.figure(figsize=(12, 10)) # Adjust figsize to fit 2x2 layout

# 1. Histogram of returns
plt.subplot(2, 2, 1) # 2 rows, 2 columns, 1st subplot
if 'pct_change' in raw_data.columns and pd.api.types.is_numeric_dtype(raw_data['pct_change']):
    plt.hist(raw_data['pct_change'].dropna(), bins=50, density=True, alpha=0.7)
else:
    plt.text(0.5, 0.5, "'pct_change' column not found or not numeric", ha='center', va='center')
plt.title('Distribution of Daily Returns')
plt.xlabel('Returns (%)')
plt.ylabel('Frequency')

# 2. Histogram of volume
plt.subplot(2, 2, 2) # 2 rows, 2 columns, 2nd subplot
if 'vol' in raw_data.columns and pd.api.types.is_numeric_dtype(raw_data['vol']):
    plt.hist(raw_data['vol'].dropna(), bins=50, density=True, alpha=0.7)
else:
    plt.text(0.5, 0.5, "'vol' column not found or not numeric", ha='center', va='center')
plt.title('Distribution of Trading Volume')
plt.xlabel('Volume')
plt.ylabel('Frequency')

# 3. Box plot for returns distribution
plt.subplot(2, 2, 3) # 2 rows, 2 columns, 3rd subplot
if 'pct_change' in raw_data.columns and pd.api.types.is_numeric_dtype(raw_data['pct_change']):
    plt.boxplot(raw_data['pct_change'].dropna(), whis=1.5)
else:
    plt.text(0.5, 0.5, "'pct_change' column not found or not numeric", ha='center', va='center')
plt.title('Returns Distribution Box Plot')
plt.ylabel('Returns (%)')

# 4. Box plot for volume distribution
plt.subplot(2, 2, 4) # 2 rows, 2 columns, 4th subplot
if 'vol' in raw_data.columns and pd.api.types.is_numeric_dtype(raw_data['vol']):
    plt.boxplot(raw_data['vol'].dropna(), whis=1.5)
else:
    plt.text(0.5, 0.5, "'vol' column not found or not numeric", ha='center', va='center')
plt.title('Volume Distribution Box Plot')
plt.ylabel('Volume')

plt.tight_layout() # Automatically adjust subplot parameters for a tight layout
plt.show() # Display single figure containing all four subplots

# ---- Time Series Specific Visualizations ----
# Create a figure with 2 subplots
plt.figure(figsize=(15, 8))

# Plot closing price over time
plt.subplot(2, 1, 1)
plt.plot(raw_data.index, raw_data['close'], color='blue')
plt.title('Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

# Plot trading volume over time 
plt.subplot(2, 1, 2)
plt.plot(raw_data.index, raw_data['vol'], color='red')
plt.title('Trading Volume Over Time')
plt.xlabel('Date') 
plt.ylabel('Volume')
plt.grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()

# ---- Correlation Matrix Heatmap ----
# Calculate the correlation matrix
corr_matrix = raw_data.corr()

# Create a mask to hide the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(15, 12)) # Slightly larger figure size

# Draw the heatmap with the mask and all annotations
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", square=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Correlation Matrix Heatmap (All Correlations)')
plt.show()


# ====Step 4: Data Cleaning ====
print("--- Preprocessing Data ---")
processed_data = raw_data.copy()

# Ensure 'pct_change' column exists as it will be used directly
if 'pct_change' not in processed_data.columns:
    raise ValueError("Column 'pct_change' not found in fetched data from idx_factor_pro. This column is essential.")

# Sanity Check: Prices should be positive
price_columns = ['open', 'high', 'low', 'close']
for col in price_columns:
    if (processed_data[col] <= 0).any():
        raise ValueError(f"Column '{col}' contains non-positive values. Prices must be positive.")

# Sanity Check: OHLC logical relationship
if not (processed_data['high'] >= processed_data['open']).all():
    raise ValueError("Data integrity issue: 'high' price is less than 'open' price for some entries.")
if not (processed_data['high'] >= processed_data['close']).all():
    raise ValueError("Data integrity issue: 'high' price is less than 'close' price for some entries.")
if not (processed_data['low'] <= processed_data['open']).all():
    raise ValueError("Data integrity issue: 'low' price is greater than 'open' price for some entries.")
if not (processed_data['low'] <= processed_data['close']).all():
    raise ValueError("Data integrity issue: 'low' price is greater than 'close' price for some entries.")

# Checking the NaN values count and percentages for each column
nan_counts = processed_data.isna().sum()
nan_percentages = (nan_counts / len(processed_data) * 100).round(2)

print("--- NaN Statistics by Column ---")
for col, count in nan_counts.items():
    percentage = nan_percentages[col]
    
    if count > 0: # Only process columns with NaNs
        first_valid_idx = processed_data[col].first_valid_index()
        
        if first_valid_idx is not None:
            # Get the integer position of the first valid index
            idx_loc = processed_data[col].index.get_loc(first_valid_idx)
            
            if idx_loc > 0:
                # This means there are leading NaNs. The count of leading NaNs is idx_loc.
                # By definition of first_valid_idx, all processed_data[col].iloc[:idx_loc] are NaN.
                print(f"{col}: {count} NaNs ({percentage}%) - {idx_loc} consecutive NaN(s) from start, likely due to normal calculation window for indicators.")
            else: # idx_loc == 0
                # The series starts with a non-NaN value.
                # Any NaNs present (count > 0) must be intermittent or trailing.
                print(f"{col}: {count} NaNs ({percentage}%) - NaNs present but not at the start. Data quality needs attention for intermittent NaNs.")
        else: # All values in the column are NaN
            print(f"{col}: {count} NaNs ({percentage}%) - All values are NaN. This column might be unusable or data source issue.")
    else: # No NaNs in the column
        print(f"{col}: {count} NaNs ({percentage}%)")

print(f"\nTotal rows: {len(processed_data)}")
rows_with_nan = processed_data.isna().any(axis=1).sum()
nan_row_percentage = (rows_with_nan / len(processed_data) * 100).round(2)
print(f"Rows containing NaN: {rows_with_nan} ({nan_row_percentage}%)")
print("-"*50 + "\n")

# Remove rows with NaN values
processed_data.dropna(inplace=True)

print(f"Data shape after initial preprocessing: {processed_data.shape}")
print("-"*50 + "\n")


# ==== Step 5: Data Transformation & Feature Engineering ====

# ---- 1. Mathematical/statistical transformations ----

print("\n--- Applying Feature Transformations ---")
# Apply log transformation to 'volume' and 'amount'
# np.log1p(x) computes log(1+x), which is suitable for data that might contain zeros.
if 'vol' in processed_data.columns:
    processed_data['vol_log'] = np.log1p(processed_data['vol'])
    processed_data.drop(columns=['vol'], inplace=True)
    print("Applied log1p transformation to 'vol', created 'vol_log', and dropped original 'vol'.")
else:
    print("Column 'vol' not found for log transformation.")

if 'amount' in processed_data.columns:
    processed_data['amount_log'] = np.log1p(processed_data['amount'])
    processed_data.drop(columns=['amount'], inplace=True)
    print("Applied log1p transformation to 'amount', created 'amount_log', and dropped original 'amount'.")
else:
    print("Column 'amount' not found for log transformation.")

# Apply differencing to 'OBV'
if 'obv_bfq' in processed_data.columns:
    processed_data['obv_bfq_diff'] = processed_data['obv_bfq'].diff()
    processed_data.drop(columns=['obv_bfq'], inplace=True) # Drop original obv_bfq
    print("Applied differencing to 'obv_bfq', created 'obv_bfq_diff', and dropped original 'obv_bfq'.")
else:
    print("Column 'obv_bfq' not found for differencing.")

# Handle any NaNs introduced by transformations (e.g., from .diff() which creates NaN for the first row)
print("Handling NaNs after transformations...")
original_rows_before_transform_dropna = len(processed_data)
processed_data.dropna(inplace=True)
rows_dropped_by_transform_dropna = original_rows_before_transform_dropna - len(processed_data)
if rows_dropped_by_transform_dropna > 0:
    print(f"Dropped {rows_dropped_by_transform_dropna} additional rows due to NaNs from feature transformations.")
else:
    print("No additional rows dropped due to NaNs from feature transformations.")
print(f"Data shape after feature transformations and NaN handling: {processed_data.shape}")
print("-"*50 + "\n")

# ---- 2. Interaction features ----

print("\n--- Creating Interaction Features ---")
interaction_feature_cols = []

# 1. Interaction between Trend and Momentum/Overbought-Oversold
if 'ma_bfq_20' in processed_data.columns and 'rsi_bfq_12' in processed_data.columns:
    processed_data['inter_ma20_rsi12'] = processed_data['ma_bfq_20'] * processed_data['rsi_bfq_12']
    interaction_feature_cols.append('inter_ma20_rsi12')
    print("Created interaction feature: inter_ma20_rsi12 (ma_bfq_20 * rsi_bfq_12)")
else:
    print("Skipping inter_ma20_rsi12: ma_bfq_20 or rsi_bfq_12 not found.")

# 2. Interaction between Trend and Volatility
if 'ma_bfq_20' in processed_data.columns and 'atr_bfq' in processed_data.columns:
    # Handle potential division by zero if atr_bfq is 0
    processed_data['inter_ma20_div_atr'] = processed_data['ma_bfq_20'] / (processed_data['atr_bfq'] + 1e-6) # Add small epsilon to avoid division by zero
    interaction_feature_cols.append('inter_ma20_div_atr')
    print("Created interaction feature: inter_ma20_div_atr (ma_bfq_20 / atr_bfq)")
else:
    print("Skipping inter_ma20_div_atr: ma_bfq_20 or atr_bfq not found.")

# 3. Interaction between Momentum/Overbought-Oversold Indicators
if 'rsi_bfq_12' in processed_data.columns and 'mfi_bfq' in processed_data.columns:
    processed_data['inter_rsi12_mfi'] = processed_data['rsi_bfq_12'] * processed_data['mfi_bfq']
    interaction_feature_cols.append('inter_rsi12_mfi')
    print("Created interaction feature: inter_rsi12_mfi (rsi_bfq_12 * mfi_bfq)")
else:
    print("Skipping inter_rsi12_mfi: rsi_bfq_12 or mfi_bfq not found.")

# 4. Combination of Specific Patterns
if 'close' in processed_data.columns and 'boll_lower_bfq' in processed_data.columns and 'rsi_bfq_12' in processed_data.columns:
    processed_data['inter_boll_rsi_pattern'] = (processed_data['close'] - processed_data['boll_lower_bfq']) * (processed_data['rsi_bfq_12'] < 30).astype(int)
    interaction_feature_cols.append('inter_boll_rsi_pattern')
    print("Created interaction feature: inter_boll_rsi_pattern ((close - boll_lower_bfq) * (rsi_bfq_12 < 30))")
else:
    print("Skipping inter_boll_rsi_pattern: close, boll_lower_bfq, or rsi_bfq_12 not found.")

# Handle any NaNs introduced by interaction features (though direct arithmetic operations on existing non-NaN columns shouldn't introduce new NaNs unless division by zero occurs and isn't handled)
print("Checking for NaNs after creating interaction features...")
if processed_data[interaction_feature_cols].isnull().any().any():
    original_rows_before_interact_dropna = len(processed_data)
    processed_data.dropna(subset=interaction_feature_cols, inplace=True)
    rows_dropped_by_interact_dropna = original_rows_before_interact_dropna - len(processed_data)
    if rows_dropped_by_interact_dropna > 0:
        print(f"Dropped {rows_dropped_by_interact_dropna} additional rows due to NaNs from interaction feature creation.")
else:
    print("No NaNs found in new interaction features or no rows dropped.")

print(f"Data shape after interaction feature engineering: {processed_data.shape}")
print("-"*50 + "\n")

# ---- 3. Model-based features ----

# --- Defining Target Variable ---
print("--- Defining Target Variable (Predicting NEXT DAY's move) ---")
data_with_target = processed_data.copy()

# IMPORTANT: Shift pct_change to predict the NEXT day's move
# 'pct_change_next_day' for row T will be the pct_change of day T+1
data_with_target['pct_change_next_day'] = data_with_target['pct_change'].shift(-1)

# Drop the last row since it will have NaN for 'pct_change_next_day'
data_with_target.dropna(subset=['pct_change_next_day'], inplace=True)

# Define Target based on the next day's percent change
# POSITIVE_MOVE_THRESHOLD is 0.2, meaning next day's pct_change > 0.2% is considered a positive move (Target=1).
data_with_target['Target'] = np.where(data_with_target['pct_change_next_day'] > POSITIVE_MOVE_THRESHOLD, 1, 0)

print(f"Target variable distribution:\n{data_with_target['Target'].value_counts(normalize=True)}")
print("-"*50 + "\n")

# --- Prepare X_full and y_full ---
# Features are from the CURRENT day (T), Target is for the NEXT day (T+1).
# 'pct_change' (current day's) can remain as a feature if desired, as it's known at time T.
# 'pct_change_next_day' is used for target, so exclude it from features.
# OHLCV data are from the current day T.
potential_features = [col for col in data_with_target.columns if col not in [
    'Target',
    'pct_change_next_day', # This was used to create Target
    # 'pct_change' # Decide if current day's pct_change should be a feature.
                   # If kept, it represents "momentum from today".
                   # In theory, the technical indicators already capture this, it might be redundant.
                   # For now, let's keep it unless it proves to be too dominant or causes issues.
    # 'close', 'open', 'high', 'low', # These are raw price levels.
    # Many indicators are derived from them. Keeping them might be okay,
    # but often derived indicators (MAs, RSI etc.) are more useful.
    # Let's remove them here as the Tushare factors likely capture their essence.
    'close', 'open', 'high', 'low'
]]

X_full = data_with_target[potential_features].copy()
y_full = data_with_target['Target'].copy()

# Ensure indices are aligned (already handled by previous dropna and sequential operations)
if X_full.empty or len(X_full) != len(y_full):
    print("Error: Feature matrix X_full is empty or its length doesn't match target y_full after preprocessing.")
    print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
    print(f"Columns in X_full: {X_full.columns}")
    # If error, check which columns are in `potential_features` vs `data_with_target.columns`
    print(f"Columns in data_with_target: {data_with_target.columns.tolist()}")
    exit()

if len(X_full) < 20:
    print("Error: Not enough data points in X_full after initial preparation to proceed.")
    exit()

# X_full and y_full are now correctly aligned:
# X_full.loc[date_T] contains features known at end of day T.
# y_full.loc[date_T] contains the Target for day T+1.

# Split data into training and test Sets (Chronological)
print("--- Splitting Data into Training and Test Sets (Chronological) ---")
# Ensure X_full is sorted by index (time) before splitting
X_full = X_full.sort_index()
y_full = y_full.loc[X_full.index] # Realign y_full after sorting X_full

split_index = int(len(X_full) * (1 - TEST_SIZE))

if split_index <= 0 or split_index >= len(X_full):
    print(f"Error: Invalid split_index {split_index} for data of length {len(X_full)}. Check TEST_SIZE.")
    exit()

X_train_raw = X_full.iloc[:split_index].copy()
X_test_raw = X_full.iloc[split_index:].copy()
y_train = y_full.iloc[:split_index].copy()
y_test = y_full.iloc[split_index:].copy()

if X_train_raw.empty or X_test_raw.empty:
    print("Error: Training or test set is empty after chronological split. Check TEST_SIZE and data length.")
    print(f"X_train_raw shape: {X_train_raw.shape}, X_test_raw shape: {X_test_raw.shape}")
    exit()

print(f"X_train_raw shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
print(f"X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")
print(f"Training data range: {X_train_raw.index.min()} to {X_train_raw.index.max()}")
print(f"Test data range: {X_test_raw.index.min()} to {X_test_raw.index.max()}")
print("-"*50 + "\n")

# ---- K-Means Clustering for Feature Engineering ----
print("--- Applying K-Means Clustering for Feature Engineering ---")
# Select features for clustering (excluding target and identifiers)

# We use all features in X_train_raw for clustering.
# If X_train_raw is empty or has too few samples, skip clustering
if not X_train_raw.empty and X_train_raw.shape[0] > 5: # Ensure enough samples for clustering
    # Determine the optimal number of clusters (K) using the Elbow Method and Silhouette Score
    # We need to scale data *before* applying KElbowVisualizer and KMeans for optimal results
    temp_scaler_for_kmeans = StandardScaler()
    X_train_raw_scaled_for_kmeans = temp_scaler_for_kmeans.fit_transform(X_train_raw)

    print("Determining optimal K for K-Means using Elbow Method...")
    model_elbow = KMeans(random_state=RANDOM_STATE, n_init='auto')
    visualizer_elbow = KElbowVisualizer(model_elbow, k=(2,11), metric='distortion', timings=False)
    visualizer_elbow.fit(X_train_raw_scaled_for_kmeans) # Fit on scaled training data
    optimal_k_elbow = visualizer_elbow.elbow_value_
    visualizer_elbow.show() # Display the elbow plot
    print(f"Optimal K based on Elbow Method: {optimal_k_elbow}")

    if optimal_k_elbow is None or optimal_k_elbow < 2:
        print("Elbow method did not suggest a clear K or K < 2. Defaulting K to 3 for demonstration.")
        optimal_k_elbow = 3 # Fallback K

    print(f"\nApplying K-Means with K={optimal_k_elbow}...")
    kmeans = KMeans(n_clusters=optimal_k_elbow, random_state=RANDOM_STATE, n_init='auto')
    
    # We'll fit on scaled X_train_raw, then predict on X_train_raw and X_test_raw (which will be scaled later).
    # This is a common practical approach.

    kmeans.fit(X_train_raw_scaled_for_kmeans) # Fit on scaled training data
    
    # Predict cluster labels for training data (using the original X_train_raw, which will be scaled by the main scaler later)
    # To do this correctly, we should transform X_train_raw with temp_scaler_for_kmeans before predicting
    X_train_cluster_labels = kmeans.predict(temp_scaler_for_kmeans.transform(X_train_raw))
    X_train_raw['cluster_label'] = X_train_cluster_labels
    print(f"Added 'cluster_label' to X_train_raw. Distribution:\n{pd.Series(X_train_cluster_labels).value_counts(normalize=True)}")

    # Predict cluster labels for test data (using the original X_test_raw)
    # Transform X_test_raw with temp_scaler_for_kmeans before predicting
    if not X_test_raw.empty:
        X_test_raw_scaled_for_kmeans = temp_scaler_for_kmeans.transform(X_test_raw)
        X_test_cluster_labels = kmeans.predict(X_test_raw_scaled_for_kmeans)
        X_test_raw['cluster_label'] = X_test_cluster_labels
        print(f"Added 'cluster_label' to X_test_raw. Distribution:\n{pd.Series(X_test_cluster_labels).value_counts(normalize=True)}")
    else:
        print("X_test_raw is empty, skipping cluster label addition for test set.")

    # Visualize Silhouette Scores for the chosen K
    print(f"\nVisualizing Silhouette Scores for K={optimal_k_elbow}...")
    visualizer_silhouette = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer_silhouette.fit(X_train_raw_scaled_for_kmeans) # Fit on scaled training data
    visualizer_silhouette.show() # Display the silhouette plot

else:
    print("Skipping K-Means clustering: X_train_raw is empty or has too few samples.")
print("-"*50 + "\n")

# ---- 4. Feature scaling ----
# (Applied after splitting and potential K-Means feature addition)
print("--- Scaling Features (Fit on Train, Transform Train & Test) ---")
scaler = StandardScaler()
X_train_scaled_array = scaler.fit_transform(X_train_raw)
X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train_raw.columns, index=X_train_raw.index)

X_test_scaled_array = scaler.transform(X_test_raw) # Use transform, not fit_transform, on test data
X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_test_raw.columns, index=X_test_raw.index)

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print("-"*50 + "\n")


# Print all feature names after scaling
print("--- List of Features After Scaling ---")
print(f"Total number of features: {len(X_train_scaled.columns)}")
print(list(X_train_scaled.columns))
print("-"*50 + "\n")

# Print scaled values (head)
print("--- Scaled Values (Train Data - Head) ---")
print(X_train_scaled.head())
print("-"*50 + "\n")

print("--- Scaled Values (Test Data - Head) ---")
print(X_test_scaled.head())
print("-"*50 + "\n")


# ==== Step 5 (Continued): Feature Selection Using the Funnelling Approach ====
print("--- Feature Selection Using the Funnelling Approach (Fit on Training Data) ---")
print("Initial number of features for selection (from training data):", X_train_scaled.shape[1])

X_train_current_fs = X_train_scaled.copy()
# X_test_current_fs will be transformed using selectors fitted on X_train_current_fs

# Step 1a: VIF Filtering (Fit on train, transform train and test)
print("\nStep 1a: VIF Filtering")
vif_threshold = 10.0
features_to_keep_vif = X_train_current_fs.columns.tolist()

# Iteratively remove features with high VIF
while True:
    if len(features_to_keep_vif) <= 1:
        print("VIF: Stopping as 1 or fewer features remain.")
        break

    X_vif_check = X_train_current_fs[features_to_keep_vif]
    # Ensure X_vif_check has no NaN/inf values before VIF calculation
    if np.any(np.isnan(X_vif_check.values)) or np.any(np.isinf(X_vif_check.values)):
        print("VIF: NaN or Inf found in data. Attempting to fill with mean.")
        # Impute NaNs with mean, Infs with large finite numbers (or mean of non-inf)
        for col in X_vif_check.columns:
            if np.any(np.isinf(X_vif_check[col])):
                finite_vals = X_vif_check[col][np.isfinite(X_vif_check[col])]
                replace_val = finite_vals.mean() if not finite_vals.empty else 0
                X_vif_check[col] = np.nan_to_num(X_vif_check[col], nan=replace_val, posinf=replace_val, neginf=replace_val)
            if np.any(np.isnan(X_vif_check[col])):
                 X_vif_check[col] = X_vif_check[col].fillna(X_vif_check[col].mean())
        # If still NaNs after imputation (e.g., all values were NaN), drop the column
        cols_to_drop_due_to_nan_inf = X_vif_check.columns[X_vif_check.isna().all()].tolist()
        if cols_to_drop_due_to_nan_inf:
            print(f"VIF: Dropping columns with all NaN/Inf after imputation: {cols_to_drop_due_to_nan_inf}")
            features_to_keep_vif = [f for f in features_to_keep_vif if f not in cols_to_drop_due_to_nan_inf]
            if not features_to_keep_vif:
                print("VIF: No features left after dropping all NaN/Inf columns.")
                break
            X_vif_check = X_train_current_fs[features_to_keep_vif]
            if X_vif_check.empty:
                print("VIF: X_vif_check became empty after dropping NaN/Inf columns.")
                break

    # Add constant for VIF calculation if not present
    # Check if a constant column already exists or if all columns are constant
    if not (X_vif_check.nunique() == 1).all(): # if not all columns are constant
        # Check if a column is already effectively a constant (e.g., due to scaling)
        is_const_col = (X_vif_check.std() < 1e-9).any()
        if not is_const_col:
            X_vif_check_with_const = sm.add_constant(X_vif_check, has_constant='add')
        else:
            X_vif_check_with_const = X_vif_check # Use as is if a constant-like column exists
    else:
        print("VIF: All features are constant. Skipping VIF calculation.")
        break

    if X_vif_check_with_const.shape[1] <= 1: # Need at least two columns for VIF (one const, one feature)
        print("VIF: Not enough features to calculate VIF (after adding constant).")
        break

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif_check_with_const.columns
    try:
        vif_data["VIF"] = [variance_inflation_factor(X_vif_check_with_const.values, i) for i in range(X_vif_check_with_const.shape[1])]
    except Exception as e:
        print(f"VIF: Error during VIF calculation: {e}. Skipping VIF for this iteration.")
        # This might happen due to perfect multicollinearity not caught by previous checks
        # or if X_vif_check_with_const becomes singular.
        # One strategy could be to remove a feature at random or based on another heuristic.
        # For now, we break to avoid an infinite loop if the error persists.
        if len(features_to_keep_vif) > 0:
            feature_to_remove_on_error = features_to_keep_vif[-1] # Example: remove last feature
            print(f"VIF: Attempting to remove feature '{feature_to_remove_on_error}' due to calculation error.")
            features_to_keep_vif.remove(feature_to_remove_on_error)
            if not features_to_keep_vif:
                print("VIF: No features left after error handling.")
                break
            continue # Retry VIF calculation
        else:
            print("VIF: No features to remove after error.")
            break

    # Exclude 'const' from VIF removal consideration if it was added
    vif_data_filtered = vif_data[vif_data['feature'] != 'const']
    if vif_data_filtered.empty:
        print("VIF: No non-constant features to assess VIF for.")
        break

    max_vif = vif_data_filtered['VIF'].max()
    if max_vif > vif_threshold:
        feature_to_remove = vif_data_filtered.sort_values('VIF', ascending=False)['feature'].iloc[0]
        features_to_keep_vif.remove(feature_to_remove)
        print(f"VIF: Removed feature '{feature_to_remove}' with VIF {max_vif:.2f}")
    else:
        print(f"VIF: All remaining features have VIF <= {vif_threshold}. Max VIF: {max_vif:.2f}")
        break

if not features_to_keep_vif:
    print("VIF: No features selected after VIF. Using all original features for subsequent steps.")
    # Fallback: if all features were removed, revert to original set before VIF
    # This might happen if data has severe multicollinearity or issues
    X_train_current_fs = X_train_scaled.copy()
    selected_vif_features = X_train_scaled.columns.tolist()
else:
    X_train_current_fs = X_train_scaled[features_to_keep_vif].copy()
    selected_vif_features = features_to_keep_vif

initial_features_before_vif = X_train_scaled.columns.tolist()
num_removed_vif = len(initial_features_before_vif) - len(selected_vif_features)
print(f"VIF: Removed {num_removed_vif} features.")
print(f"Selected {len(selected_vif_features)} features after VIF: {selected_vif_features}")

# Step 1b: Filter Methods (SelectKBest with f_classif) - Applied on VIF-filtered data
print("\nStep 1b: Filter Method (SelectKBest with f_classif) on VIF-filtered data")
selected_filter_features = X_train_current_fs.columns.tolist() # Default to all if step is skipped
if not X_train_current_fs.empty and not y_train.empty and X_train_current_fs.shape[1] > 0:
    k_filter = max(10, int(X_train_current_fs.shape[1] * 0.75))
    if k_filter > X_train_current_fs.shape[1]: k_filter = X_train_current_fs.shape[1]
    if k_filter <= 0 : k_filter = min(1, X_train_current_fs.shape[1]) # Ensure k is at least 1 if features exist
    
    if k_filter > 0: # Proceed only if k_filter is positive
        selector_filter = SelectKBest(score_func=f_classif, k=k_filter)
        X_train_filtered_array = selector_filter.fit_transform(X_train_current_fs, y_train)
        features_before_selectkbest = X_train_current_fs.columns.tolist()
        selected_filter_features_mask = selector_filter.get_support()
        selected_filter_features = X_train_current_fs.columns[selected_filter_features_mask].tolist()
        removed_by_selectkbest = [f for i, f in enumerate(features_before_selectkbest) if not selected_filter_features_mask[i]]
        num_removed_selectkbest = len(removed_by_selectkbest)
        print(f"SelectKBest: Removed {num_removed_selectkbest} features: {removed_by_selectkbest}")
        print(f"Selected {len(selected_filter_features)} features after filter method: {selected_filter_features}")
        X_train_current_fs = pd.DataFrame(X_train_filtered_array, columns=selected_filter_features, index=X_train_current_fs.index)
    else:
        print("Skipping Filter Method: k_filter is not positive (no features to select or all features are constant).")
else:
    print("Skipping Filter Method: Training data/target is empty or no features to select.")

# Apply filter selection to test set
X_test_current_fs = X_test_scaled[selected_filter_features].copy() # Select same features on test set

# Step 2: Wrapper Methods (Fit on train, transform train and test)
print("\nStep 2: Wrapper Method (RFE with Logistic Regression)")

# Define the input for this wrapper stage based on the output of the Filter stage.
# selected_filter_features contains the column names from the Filter stage.
# This ensures that if this cell is run multiple times, it always starts from the same state post-Filter.
_X_train_input_for_wrapper = X_train_scaled[selected_filter_features].copy()

# Initialize the features that will be selected by this wrapper stage.
# If RFE is skipped, these will be the features passed to the next stage.
_selected_features_after_this_wrapper_run = _X_train_input_for_wrapper.columns.tolist()
_X_train_processed_by_wrapper = _X_train_input_for_wrapper.copy() # Default if RFE is skipped or fails

if not _X_train_input_for_wrapper.empty and not y_train.empty and _X_train_input_for_wrapper.shape[1] > 0:
    n_wrapper = max(5, int(_X_train_input_for_wrapper.shape[1] * 0.67))
    if n_wrapper > _X_train_input_for_wrapper.shape[1]: n_wrapper = _X_train_input_for_wrapper.shape[1]
    if n_wrapper <= 0 : n_wrapper = min(1, _X_train_input_for_wrapper.shape[1])

    if n_wrapper > 0:
        estimator_rfe = LogisticRegression(solver='liblinear', max_iter=200, random_state=RANDOM_STATE, C=0.1) # Added C for regularization
        selector_wrapper = RFE(estimator_rfe, n_features_to_select=n_wrapper, step=1)
        
        # Perform RFE on _X_train_input_for_wrapper
        _X_train_wrapper_array_temp = selector_wrapper.fit_transform(_X_train_input_for_wrapper, y_train) # fit_transform returns an array
        
        _features_before_rfe_in_wrapper = _X_train_input_for_wrapper.columns.tolist()
        # fit_transform already called fit, so get_support is available
        _selected_features_mask_rfe = selector_wrapper.get_support()
        _selected_features_after_this_wrapper_run = _X_train_input_for_wrapper.columns[_selected_features_mask_rfe].tolist()
        
        _removed_by_rfe_in_wrapper = [f for i, f in enumerate(_features_before_rfe_in_wrapper) if not _selected_features_mask_rfe[i]]
        _num_removed_rfe_in_wrapper = len(_removed_by_rfe_in_wrapper)
        print(f"Wrapper Method: Removed {_num_removed_rfe_in_wrapper} features: {_removed_by_rfe_in_wrapper}")
        print(f"Selected {len(_selected_features_after_this_wrapper_run)} features after wrapper method: {_selected_features_after_this_wrapper_run}")
        
        # Update _X_train_processed_by_wrapper with the result of RFE
        _X_train_processed_by_wrapper = pd.DataFrame(_X_train_wrapper_array_temp, columns=_selected_features_after_this_wrapper_run, index=_X_train_input_for_wrapper.index)
    else:
        print("Skipping Wrapper Method: n_wrapper is not positive.")
        # _X_train_processed_by_wrapper and _selected_features_after_this_wrapper_run retain their initial values (all features from input)
else:
    print("Skipping Wrapper Method: Training data from previous step/target is empty or no features to select.")
    # _X_train_processed_by_wrapper and _selected_features_after_this_wrapper_run retain their initial values

# Update the main X_train_current_fs and X_test_current_fs for subsequent stages
X_train_current_fs = _X_train_processed_by_wrapper.copy()
# X_test_current_fs should be based on X_test_scaled and the final _selected_features_after_this_wrapper_run
X_test_current_fs = X_test_scaled[_selected_features_after_this_wrapper_run].copy()
selected_wrapper_features = _selected_features_after_this_wrapper_run[:] # Make a copy

# Step 3: Embedded Methods (Fit on train, select features for train and test)
print("\nStep 3: Embedded Method (Feature Importances from GradientBoostingClassifier)")
selected_embedded_features = X_train_current_fs.columns.tolist() # Default to current features
if not X_train_current_fs.empty and not y_train.empty and X_train_current_fs.shape[1] > 0:
    temp_gb_model = GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=50, max_depth=3) # Simpler model for feature importance
    temp_gb_model.fit(X_train_current_fs, y_train)
    importances = pd.Series(temp_gb_model.feature_importances_, index=X_train_current_fs.columns)
    
    if not importances.empty:
        n_embedded = max(3, int(len(importances) * 0.8))
        if n_embedded > len(importances): n_embedded = len(importances)
        if n_embedded <= 0 and len(importances) > 0: n_embedded = min(1, len(importances))
        
        if n_embedded > 0:
            selected_embedded_features = importances.nlargest(n_embedded).index.tolist()
            if not selected_embedded_features and len(importances) > 0: # Fallback if nlargest is empty but importances exist
                 selected_embedded_features = importances.index.tolist()[:n_embedded]
            print(f"Selected {len(selected_embedded_features)} features after embedded method: {selected_embedded_features}")
            removed_features = [f for f in X_train_current_fs.columns if f not in selected_embedded_features]
            if removed_features:
                print(f"Removed {len(removed_features)} features by embedded method: {removed_features}")
            else:
                print("No features were removed by the embedded method.")
        else:
            print("No features selected by embedded method as n_embedded is not positive, using features from wrapper method.")
    else:
        print("No importances found from embedded method, using features from wrapper method.")
else:
    print("Skipping Embedded Method: Training data from previous step/target is empty or no features to select.")

# Final selected features for training and testing
X_train_selected = X_train_current_fs[selected_embedded_features].copy()
X_test_selected = X_test_current_fs[selected_embedded_features].copy()

print("\n--- Final List of Selected Features (Applied to Train and Test Sets) ---")
print(f"Features for Training: {list(X_train_selected.columns)}")
print(f"Total features selected for training: {X_train_selected.shape[1]}")
print(f"Features for Testing: {list(X_test_selected.columns)}")
print(f"Total features selected for testing: {X_test_selected.shape[1]}")
print("-"*50 + "\n")

if X_train_selected.empty:
    print("Error: No features selected for training. Halting execution.")
    exit()

# Ensure X_test_selected has columns, even if empty, it should match X_train_selected columns for consistency if predictions are made
if X_test_selected.empty and not X_train_selected.empty:
    print("Warning: X_test_selected is empty but X_train_selected is not. This might happen if test set is very small or features are dropped.")

# Rename variables for clarity in the model training part
X_train = X_train_selected
X_test = X_test_selected

# The y_train and y_test are already defined from the chronological split

print(f"Final X_train shape for model: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Final X_test shape for model: {X_test.shape}, y_test shape: {y_test.shape}")
print("-"*50 + "\n")


# ==== Step 6: Model Training ====
print("--- Model Building and Tuning for Multiple Boosters ---")

# Common TimeSeriesSplit for all models
tscv = TimeSeriesSplit(n_splits=3) # Or your preferred number of splits

# --- 1. Define Models and their Hyperparameter Grids ---
models_to_try = {
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'), # eval_metric for suppressing warning
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1), # verbosity to suppress output
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [20, 31, 40], # max_depth alternative for LGBM
            'subsample': [0.8],
            'colsample_bytree': [0.8, 1.0] # Also known as feature_fraction
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=RANDOM_STATE, verbose=0), # verbose=0 to suppress output
        'params': {
            'iterations': [100, 200], # n_estimators equivalent
            'learning_rate': [0.05, 0.1],
            'depth': [3, 4],          # max_depth equivalent
            'l2_leaf_reg': [1, 3, 5]  # L2 regularization
        }
    }
}

best_estimators = {}
grid_search_results = {}

# --- 2. Loop through models, perform GridSearchCV, and store results ---
for model_name, config in models_to_try.items():
    print(f"\n--- Tuning {model_name} ---")
    grid_search = GridSearchCV(estimator=config['model'],
                               param_grid=config['params'],
                               cv=tscv,
                               scoring='roc_auc',
                               n_jobs=-1,
                               verbose=1)
    try:
        grid_search.fit(X_train, y_train) # X_train and y_train are from your feature selection
        best_estimators[model_name] = grid_search.best_estimator_
        grid_search_results[model_name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        print(f"Best AUC for {model_name} (on validation folds): {grid_search.best_score_:.4f}")
        print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
    except Exception as e:
        print(f"Error tuning {model_name}: {e}")
        best_estimators[model_name] = None # Mark as None if tuning failed
        grid_search_results[model_name] = {
            'best_score': -1, # Indicate failure
            'best_params': {}
        }

print("\n--- Hyperparameter Tuning Complete for all models ---")
for model_name, results in grid_search_results.items():
    if results['best_score'] != -1:
        print(f"{model_name}: Best CV AUC = {results['best_score']:.4f}, Params = {results['best_params']}")
    else:
        print(f"{model_name}: Tuning failed.")


# ==== Step 7: Model Evaluation & Iteration ====
print("\n--- Evaluating Optimal Models on Test Set ---")

model_performance_reports = {}
all_roc_auc_scores = {}
all_fpr = {}
all_tpr = {}

for model_name, best_model_instance in best_estimators.items():
    if best_model_instance is None: # Skip if model tuning failed
        print(f"\n--- Skipping evaluation for {model_name} (tuning failed) ---")
        all_roc_auc_scores[model_name] = -1 # Indicate failure
        continue

    print(f"\n--- Evaluating {model_name} ---")
    y_pred_proba = best_model_instance.predict_proba(X_test)[:, 1]
    y_pred = best_model_instance.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    all_roc_auc_scores[model_name] = roc_auc
    print(f"Test Set AUC for {model_name}: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    report_dict = classification_report(y_test, y_pred, target_names=['Negative Move (0)', 'Positive Move (1)'], output_dict=True)
    model_performance_reports[model_name] = report_dict
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative Move (0)', 'Positive Move (1)']))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    all_fpr[model_name] = fpr
    all_tpr[model_name] = tpr

# --- Plotting all ROC Curves ---
plt.figure(figsize=(10, 8))
plotted_anything = False
for model_name in best_estimators.keys():
    if model_name in all_fpr and model_name in all_tpr and all_roc_auc_scores.get(model_name, -1) != -1:
        plt.plot(all_fpr[model_name], all_tpr[model_name], lw=2,
                 label=f'{model_name} (AUC = {all_roc_auc_scores[model_name]:.2f})')
        plotted_anything = True

if plotted_anything:
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves - Test Set')
    plt.legend(loc='lower right')
    plt.show()
else:
    print("No models available to plot ROC curves.")

# --- Select the Best Model based on a chosen metric (e.g., Test AUC) ---
final_model = None
best_overall_model_name = ""

# Filter out models that failed or had no score
valid_scores = {name: score for name, score in all_roc_auc_scores.items() if score != -1 and best_estimators.get(name) is not None}

if valid_scores:
    best_overall_model_name = max(valid_scores, key=valid_scores.get)
    final_model = best_estimators[best_overall_model_name]

    print(f"\n--- Best Overall Model Selected: {best_overall_model_name} ---")
    print(f"Test AUC for {best_overall_model_name}: {valid_scores[best_overall_model_name]:.4f}")
    print(f"Parameters for {best_overall_model_name}: {final_model.get_params()}")

    # Display confusion matrix and classification report for the final best model
    print(f"\n--- Detailed Evaluation for the Best Overall Model: {best_overall_model_name} ---")
    y_pred_final_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred_final = final_model.predict(X_test)

    cm_final = confusion_matrix(y_test, y_pred_final)
    print("\nConfusion Matrix (Best Overall Model):")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix - {best_overall_model_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    print("\nClassification Report (Best Overall Model):")
    print(classification_report(y_test, y_pred_final, target_names=['Negative Move (0)', 'Positive Move (1)']))
else:
    print("\n--- No models were successfully trained and evaluated. Cannot select a best model. ---_model = None") # Ensure final_model is None if no models are trained/evaluated successfully

print("-"*50 + "\n")
# final_model is now set (or None) and will be used by the backtesting step below.


# ==== Step 7 (Optional Add-on): Back-testing for Multiple Models ====
def backtest_strategy(df_with_predictions, model_name, initial_capital=100000):
    """
    A simple backtest for the trading signals for a specific model.
    Assumes df_with_predictions has 'close' prices and 'Predicted_Signal' (1 for buy/hold, 0 for sell/neutral).
    Returns the portfolio values over time for this model.
    """
    print(f"\n--- Backtesting Trading Strategy for {model_name} ---")
    capital = initial_capital
    shares_held = 0
    portfolio_values = []
    trade_log = [] # To log trades

    # --- Buy and Hold Strategy Initialization ---
    buy_and_hold_capital = initial_capital
    buy_and_hold_shares_held = 0
    buy_and_hold_portfolio_values = []
    # Ensure df_with_predictions is not empty and has 'close' prices
    if not df_with_predictions.empty and 'close' in df_with_predictions.columns:
        first_price_bh = df_with_predictions['close'].iloc[0]
        if pd.notna(first_price_bh) and first_price_bh > 0: # Ensure price is valid and positive
            buy_and_hold_shares_held = buy_and_hold_capital // first_price_bh
            buy_and_hold_capital -= buy_and_hold_shares_held * first_price_bh
    # --- End Buy and Hold Strategy Initialization ---

    # Ensure 'close' and 'Predicted_Signal' are in the dataframe
    if 'close' not in df_with_predictions.columns or 'Predicted_Signal' not in df_with_predictions.columns:
        print("Error: 'close' or 'Predicted_Signal' column missing in data for backtesting.")
        return

    for date, row in df_with_predictions.iterrows():
        current_price = row['close']
        signal = row['Predicted_Signal']

        # Buy signal
        if signal == 1 and shares_held == 0:
            shares_to_buy = capital // current_price # Buy as many shares as possible
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                capital -= cost
                shares_held += shares_to_buy
                trade_log.append(f"{date}: BUY {shares_to_buy} shares @ {current_price:.2f}, Cost: {cost:.2f}")
        # Sell signal
        elif signal == 0 and shares_held > 0:
            proceeds = shares_held * current_price
            capital += proceeds
            trade_log.append(f"{date}: SELL {shares_held} shares @ {current_price:.2f}, Proceeds: {proceeds:.2f}")
            shares_held = 0
        
        current_portfolio_value = capital + (shares_held * current_price)
        portfolio_values.append(current_portfolio_value)

        # --- Buy and Hold Strategy Daily Value ---
        if buy_and_hold_shares_held > 0: # If shares were bought for B&H
            current_buy_and_hold_portfolio_value = buy_and_hold_capital + (buy_and_hold_shares_held * current_price)
            buy_and_hold_portfolio_values.append(current_buy_and_hold_portfolio_value)
        else: # If no shares were bought (e.g., initial price was zero/NaN or not enough capital) or df was empty
            buy_and_hold_portfolio_values.append(initial_capital) # Portfolio value remains initial capital
        # --- End Buy and Hold Strategy Daily Value ---

    # Final portfolio value
    final_portfolio_value = portfolio_values[-1] if portfolio_values else initial_capital
    print(f"Initial Capital: {initial_capital:.2f}")
    
    print("\n--- Active Trading Strategy Results ---")
    if portfolio_values:
        final_active_portfolio_value = portfolio_values[-1] if portfolio_values else initial_capital
        print(f"Final Portfolio Value (Active Strategy): {final_active_portfolio_value:.2f}")
        total_return_active_pct = ((final_active_portfolio_value - initial_capital) / initial_capital) * 100
        print(f"Total Return (Active Strategy): {total_return_active_pct:.2f}%")
    else:
        print("No portfolio values generated for Active Strategy.")

    # --- Buy and Hold Strategy Results ---
    if buy_and_hold_portfolio_values:
        final_buy_and_hold_portfolio_value = buy_and_hold_portfolio_values[-1]
        print("\n--- Buy and Hold Strategy Results ---")
        print(f"Final Portfolio Value (Buy and Hold): {final_buy_and_hold_portfolio_value:.2f}")
        total_return_bh_pct = ((final_buy_and_hold_portfolio_value - initial_capital) / initial_capital) * 100
        print(f"Total Return (Buy and Hold): {total_return_bh_pct:.2f}%")
    else:
        print("\nNo portfolio values generated for Buy and Hold Strategy.")
    # --- End Buy and Hold Strategy Results ---

    # Print trade log
    print("\nTrade Log:")
    if not trade_log:
        print("No trades executed.")
    else:
        for log_entry in trade_log:
            print(log_entry)

    # Return portfolio values for plotting later
    # Also return buy_and_hold_portfolio_values to plot it once
    return portfolio_values, buy_and_hold_portfolio_values, df_with_predictions.index[:len(portfolio_values)]

print("\n--- Performing Backtesting for All Successful Models ---")
all_models_portfolio_values = {}
buy_and_hold_pv = None # To store buy and hold portfolio values once
backtest_dates = None # To store dates for plotting
initial_capital_backtest = 100000

if not X_test.empty and 'close' in data_with_target.columns:
    base_test_data_for_backtest = data_with_target.loc[X_test.index].copy()

    for model_name, model_instance in best_estimators.items():
        if model_instance is not None: # Check if the model was successfully trained
            print(f"\nPreparing backtest for {model_name}...")
            current_test_data = base_test_data_for_backtest.copy()
            current_test_data['Predicted_Signal'] = model_instance.predict(X_test)
            
            portfolio_values, bh_pv, dates = backtest_strategy(current_test_data, model_name, initial_capital=initial_capital_backtest)
            all_models_portfolio_values[model_name] = portfolio_values
            if buy_and_hold_pv is None and bh_pv: # Store B&H and dates only once
                buy_and_hold_pv = bh_pv
                backtest_dates = dates # Assuming all models run on the same date range
        else:
            print(f"Skipping backtest for {model_name} as it was not successfully trained.")

    # --- Plotting all backtest results --- 
    if all_models_portfolio_values:
        plt.figure(figsize=(14, 8))
        
        # Plot Buy & Hold strategy first if available
        if buy_and_hold_pv and backtest_dates is not None:
             # Ensure buy_and_hold_pv is sliced to match the length of backtest_dates if necessary
            plt.plot(backtest_dates, buy_and_hold_pv[:len(backtest_dates)], label='Buy & Hold Strategy', linestyle='--', color='grey')

        # Plot each model's active strategy
        # Define a list of colors and linestyles to cycle through
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'olive', 'cyan']
        linestyles = ['-', '--', '-.', ':']
        color_idx = 0
        linestyle_idx = 0

        for model_name, p_values in all_models_portfolio_values.items():
            if p_values and backtest_dates is not None:
                # Ensure p_values is sliced to match the length of backtest_dates
                current_color = colors[color_idx % len(colors)]
                current_linestyle = linestyles[linestyle_idx % len(linestyles)]
                plt.plot(backtest_dates, p_values[:len(backtest_dates)], label=f'{model_name} Strategy', color=current_color, linestyle=current_linestyle)
                color_idx += 1
                # Change linestyle less frequently than color, or based on some other logic if preferred
                if color_idx % len(colors) == 0:
                    linestyle_idx += 1

        plt.title('Portfolio Value Over Time: All Models vs. Buy & Hold')
        plt.xlabel('Date')
        plt.ylabel(f'Portfolio Value (Initial Capital: {initial_capital_backtest})')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No successful models to backtest or plot.")

else:
    print("Skipping all backtesting: X_test is empty, or 'close' column missing in data_with_target.")

print("-"*50 + "\n")
