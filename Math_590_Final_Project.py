#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('gold_silver_ratio_yahoo_2000_2025.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# Feature engineering on Gold/Silver Ratio
df['ratio_1d'] = df['Gold_Silver_Ratio'].pct_change()
df['ratio_2d'] = df['Gold_Silver_Ratio'].pct_change(2)
df['ma_3'] = df['Gold_Silver_Ratio'].rolling(3).mean()
df['ma_5'] = df['Gold_Silver_Ratio'].rolling(5).mean()
df['ma_diff'] = df['ma_3'] - df['ma_5']
df['vol_5d'] = df['ratio_1d'].rolling(5).std()

# Label: will ratio go up tomorrow?
df['target'] = np.where(df['Gold_Silver_Ratio'].shift(-1) > df['Gold_Silver_Ratio'], 1, 0)

# Drop NA rows created by rolling
df.dropna(inplace=True)

# Filter to 2013–2019
df = df[(df['Date'] >= '2013-01-01') & (df['Date'] <= '2019-12-31')]

# Split train/test: train 2013–2018, test 2019
train_df = df[df['Date'] < '2019-01-01'].copy()
test_df = df[df['Date'] >= '2019-01-01'].copy()

features = ['ratio_1d', 'ratio_2d', 'ma_diff', 'vol_5d']
X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
test_df['pred'] = model.predict(X_test)

# Evaluate predictive performance
acc = accuracy_score(y_test, test_df['pred'])
report = classification_report(y_test, test_df['pred'])

# Simulate trading strategy
capital = 10000.0
# We will iterate through test_df by index position
gold = test_df['Gold_Price'].values
silver = test_df['Silver_Price'].values
preds = test_df['pred'].values

for i in range(len(test_df) - 1):
    if preds[i] == 1:
        # hold gold
        ret = gold[i+1] / gold[i]
    else:
        # hold silver
        ret = silver[i+1] / silver[i]
    capital *= ret

final_capital = capital
total_return = (final_capital - 10000) / 10000 * 100  # percentage

# Display results
print(f"Test accuracy (2019): {acc:.2%}")
print("Classification Report:\n", report)
print(f"Final portfolio value at end of 2019: ${final_capital:,.2f}")
print(f"Total return: {total_return:.2f}%")


# In[9]:


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
test_df['pred'] = rf.predict(X_test)

# Evaluate predictive performance
acc = accuracy_score(y_test, test_df['pred'])
report = classification_report(y_test, test_df['pred'])

# Simulate trading strategy
capital = 10000.0
gold = test_df['Gold_Price'].values
silver = test_df['Silver_Price'].values
preds = test_df['pred'].values

for i in range(len(test_df) - 1):
    if preds[i] == 1:
        ret = gold[i+1] / gold[i]
    else:
        ret = silver[i+1] / silver[i]
    capital *= ret

final_capital = capital
total_return = (final_capital - 10000) / 10000 * 100  # percentage

# Display results
print(f"Random Forest Test accuracy (2019): {acc:.2%}")
print("Classification Report:\n", report)
print(f"Final portfolio value at end of 2019: ${final_capital:,.2f}")
print(f"Total return: {total_return:.2f}%")


# In[10]:


from xgboost import XGBClassifier

# === Train XGBoost with limited resources ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                      random_state=42, n_jobs=1, tree_method='hist', verbosity=0)
model.fit(X_train, y_train)

# === Predict & Evaluate ===
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)

# === Simulate trading ===
capital = 10000.0
gold = test_df['Gold_Price'].values
silver = test_df['Silver_Price'].values

for i in range(len(preds) - 1):
    ret = gold[i+1]/gold[i] if preds[i] == 1 else silver[i+1]/silver[i]
    capital *= ret

final_capital = capital
total_return = (final_capital - 10000) / 10000 * 100

# === Results ===
print("Model used: XGBoost")
print(f"Test accuracy (2019): {acc:.2%}")
print("Classification Report:\n", report)
print(f"Final portfolio value at end of 2019: ${final_capital:,.2f}")
print(f"Total return: {total_return:.2f}%")


# In[11]:


# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, tree_method='hist', n_jobs=1, verbosity=0)
}
switch_counts = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # Count switches: number of times holding prediction changes
    switches = int(np.sum(preds[1:] != preds[:-1]))
    switch_counts[name] = switches

# Display results
switch_df = pd.DataFrame.from_dict(switch_counts, orient='index', columns=['Switch Count'])
switch_df


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

# Provide the path to your local CSV file
file_path = 'gold_silver_ratio_yahoo_2000_2025.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Find the data for 2019
df['Date'] = pd.to_datetime(df['Date'])
df_2019 = df[df['Date'].dt.year == 2019].copy()

# Calculate the daily return
df_2019['daily_return'] = df_2019['Gold_Silver_Ratio'].pct_change()

# Define your MA windows
short_windows = [5, 10, 15, 20, 25]
long_windows  = [30, 35, 40, 45, 50, 55, 60]

best_result = -999
best_short  = None
best_long   = None

for short_w in short_windows:
    for long_w in long_windows:
        if short_w >= long_w:
            continue

        temp_df = df_2019.copy()
        temp_df['short_ma'] = temp_df['Gold_Silver_Ratio'].rolling(window=short_w).mean()
        temp_df['long_ma']  = temp_df['Gold_Silver_Ratio'].rolling(window=long_w).mean()
        temp_df['signal']            = (temp_df['short_ma'] > temp_df['long_ma']).astype(int)
        temp_df['strategy_position'] = temp_df['signal'].shift(1)
        temp_df['strategy_return']   = temp_df['strategy_position'] * temp_df['daily_return']

        # —— New code to count switches:
        # Compute the abs difference between consecutive positions (0→1 or 1→0 gives 1)
        switches_series = temp_df['strategy_position'].diff().abs().fillna(0)
        num_switches   = int(switches_series.sum())
        # You can print it, or store it alongside the result if you like:
        # print(f"{short_w}/{long_w} → switches: {num_switches}")

        cumulative_return = (1 + temp_df['strategy_return']).cumprod().iloc[-1]

        # If this is the best so far, remember the switch count too
        if cumulative_return > best_result:
            best_result = cumulative_return
            best_short  = short_w
            best_long   = long_w
            best_switches = num_switches

# Final output
print(f"The best choice is: short window {best_short} days, "
      f"long window {best_long} days, final return {best_result:.2f}, "
      f"with {best_switches} total position switches.")


# In[13]:


# Plot
# Calculate rolling mean
df_2019['rolling_mean_25'] = df_2019['Gold_Silver_Ratio'].rolling(window=25).mean()
df_2019['rolling_mean_55'] = df_2019['Gold_Silver_Ratio'].rolling(window=55).mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_2019['Date'], df_2019['Gold_Silver_Ratio'], label='Gold_Silver_Ratio', color='dimgray', linewidth=1.5)
plt.plot(df_2019['Date'], df_2019['rolling_mean_25'], label='25-Day Rolling Mean', color='gold', linestyle='--')
plt.plot(df_2019['Date'], df_2019['rolling_mean_55'], label='55-Day Rolling Mean', color='green', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Gold_Silver_Ratio')
plt.title('Gold Silver Ratio with 25 and 55-Day Rolling Means')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# produce trading signals
df_2019['signal'] = 0
df_2019['signal'] = (df_2019['rolling_mean_25'] > df_2019['rolling_mean_55']).astype(int)

# change of signal(1 is buy, -1 is sell)
df_2019['position'] = df_2019['signal'].diff()

# plot with signal
plt.figure(figsize=(14, 7))

# plot price and MA
plt.plot(df_2019['Date'], df_2019['Gold_Silver_Ratio'], label='Gold_Silver_Ratio', color='dimgray', linewidth=1)
plt.plot(df_2019['Date'], df_2019['rolling_mean_25'], label='25-Day MA', color='gold', linestyle='--')
plt.plot(df_2019['Date'], df_2019['rolling_mean_55'], label='55-Day MA', color='red', linestyle='--')

# plot the buying signal
plt.plot(df_2019[df_2019['position'] == 1]['Date'],
         df_2019[df_2019['position'] == 1]['Gold_Silver_Ratio'],
         '^', markersize=10, color='green', label='Buy Signal')

# plot the selling signal
plt.plot(df_2019[df_2019['position'] == -1]['Date'],
         df_2019[df_2019['position'] == -1]['Gold_Silver_Ratio'],
         'v', markersize=10, color='red', label='Sell Signal')

plt.xlabel('Date')
plt.ylabel('Gold_Silver_Ratio')
plt.title('Gold Silver Ratio with 25/55-Day Moving Average Crossover Strategy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Use last day signal to decide today's hold（shift(1)）
df_2019['strategy_position'] = df_2019['signal'].shift(1)

# Calculate daily return for strategy
df_2019['strategy_return'] = df_2019['strategy_position'] * df_2019['daily_return']

# Calculate Equity Curve
df_2019['cumulative_market_return'] = (1 + df_2019['daily_return']).cumprod()
df_2019['cumulative_strategy_return'] = (1 + df_2019['strategy_return']).cumprod()

# Plot the return curve
plt.figure(figsize=(14, 7))
plt.plot(df_2019['Date'], df_2019['cumulative_market_return'], label='Buy and Hold (Gold/Silver Ratio)', color='gray')
plt.plot(df_2019['Date'], df_2019['cumulative_strategy_return'], label='25/55 MA Strategy', color='blue')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Backtest: 25/55 MA Crossover Strategy vs Buy and Hold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[14]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# === Step 1: Load Data ===
df = pd.read_csv('gold_silver_ratio_yahoo_2000_2025.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
series = df['Gold_Silver_Ratio']

# === Step 2: Define Train and Forecast Days ===
train_series = series[(series.index >= '2013-01-01') & (series.index <= '2018-12-31')]
forecast_days = series[(series.index >= '2019-01-01') & (series.index <= '2019-12-31')].index
test_actual = series.loc[forecast_days]

# === Step 3: Select Best ARIMA(p,d,q) on 2013–2018 ===
best_aic = float("inf")
best_order = None

print("🔍 Selecting best ARIMA(p,d,q) model on training data...")
for p in range(0, 4):
    for d in range(0, 2):
        for q in range(0, 4):
            try:
                model = ARIMA(train_series, order=(p, d, q)).fit()
                aic = model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                print(f"ARIMA({p},{d},{q}) - AIC: {aic:.2f}")
            except Exception as e:
                # Skip invalid configurations
                continue

print(f"\n✅ Best ARIMA model: ARIMA{best_order} with AIC: {best_aic:.2f}")

# === Step 4: Rolling One-Day-Ahead Forecast ===
rolling_forecasts = []
full_series = series.copy()

print("\n🔁 Rolling one-day-ahead forecast for 2019...")
for day in forecast_days:
    train_window = full_series[full_series.index < day]
    try:
        model = ARIMA(train_window, order=best_order).fit()
        forecast_value = model.forecast()[0]
    except:
        forecast_value = train_window.iloc[-1]
    rolling_forecasts.append(forecast_value)

# === Step 5: Assemble Forecast Series ===
forecast_series = pd.Series(rolling_forecasts, index=forecast_days)

# === Optional Step: Count Position Switches ===
import numpy as np

# a. Define positions: +1 if forecasted ratio ↑ vs. prior day, –1 if ↓ (ties carry over last position)
#    We prepend a 0 so that diff alignment works; we'll drop it immediately after.
pos_sign = np.sign(forecast_series.diff().fillna(0))
# Replace 0’s (no-change days) with the prior non-zero position:
positions = pos_sign.replace(to_replace=0, method='ffill').fillna(1).astype(int)

# b. Count switches: times when today's position ≠ yesterday's
switches = (positions != positions.shift(1)).sum()

print(f"\n🔄 Number of position switches in your 2019 ARIMA strategy: {switches}")


# In[15]:


# === Step 6: Evaluation ===
mae = mean_absolute_error(test_actual, forecast_series)
rmse = np.sqrt(mean_squared_error(test_actual, forecast_series))
r2 = r2_score(test_actual, forecast_series)

print("\n📊 Evaluation on 2019 One-Day-Ahead Forecast (No Manual Differencing):")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# === Step 7: Plot ===
plt.figure(figsize=(14, 6))
plt.plot(series['2013':'2018'], label='Train (2013–2018)', alpha=0.6)
plt.plot(test_actual.index, test_actual, label='Actual 2019')
plt.plot(forecast_series.index, forecast_series, label='One-Day-Ahead Forecast', linestyle='--')
plt.title(f'One-Day-Ahead Forecast of Gold-Silver Ratio (ARIMA{best_order}) — No Differencing')
plt.xlabel('Date')
plt.ylabel('Gold/Silver Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[42]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import friedmanchisquare, wilcoxon
from numpy.linalg import LinAlgError

# Load and preprocess data
df = pd.read_csv('gold_silver_ratio_yahoo_2000_2025.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df['ratio_1d'] = df['Gold_Silver_Ratio'].pct_change()
df.dropna(inplace=True)
df.set_index('Date', inplace=True)

# Train/test split
train = df['2013-01-01':'2018-12-31']
test  = df['2019-01-01':'2019-12-31']

# 1) ML models
features = ['ratio_1d']
X_train, y_train = train[features], (train['Gold_Silver_Ratio'].shift(-1) > train['Gold_Silver_Ratio']).astype(int)
X_test = test[features]
models = {
    'Logistic': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, tree_method='hist', n_jobs=1, verbosity=0)
}
for name, m in models.items():
    m.fit(X_train, y_train)
    test[f'pred_{name}'] = m.predict(X_test)

# Compute ML returns
returns = {
    'Logistic': test['pred_Logistic'].shift(1).fillna(0) * test['ratio_1d'],
    'RandomForest': test['pred_RandomForest'].shift(1).fillna(0) * test['ratio_1d'],
    'XGBoost': test['pred_XGBoost'].shift(1).fillna(0) * test['ratio_1d']
}

# 2) MA strategy
test['signal_MA'] = (test['Gold_Silver_Ratio'].rolling(25).mean() >
                     test['Gold_Silver_Ratio'].rolling(55).mean()).astype(int)
returns['MA_25_55'] = test['signal_MA'].shift(1).fillna(0) * test['ratio_1d']

# 3) ARIMA strategy
best_order = (3,1,2)
forecasts = []
for dt in test.index:
    hist = df['Gold_Silver_Ratio'][:dt].iloc[:-1]
    try:
        m = ARIMA(hist, order=best_order).fit()
        fv = m.forecast().iloc[0]
    except (LinAlgError, Exception):
        fv = hist.iloc[-1]
    forecasts.append(fv)
test['signal_ARIMA'] = (pd.Series(forecasts, index=test.index) >
                        test['Gold_Silver_Ratio'].shift(1)).astype(int)
returns['ARIMA(3,1,2)'] = test['signal_ARIMA'].shift(1).fillna(0) * test['ratio_1d']

# Assemble returns DataFrame
returns_df = pd.DataFrame(returns)

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# ==== 1) Friedman test ====
stat, p_val = friedmanchisquare(
    returns_df['Logistic'],
    returns_df['RandomForest'],
    returns_df['XGBoost'],
    returns_df['MA_25_55'],
    returns_df['ARIMA(3,1,2)']
)

friedman_df = pd.DataFrame({
    'Chi² statistic': [stat],
    'p-value': [p_val]
}, index=['Friedman Test'])

# ==== 2) Pairwise Wilcoxon vs ARIMA ====
wilcox_results = []
for other in ['Logistic','RandomForest','XGBoost','MA_25_55']:
    w_stat, w_p = wilcoxon(
        returns_df['ARIMA(3,1,2)'],
        returns_df[other]
    )
    wilcox_results.append({
        'Comparison':    f'ARIMA vs {other}',
        'Wilcoxon W':    w_stat,
        'p-value':       w_p
    })

wilcox_df = pd.DataFrame(wilcox_results).set_index('Comparison')

# ==== Display results ====
print("\n=== Friedman Test ===")
print(friedman_df, "\n\n")

print("=== Wilcoxon Pairwise Comparisons vs ARIMA ===")
print(wilcox_df)


# In[32]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from statsmodels.tsa.arima.model import ARIMA

# === Helper function, updated ===
def compute_metrics(returns, cost_per_switch=0.001, total_switches=None):
    """
    returns: pd.Series of daily P&L returns
    cost_per_switch: transaction cost per switch (decimal, e.g. 0.001 = 10 bps)
    total_switches: if provided, override the calculated switch count
    """
    returns = returns.dropna()
    n = len(returns)

    # 1) Gross Sharpe
    ann_mean = returns.mean() * 252
    ann_vol  = returns.std() * np.sqrt(252)
    sharpe   = ann_mean / ann_vol

    # 2) Max drawdown
    cum = (1 + returns).cumprod()
    dd  = (cum / cum.cummax() - 1).min()

    # 3) Net Sharpe: apply cost_per_switch for each switch
    if total_switches is None:
        # infer switches from changes in the position
        switches = (returns != returns.shift(1)).astype(int).sum()
    else:
        switches = total_switches

    # spread total cost evenly across days
    daily_cost   = (switches * cost_per_switch) / n
    net_returns  = returns - daily_cost
    net_mean     = net_returns.mean() * 252
    net_vol      = net_returns.std() * np.sqrt(252)
    net_sharpe   = net_mean / net_vol

    return sharpe, dd, net_sharpe

# === Load and prepare data ===
df = pd.read_csv('gold_silver_ratio_yahoo_2000_2025.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df['ratio_1d'] = df['Gold_Silver_Ratio'].pct_change()
df['ratio_2d'] = df['Gold_Silver_Ratio'].pct_change(2)
df['ma_3'] = df['Gold_Silver_Ratio'].rolling(3).mean()
df['ma_5'] = df['Gold_Silver_Ratio'].rolling(5).mean()
df['ma_diff'] = df['ma_3'] - df['ma_5']
df['vol_5d'] = df['ratio_1d'].rolling(5).std()
df['target'] = np.where(df['Gold_Silver_Ratio'].shift(-1) > df['Gold_Silver_Ratio'], 1, 0)
df.dropna(inplace=True)

# 2019 test slice
df['Date'] = pd.to_datetime(df['Date'])
mask_2013_18 = (df['Date'] >= '2013-01-01') & (df['Date'] < '2019-01-01')
mask_2019    = (df['Date'] >= '2019-01-01') & (df['Date'] <= '2019-12-31')

train = df.loc[mask_2013_18].copy()
test  = df.loc[mask_2019].copy()
daily_gold_ret   = test['Gold_Price'].pct_change()
daily_silver_ret = test['Silver_Price'].pct_change()

# === Train ML models on 2013–2018 ===
features = ['ratio_1d','ratio_2d','ma_diff','vol_5d']
X_train, y_train = train[features], train['target']
X_test, _         = test[features], test['target']

ml_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost'           : XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                         random_state=42, tree_method='hist', n_jobs=1, verbosity=0)
}
for name, model in ml_models.items():
    model.fit(X_train, y_train)
    test[f'pred_{name}'] = model.predict(X_test)

# === Construct the 25/55 MA strategy ===
test['ma25']   = test['Gold_Silver_Ratio'].rolling(25).mean()
test['ma55']   = test['Gold_Silver_Ratio'].rolling(55).mean()
test['signal_MA']    = (test['ma25'] > test['ma55']).astype(int)
test['position_MA']  = test['signal_MA'].shift(1).fillna(0)
test['return_MA']    = test['position_MA'] * test['ratio_1d']

# === Rolling ARIMA(3,1,2) forecasts & signal ===
# === Define train & test windows ===
train_series = series['2013-01-01':'2018-12-31']
test_index   = series['2019-01-01':'2019-12-31'].index

from numpy.linalg import LinAlgError
from statsmodels.tsa.arima.model import ARIMA

best_order = (3, 1, 2)
full_series = df['Gold_Silver_Ratio']  # df must still be in scope with Date as index

# Rolling one‐day‐ahead forecasts:
forecasts = []
for current_date in test.index:
    hist = full_series[:current_date].iloc[:-1]   # history up to the prior day
    try:
        m = ARIMA(hist, order=best_order).fit()
        fv = m.forecast().iloc[0]
    except (LinAlgError, ValueError, Exception):
        fv = hist.iloc[-1]  # fallback = last observed value
    forecasts.append(fv)

# Now simply attach to the existing test DataFrame:
test['arima_forecast']  = pd.Series(forecasts, index=test.index)
test['signal_ARIMA']    = (test['arima_forecast'] > test['Gold_Silver_Ratio'].shift(1)).astype(int)
test['pos_ARIMA']       = test['signal_ARIMA'].shift(1).fillna(0).astype(int)
test['return_ARIMA']       = test['pos_ARIMA'] * test['ratio_1d']


# Provided switch counts
switch_counts = {
    'LogisticRegression': 59,
    'RandomForest':       129,
    'XGBoost':            125,
    'MA_25_55':            3,
    'ARIMA(3,1,2)':      126
}

# === Simulate P&L & compute metrics ===
results = {}

# ML strategies
for name in ml_models:
    pos = test[f'pred_{name}'].shift(1).fillna(0).astype(int)
    ret = pos * test['ratio_1d']
    # pass in override for switches:
    results[name] = compute_metrics(ret,
                                    cost_per_switch=0.001,
                                    total_switches=switch_counts[name])

# MA strategy
results['MA_25_55'] = compute_metrics(test['return_MA'],
                                      cost_per_switch=0.001,
                                      total_switches=switch_counts['MA_25_55'])

# ARIMA strategy
results['ARIMA(3,1,2)'] = compute_metrics(test['return_ARIMA'],
                                          cost_per_switch=0.001,
                                          total_switches=switch_counts['ARIMA(3,1,2)'])

# === Summarize ===
metrics_df = pd.DataFrame(results,
                          index=['Gross Sharpe','Max DD','Net Sharpe']).T
print(metrics_df)


# In[33]:


import pandas as pd
import matplotlib.pyplot as plt

# Sample metrics data (replace with your actual metrics_df if available)
metrics = {
    'LogisticRegression': {'Gross Sharpe': 1.16, 'Max Drawdown': -0.148, 'Net Sharpe': 0.58},
    'RandomForest'      : {'Gross Sharpe': 1.65, 'Max Drawdown': -0.041, 'Net Sharpe': 0.20},
    'XGBoost'           : {'Gross Sharpe': 2.02, 'Max Drawdown': -0.032, 'Net Sharpe': 0.25},
    'MA_25_55'          : {'Gross Sharpe': 0.90, 'Max Drawdown': -0.082, 'Net Sharpe': 0.86},
    'ARIMA(3,1,2)'      : {'Gross Sharpe': 4.74, 'Max Drawdown': -0.018, 'Net Sharpe': 2.80},
}

# Create DataFrame
metrics_df = pd.DataFrame(metrics).T

# Plot Gross Sharpe
plt.figure()
metrics_df['Gross Sharpe'].plot(kind='bar')
plt.title('Gross Sharpe by Strategy')
plt.xlabel('Strategy')
plt.ylabel('Gross Sharpe')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot Max Drawdown
plt.figure()
metrics_df['Max Drawdown'].plot(kind='bar')
plt.title('Max Drawdown by Strategy')
plt.xlabel('Strategy')
plt.ylabel('Max Drawdown')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot Net Sharpe
plt.figure()
metrics_df['Net Sharpe'].plot(kind='bar')
plt.title('Net Sharpe by Strategy (After Costs)')
plt.xlabel('Strategy')
plt.ylabel('Net Sharpe')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[38]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Assuming `test` DataFrame is already in scope with ML predictions and signals

# Step 1 & 2: Dynamically detect prediction and signal columns
strategies = {}
for col in test.columns:
    if col.startswith("pred_"):
        name = col.replace("pred_", "")
        strategies[name] = {'pred_col': col}
# Add MA and ARIMA signals
strategies['MA_25_55'] = {'pred_col': 'signal_MA'}
strategies['ARIMA']    = {'pred_col': 'signal_ARIMA'}

# Step 3: Compute daily P&L for each strategy
for name, spec in strategies.items():
    pos = test[spec['pred_col']].shift(1).fillna(0).astype(int)
    test[f'ret_{name}'] = pos * test['ratio_1d']

# Step 4: Bootstrap setup
n_boot = 1000
boot_results = {name: {'acc': [], 'shp': []} for name in strategies}
idx = np.arange(len(test))

# Step 5: Run bootstrap
for _ in range(n_boot):
    sample_idx = np.random.choice(idx, size=len(idx), replace=True)
    sample = test.iloc[sample_idx]

    for name, spec in strategies.items():
        # Accuracy
        a = accuracy_score(sample['target'], sample[spec['pred_col']])
        # Sharpe
        rets = sample[f'ret_{name}']
        s = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() != 0 else np.nan

        boot_results[name]['acc'].append(a)
        boot_results[name]['shp'].append(s)

# Step 6: Compute 95% CIs
ci_data = []
for name in strategies:
    accs = np.array(boot_results[name]['acc'])
    shps = np.array(boot_results[name]['shp'])
    ci_data.append({
        'Strategy': name,
        'Acc 2.5%': np.nanpercentile(accs, 2.5),
        'Acc 97.5%': np.nanpercentile(accs, 97.5),
        'Sharpe 2.5%': np.nanpercentile(shps, 2.5),
        'Sharpe 97.5%': np.nanpercentile(shps, 97.5),
    })

ci_df = pd.DataFrame(ci_data).set_index('Strategy')
print("\nBootstrap 95% Confidence Intervals:")
print(ci_df)


# In[39]:


import matplotlib.pyplot as plt

# suppose ci_df looks like:
#                       Acc 2.5%  Acc 97.5%  Sharpe 2.5%  Sharpe 97.5%
# Strategy
# LogisticRegression    0.484127  0.611111     -0.816609      3.238200
# RandomForest          0.491964  0.607143     -0.339538      3.394605
# XGBoost               0.496032  0.615079      0.084126      3.866194
# MA_25_55              0.432540  0.555556     -1.166603      2.764800
# ARIMA(3,1,2)          0.583333  0.698413      2.949326      6.299024

# Compute mid‐points and error bars directly
acc_mid   = (ci_df['Acc 2.5%']    + ci_df['Acc 97.5%']) / 2
acc_err   = [acc_mid - ci_df['Acc 2.5%'], ci_df['Acc 97.5%'] - acc_mid]

sh_mid    = (ci_df['Sharpe 2.5%'] + ci_df['Sharpe 97.5%']) / 2
sh_err    = [sh_mid - ci_df['Sharpe 2.5%'], ci_df['Sharpe 97.5%'] - sh_mid]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,6))

# Accuracy CIs
ax1.errorbar(ci_df.index, acc_mid, yerr=acc_err, fmt='o', capsize=5)
ax1.set_title('95% CI for Predictive Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xticklabels(ci_df.index, rotation=45, ha='right')

# Sharpe CIs
ax2.errorbar(ci_df.index, sh_mid, yerr=sh_err, fmt='o', capsize=5)
ax2.set_title('95% CI for Annualized Sharpe')
ax2.set_ylabel('Sharpe Ratio')
ax2.set_xticklabels(ci_df.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()


# In[44]:


import numpy as np
import pandas as pd
from scipy import stats

def diebold_mariano(series1, series2, h=1, power=1):
    """
    Perform the Diebold-Mariano test for equal predictive accuracy
    between two loss/error series.
    
    series1, series2 : 1-d arrays or pd.Series of daily P&L (or losses)
    h                : forecast horizon (1 for one‐day‐ahead)
    power            : 1 for absolute loss, 2 for squared loss
    
    Returns: DM statistic, two‐sided p‐value
    """
    # compute loss differential d_t = |e1_t|^power - |e2_t|^power
    d = np.abs(series1)**power - np.abs(series2)**power
    T = len(d)
    
    # mean of d
    d_bar = np.mean(d)
    
    # denominator: Newey-West estimate of Var(d_bar)
    # autocovariances up to lag h-1
    gamma = [np.cov(d[:-lag], d[lag:])[0,1] if lag>0 else np.var(d, ddof=0)
             for lag in range(h+1)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    
    DM_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(DM_stat)))
    return DM_stat, p_value

# === assume you have returns_df from before ===
# columns = ['Logistic','RandomForest','XGBoost','MA_25_55','ARIMA(3,1,2)']

results = []
strategies = list(returns_df.columns)

for i in range(len(strategies)):
    for j in range(i+1, len(strategies)):
        s1, s2 = strategies[i], strategies[j]
        dm_stat, p = diebold_mariano(returns_df[s1], returns_df[s2], h=1, power=1)
        results.append({
            'Model 1': s1,
            'Model 2': s2,
            'DM stat': dm_stat,
            'p-value': p
        })

dm_df = pd.DataFrame(results)
print(dm_df)


# In[ ]:




