import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Reading files
def read_csv(path):
    return pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/" + path)

data_holidays = read_csv("holidays_events.csv")
data_oil = read_csv("oil.csv")
data_stores = read_csv("stores.csv")
data_transactions = read_csv("transactions.csv")
data_train = read_csv("train.csv")
data_test = read_csv("test.csv")

# Setting up columns to proper date format
data_oil['date'] = pd.to_datetime(data_oil['date'])
data_holidays['date'] = pd.to_datetime(data_holidays['date'])
data_train['date'] = pd.to_datetime(data_train['date'])
data_test['date'] = pd.to_datetime(data_test['date'])
data_transactions['date'] = pd.to_datetime(data_transactions['date'])

# Only keeping rows after 2013-01-01
data_train = data_train[data_train['date'] > '2013-01-01']

def process_data(df: pd.DataFrame, is_test=False):
    # Columns to keep
    col_train = ['date', 'family', 'onpromotion', 'sales', 'store_nbr', 'id']
    col_test = ['date', 'family', 'onpromotion', 'store_nbr', 'id']
    col_holidays = ['date', 'type', 'transferred']

    col_df = col_test if is_test else col_train

    # left Merge to form the dataset on the basis of date
    df = df[col_df].merge(data_oil, 'left', 'date') \
                   .merge(data_holidays[col_holidays], 'left', 'date') \
                   .rename(columns={'type': 'day_type', 'dcoilwtico': 'crude_price'})
    

    
    df['day_type'] = df['day_type'].fillna('Work Day')
    df['transferred'] = df['transferred'].fillna(False)

    # Only keep holidays that are not transferred
    df['is_holiday'] = np.where(
        (df['day_type'] == 'Work Day'), 0,
        np.where(
            (df['day_type'].isin(['Holiday', 'Additional', 'Event', 'Transfer', 'Bridge'])) &
            (df['transferred'] == False),
            1,
            0
        )
    )

    # Update 'onpromotion'
    df['onpromotion'] = np.where(df['onpromotion'] == 0.0, 0, 1)

    # Create a day number column
    df['day_number'] = pd.factorize(df['date'])[0] + 1
    df = df.drop(['day_type', 'transferred', 'date'], axis=1)

    # Fill missing crude prices
    df['crude_price'] = df['crude_price'].fillna(method='ffill')

    # One-hot encode 'family'
    family_encoding = pd.get_dummies(df['family'], prefix='family').astype(int)
    df = pd.concat([df, family_encoding], axis=1)

    # Drop 'family' and any remaining NaNs
    df = df.drop(['family'], axis=1).dropna()

    return df

# Process the train and test data
data_train = process_data(data_train)
data_test = process_data(data_test, is_test=True)

# Prepare data for modeling
train_X = data_train.drop(['sales'], axis=1)
train_y = data_train['sales']

# Adjust column names for XGBoost compatibility
train_X.columns = [col.replace(" ", "_").replace("-", "_") for col in train_X.columns]
data_test.columns = [col.replace(" ", "_").replace("-", "_") for col in data_test.columns]

# Calculate log mean target for the base score as output(y)it is very kewed 
log_mean_target = np.log1p(train_y.mean())
print("Log mean target:", log_mean_target)

# XGBoost model uding poission loss
xgb_model = xgb.XGBRegressor(objective='count:poisson', n_estimators=100, base_score=log_mean_target)

# Train the model
xgb_model.fit(train_X, train_y)

# Make predictions 
predict_y_xg = xgb_model.predict(data_test)

# Save predictions to a new CSV file
end_data = data_test[['id']].copy()
end_data['sales'] = predict_y_xg
end_data.to_csv('/kaggle/working/submission_v2.csv', index=False)

print("Predictions saved to 'submission_v2.csv'.")
 
 