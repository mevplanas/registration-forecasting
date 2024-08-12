import pandas as pd 
import numpy as np 

# Train test spliting 
from sklearn.model_selection import train_test_split

# Linear models 
from sklearn.linear_model import LinearRegression 

# Multilabel binarizer 
from sklearn.preprocessing import MultiLabelBinarizer

# Iteration tracking 
from tqdm import tqdm 

def convert_to_timestamp(x):
    try: 
        return pd.to_datetime(x)
    except: 
        return None 

def predict_permits(input, output):

    d = pd.read_parquet(input)

    d["USER_registrationDate"] = [convert_to_timestamp(x) for x in d["USER_registrationDate"]]
    d['LEIDIMO_DATA'] = [convert_to_timestamp(x) for x in d["LEIDIMO_DATA"]]

    d.dropna(subset=['USER_dateOfBirth', 'USER_registrationDate'], inplace=True) 
    # d['ageAtRegistration'] = (d['USER_registrationDate'] - d['USER_dateOfBirth']).dt.days // 365

    # Binning the ages to bins of 10 years
    # d['age_bin'] = pd.cut(d['ageAtRegistration'], bins=np.arange(-1, 120, 10))
    # d['age_bin'] = d['age_bin'].astype(str)

    # Converting registrationDate to YYYY-MM format 
    d['timestep'] = d['USER_registrationDate'].dt.to_period('M')

    # Converting the time steps to integers 
    min_timestep = d['timestep'].min()
    max_timestep = d['timestep'].max() 
    timestep_sequence = pd.period_range(min_timestep, max_timestep, freq='M') 
    timestep_df = pd.DataFrame({'timestep': timestep_sequence, 'timestep_int': np.arange(len(timestep_sequence))}) 
    d = pd.merge(d, timestep_df, on='timestep', how='left') 

    # Defining a list of dummy features 
    dummy_features = ['USER_eldership']

    # Dropping the missing rows in the dummy features
    d = d.dropna(subset=dummy_features)

    dd = d[['timestep_int'] + dummy_features].copy() 
    dd = dd.groupby(['timestep_int'] + dummy_features).size().reset_index(name='count')

    # Creating the dummy data 
    dd = pd.get_dummies(dd, columns=dummy_features)

    # Spliting to train and test frames 
    train, test = train_test_split(dd, test_size=0.2, random_state=42) 

    # Creating the x, y pairs
    X_train = train.drop('count', axis=1)
    y_train = train['count']

    X_test = test.drop('count', axis=1)
    y_test = test['count']

    model = LinearRegression()
    model.fit(X_train, y_train) 

    # Saving the features 
    features = X_train.columns

    # Predicting 
    yhat = model.predict(X_train)

    # Calculating the metrics 
    errors = y_train - yhat 
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2 
    print('Mean absolute error:', abs_errors.mean())
    print('Mean squared error:', sq_errors.mean()) 

    # Extracting the coefficients 
    coefs = pd.Series(model.coef_, index=features) 
    coefs = coefs.sort_values(ascending=False) 

    # Adding the intercept
    coefs['intercept'] = model.intercept_ 

    # Predicting on the test set 
    yhat_test = model.predict(X_test) 

    # Calculating the metrics
    errors_test = y_test - yhat_test
    abs_errors_test = np.abs(errors_test)
    sq_errors_test = errors_test ** 2

    # Getting maximum date in the dataset 
    max_date = d['USER_registrationDate'].max() 

    # Defining the number of maximum months to forecast ahead 
    n_months = 24 

    # Creating the future time steps 
    future_timesteps = pd.period_range(max_date, periods=n_months, freq='M') 
    future_timesteps_df = pd.DataFrame({'timestep': future_timesteps, 'timestep_int': np.arange(len(timestep_sequence), len(timestep_sequence) + n_months)})

    # Extracting the unique timesteps 
    unique_timesteps = future_timesteps_df['timestep_int'].unique()

    # For each unique dummy feature, getting all the unique values 
    dummy_features_values = {}
    for dummy_feature in dummy_features:
        dummy_features_values[dummy_feature] = d[dummy_feature].unique().tolist()

    # Creating the meshed grid of all possible combinations of the dummy features
    from itertools import product
    meshed_grid = list(product(*dummy_features_values.values()))

    # Iterating over each timestep to the future and predicing the counts
    future_predictions = []
    for timestep_int in tqdm(unique_timesteps):
        # Iterating over all the tuples 
        for obs in meshed_grid:
            # Creating the prediction frame
            prediction_df = pd.DataFrame({'timestep_int': [timestep_int], **dict(zip(dummy_features, obs))})

            # Creating the dummy data
            prediction_df = pd.get_dummies(prediction_df, columns=dummy_features)

            # Ensuring the columns are the same as the training columns
            missing_columns = set(features) - set(prediction_df.columns)
            for column in missing_columns:
                prediction_df[column] = 0

            # Sorting the columns
            prediction_df = prediction_df[features]

            # Predicting
            prediction = model.predict(prediction_df) 

            # Appending the prediction
            future_predictions.append({'timestep_int': timestep_int, **dict(zip(dummy_features, obs)), 'count': prediction[0]})

    # Creating a dataframe out of the predictions 
    future_predictions_df = pd.DataFrame(future_predictions)

    # Giving the timesteps the correct format
    future_predictions_df = pd.merge(future_predictions_df, future_timesteps_df, on='timestep_int', how='left') 

    future_predictions_df.to_csv(output)
    future_predictions_df.groupby('USER_eldership').plot(x='timestep', y='count')

data = 'registration-forecasting/input/residents1.parquet'
output = 'result_2.csv'

predict_permits(data,output)