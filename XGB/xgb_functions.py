import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

import holidays




# GET INDEXES OF DATA SUBSEQUENCES (ONE SUBSEQUENCE INCLUDES LAGGED (X) AND HORIZON (Y) LOADS)

def get_subsequence_indexes(in_data_df, window_size, step_size):
    # in_data_df = full (or train/test) dataset, window_size = number of lagged values + predicted (horizon) values, step_size = spacing between datapoints

    last_index = len(in_data_df)
    
    subseq_first_idx = 0 # Subsequence start and end index
    subseq_last_idx = window_size
    
    subseq_indexes = []
    
    while subseq_last_idx <= last_index: # Divide all data into subsequences (and get their indexes)

        subseq_indexes.append((subseq_first_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_last_idx += step_size

    return subseq_indexes




# GET X,Y DATA SPLIT (EVERY DATAPOINT IS MADE UP OF SUB-SEQUENCE)

def get_xy_lagged(subseq_indexes, load_data, horizon_size, lag_size):

    for i, idx in enumerate(subseq_indexes):

        # Create subsequences
        subsequence = load_data[idx[0]:idx[1]] # Flat np array

        xi = subsequence[0: lag_size]
        yi = subsequence[lag_size: lag_size + horizon_size]

        if i == 0: # No existing array to append to
            y = np.array([yi]) # Turn y and x into rows, to make an array of arrays
            x = np.array([xi])

        else:
            y = np.concatenate((y, np.array([yi])), axis=0) # shape (datapoints, horizon)
            x = np.concatenate((x, np.array([xi])), axis=0) # shape (datapoints, input features)

    return x, y




# GET DATETIME FEATURES

def create_dt_features(df):
    df_c = df.copy()
    df_c['Hour'] = df_c.index.hour
    df_c['Workday'] = df_c.index.map(lambda x: 0 if (x in holidays.Netherlands() or x.dayofweek in (5,6)) else 1) # 1 if workday, 0 if holiday or weekend
    df_c['Dayofweek'] = df_c.index.dayofweek
    df_c['Quarter'] = df_c.index.quarter
    df_c['Month'] = df_c.index.month
    df_c['Dayofyear'] = df_c.index.dayofyear
    df_c['Dayofmonth'] = df_c.index.day
    return df_c




# GET CYCLICAL ENCODING

def cyclical_encoding(df, features):
    df_c = df.copy()
    
    for f in features:
        total_values = df_c[f].max() # E.g. total months = 12, starting at 1

        if df_c[f].min() == 0: # If first value is 0, total values is 1 more e.g. 24 hours
            total_values += 1 

        df_c[f + '_cos'] = np.cos(2*np.pi* df_c[f]/ total_values) # Encode into cos and sin values, that way end value is close to start value
        df_c[f + '_sin'] = np.sin(2*np.pi* df_c[f]/ total_values)
    
    return df_c




# GET FORMATTED DATA (IN DATE-TIME) AND TRAIN-TEST SPLIT

def date_formatting(bus):

    df = pd.read_csv(f"./data/bus_{bus}_load.csv")
    df.index = pd.to_datetime(df.index, unit='h', origin=pd.Timestamp("2018-01-01"))
    df.index.name = "Time"
    df = create_dt_features(df)
    df = cyclical_encoding(df, features=["Hour", "Dayofweek", "Quarter", "Month", "Dayofyear", "Dayofmonth"])

    # training_data = df[df.index < split_date] # Splitting data now will cause it to lose 24 training datapoints and 24*7 test datapoints
    # test_data = df[df.index >= split_date]

    return df

def allbus_date_formatting(buses=list(range(1, 29))):
    
    allbus_df = {}

    for b in buses:
        allbus_df[b] = date_formatting(b)

    return allbus_df




# GET X,Y DATA 

def get_xy(in_data_df, step_size, horizon_size, hyperparameters):

    subseq_indexes = get_subsequence_indexes(in_data_df = in_data_df, window_size = hyperparameters["lag_size"] + horizon_size, step_size = step_size)
    
    lagged_x, y = get_xy_lagged(subseq_indexes=subseq_indexes, load_data=in_data_df[hyperparameters["selected_features"][0]].to_numpy(), 
                                horizon_size=horizon_size, lag_size=hyperparameters["lag_size"])     
    
    no_lag_features = in_data_df[hyperparameters["selected_features"][1:]].to_numpy() # Array of features that are not lagged, by rows (each row a timestep)

    first_timestep = hyperparameters["lag_size"] # Datapoints start after all lagged values can be obtained
    last_timestep = len(in_data_df) - (horizon_size - 1) # Last datapoint until it is possible to obtain all horizon values (horizon_size - 1)

    x = np.append(lagged_x, no_lag_features[first_timestep: last_timestep], axis=1) # Append no-lag features to lagged load features

    return x, y


def get_allbus_xy(allbus_in_data, hyperparameters, buses=list(range(1, 29)), step_size=1, horizon_size=24):

    allbus_x, allbus_y = {}, {}

    for b in buses:
        allbus_x[b], allbus_y[b] = get_xy(allbus_in_data[b], step_size, horizon_size, hyperparameters)
    
    return allbus_x, allbus_y




# GET X,Y SPLIT IN TRAIN, TEST

def split_train_test(x, y, lag_size, split_date="2018-10-16", train_val_split=0.8): # Timestep 6912

    original_timestep = (pd.Timestamp(split_date) - pd.Timestamp("2018-01-01 00:00:00")).total_seconds()/3600 # Get original timestep (hour) from split date
    split_timestep = int(original_timestep - lag_size) # Split timestep in the new dataframe (starts at timestep lag_size)
    
    x_tr = x[:split_timestep]
    y_tr = y[:split_timestep]

    x_train, x_val = x_tr[:int(train_val_split * len(x_tr))], x_tr[int(train_val_split * len(x_tr)):] # Split in train and validation
    y_train, y_val = y_tr[:int(train_val_split * len(y_tr))], y_tr[int(train_val_split * len(y_tr)):]

    x_test = x[split_timestep:]
    y_test = y[split_timestep:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def allbus_split_train_test(allbus_x, allbus_y, lag_size, buses=list(range(1, 29)), split_date="2018-10-16", train_val_split=0.8):

    allbus_x_train, allbus_y_train, allbus_x_val, allbus_y_val, allbus_x_test, allbus_y_test = {}, {}, {}, {}, {}, {}

    for b in buses:
        allbus_x_train[b], allbus_y_train[b], allbus_x_val[b], allbus_y_val[b], allbus_x_test[b], allbus_y_test[b] = \
            split_train_test(allbus_x[b], allbus_y[b], lag_size, split_date, train_val_split)
    
    return allbus_x_train, allbus_y_train, allbus_x_val, allbus_y_val, allbus_x_test, allbus_y_test




# GET AND STORE ALL MODELS FOR ALL BUSSES AND THEIR TRAIN/TEST SCORES

def get_model(x_train, y_train, x_val, y_val, x_test, y_test, hyperparameters, score_function):

    if x_val == "None" or y_val == "None":   
        model = XGBRegressor(n_estimators=hyperparameters["n_estimators"], max_depth=hyperparameters["max_depth"], subsample=hyperparameters["subsample"], 
                            gamma=hyperparameters["gamma"], reg_lambda=hyperparameters["lambda"], objective="reg:squarederror", tree_method="hist", 
                            verbosity=3, learning_rate=hyperparameters["learning_rate"])
        
        trained_model = MultiOutputRegressor(model).fit(x_train, y_train, verbose=True) # Evaluate on validation data

        valid_score = None
    
    
        
    else:
        model = XGBRegressor(n_estimators=hyperparameters["n_estimators"], max_depth=hyperparameters["max_depth"], subsample=hyperparameters["subsample"], 
                            gamma=hyperparameters["gamma"], reg_lambda=hyperparameters["lambda"], objective="reg:squarederror", tree_method="hist", 
                            verbosity=3, learning_rate=hyperparameters["learning_rate"], early_stopping_rounds=hyperparameters["early_stopping_rounds"])
        
        trained_model = MultiOutputRegressor(model).fit(x_train, y_train, fit_params={"eval_set": [(x_val, y_val)]}, verbose=True) 
        # Evaluate on validation data

        valid_forecasts = trained_model.predict(x_val)
        valid_score = score_function(y_val, valid_forecasts)

    train_forecasts = trained_model.predict(x_train)
    train_score = score_function(y_train, train_forecasts)

    test_forecasts = trained_model.predict(x_test)
    test_score = score_function(y_test, test_forecasts)

    model_score = {"Train score": train_score, "Validation score": valid_score, "Test score": test_score}

    return trained_model, model_score


def get_models(models_datapath, hyperparameters, score_function, buses=list(range(1, 29)), horizon_size=24, step_size=1, 
               allbus_x_train="None", allbus_y_train="None", allbus_x_val="None", allbus_y_val="None", allbus_x_test="None", allbus_y_test="None"):

    trained_models = {}
    model_scores = {}
    
    for bus in buses:

        if (allbus_x_train != "None") and (allbus_y_train != "None") and (allbus_x_test != "None") and (allbus_y_test != "None"): # If they're all defined
            x_train, y_train, x_val, y_val, x_test, y_test = allbus_x_train[bus], allbus_y_train[bus], allbus_x_val[bus] if allbus_x_val != "None" else "None", \
            allbus_y_val[bus] if allbus_y_val != "None" else "None", allbus_x_test[bus], allbus_y_test[bus]
        
        else:
            print("Missing input data to get_models()")

            df = date_formatting(bus)

            x, y = get_xy(df, step_size, horizon_size, hyperparameters)

            x_train, y_train, x_val, y_val, x_test, y_test = split_train_test(x, y, hyperparameters["lag_size"])
        
        
        trained_model, model_scores[bus] = get_model(x_train, y_train, x_val, y_val, x_test, y_test, hyperparameters, score_function)
        
        trained_models[bus] = trained_model

        # Store models
        with open(f"{models_datapath}/MOR_bus{bus}.pkl", "wb") as f1:
            pickle.dump(trained_model, f1)

    # Store model scores
    with open(f"{models_datapath}/MOR_scores.pkl", "wb") as f2:
                pickle.dump(model_scores, f2)

    return trained_models, model_scores





# --------------------------------------------- FEATURE IMPORTANCES ------------------------------------------------------


def get_boxplot(chosen_buses, trained_models, feature_names):
    allbus_importances = []
    allbus_importance_names = []

    for bus in chosen_buses:
        allbus_importances.append(np.concatenate([trained_models[bus].estimators_[t].feature_importances_ for t in range(24)]))
        allbus_importance_names.append(np.concatenate([feature_names for _ in range(24)]))
    
    importance = np.concatenate(allbus_importances)
    ft_name = np.concatenate(allbus_importance_names)

    feature_importance_df = pd.DataFrame({"Feature": ft_name, "Importance": importance})

    sorting = feature_importance_df.groupby("Feature").mean().sort_values(by="Importance", ascending=False)

    plot = sns.boxplot(data=feature_importance_df, x="Feature", y="Importance", order=sorting.index).set_xticklabels(sorting.index, rotation=80)

    return feature_importance_df, sorting, plot





    