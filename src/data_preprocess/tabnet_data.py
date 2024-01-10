import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def tabnet_preprocess_data(args, df) :
    '''
    Parameters
    -----------
    -----------        
    '''

    df['month'] = df['Timestamp'].dt.month.astype('int64')
    df['day'] = df['Timestamp'].dt.day.astype('int64')
    df['hour'] = df['Timestamp'].dt.hour.astype('int64')

    df = df.drop(['Timestamp'], axis = 1)
    col = df.columns.tolist()
    new_col = col[:2] + col[4:] + [col[3]]
    df = df[new_col]

    categories = new_col[:-1]
    cat_dims1 = {}

    for col in df.columns :
        if col in categories : 
            df[col] = df[col].astype('category')

            l_enc = LabelEncoder()
            df[col] = l_enc.fit_transform(df[col].values)
            cat_dims1[col] = len(l_enc.classes_)

    return df, cat_dims1, categories

def tabnet_dataloader(args) :
    '''
    Parameters
    -----------
    Args : 
        data_dir : str
            데이터 위치 경로
    -----------        
    '''

    train_data = pd.read_csv(args.data_dir + 'train_data.csv', parse_dates=["Timestamp"])
    test_data = pd.read_csv(args.data_dir + 'test_data.csv', parse_dates=["Timestamp"])

    data = pd.concat([train_data, test_data])

    return data

def tabnet_datasplit(args, df) :
    '''
    Parameters
    ----------
    Args :
        valid_size : float
            Train/Valid split 비율
        seed : int
            랜덤 seed 값
    ----------
    '''
    
    train, valid = train_test_split(df[df['answerCode'] != -1], test_size = args.valid_size, random_state = args.seed)

    y_train = train['answerCode'].values
    X_train = train.drop(['answerCode'], axis = 1).values

    y_valid = valid['answerCode'].values
    X_valid = valid.drop(['answerCode'], axis = 1).values

    return X_train, y_train, X_valid, y_valid

