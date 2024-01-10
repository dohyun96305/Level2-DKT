import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def percentile(s):
    return np.sum(s) / len(s)

def mean1(s) : 
    return np.sum(s[:(len(s)-1)]) / (len(s)-1)

def tabnet_preprocess_data(args, df) :
    '''
    Parameters
    -----------
    -----------        
    '''

    df['month'] = df['Timestamp'].dt.month.astype('int64')
    df['day'] = df['Timestamp'].dt.day.astype('int64')
    df['hour'] = df['Timestamp'].dt.hour.astype('int64')

    # 유저, 태그에 따른 문제 정답률 (tag_answer_precentile)
    user_tag_groupby_answer = df[df['answerCode'] != -1].groupby(['userID', 'KnowledgeTag']).agg({
        'KnowledgeTag' : 'count',
        'answerCode' : percentile
    })

    user_tag_groupby_answer.columns = ['user_tag_count', 'user_tag_answer']
    
    df = pd.merge(df, user_tag_groupby_answer, on=['userID', 'KnowledgeTag'], how='left')

    df.loc[df['user_tag_count'].isna() == True, 'user_tag_count'] = 0
    df.loc[df['user_tag_answer'].isna() == True, 'user_tag_answer'] = 0  
          
    # 유저가 문제를 풀 때 걸린 시간 (elapsed)
    diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    diff = diff.fillna(pd.Timedelta(seconds=0))
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

    new_diff = diff[1:].reset_index(drop = True)
    new_diff[(len(df)-1)] = 0.0

    df['elapsed'] = new_diff

    # 유저, 시험지에 따른 평균 정답률 (test_answer_percentile)
    user_test_groupby_answer = df[df['answerCode'] != -1].groupby(['userID', 'testId']).agg({
        'testId' : 'count',
        'answerCode' : percentile,
        'elapsed' : mean1
    })

    user_test_groupby_answer.columns = ['user_test_count', 'user_test_answer', 'user_test_time']

    df = pd.merge(df, user_test_groupby_answer, on=['userID', 'testId'], how='left')
    df.loc[df.groupby(['userID', 'testId']).tail(1).index, 'elapsed'] = df['user_test_time']

    df = df.drop(['Timestamp', 'user_test_time'], axis = 1)
    
    col = df.columns.tolist()
    new_col = col[:3] + col[4:] + [col[3]]
    df = df[new_col]

    categories = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 'month', 'day', 'hour']    
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
    data = data.reset_index(drop = True)

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

