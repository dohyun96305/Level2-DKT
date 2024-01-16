import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def mean(s):
    return np.sum(s) / len(s)

def mean1(s) : 
    return np.sum(s[:(len(s)-1)]) / (len(s)-1)

def groupby_answer(df, str1) :  
    df_groupby_answer = df[df['answerCode'] != -1].groupby([str1]).agg({
        str1 : 'count',
        'answerCode' : mean
    })

    df_groupby_answer.columns = [f'{str1}_count', f'{str1}_answer']
    df = pd.merge(df, df_groupby_answer, on=[str1], how='left')
    return df

def user_groupby_answer(df, str1) : 
    df_groupby_answer = df[df['answerCode'] != -1].groupby(['userID', str1]).agg({
        str1 : 'count',
        'answerCode' : mean
    })

    df_groupby_answer.columns = [f'user_{str1}_count', f'user_{str1}_answer']
    df = pd.merge(df, df_groupby_answer, on=['userID', str1], how='left')
    return df

def shift_num(df, int1) : 
    df[f'user_correct_{-1 * int1}'] = df.groupby('userID')['answerCode'].shift(int1).fillna(0).apply(lambda x : 0 if x == -1 else x)
    return df

def beforecorrect_tag(df, str1) : 
    df[f'user_{str1}_beforecount'] = df.groupby(["userID", str1])["answerCode"].cumcount() # 문제를 풀기 전 맞은 문항의 누적 개수
    df[f"user_{str1}_beforecorrect"] = (df.groupby(["userID", str1])["answerCode"].transform(lambda x: x.cumsum().shift(1)).fillna(0)) # 문제를 풀기 전 맞은 문항의 누적 합
    df[f"user_{str1}_beforeanswerrate"] = (df[f"user_{str1}_beforecorrect"] / df[f'user_{str1}_beforecount']).fillna(0.0) # 문제를 풀기 전 누적 정답률
    return df

def tabnet_preprocess_data(args, df) :
    '''
    Parameters
    -----------
    -----------        
    '''

    df['month'] = df['Timestamp'].dt.month.astype('int64') 
    df['day'] = df['Timestamp'].dt.day.astype('int64') 
    df['weekday'] = df['Timestamp'].dt.weekday.astype('int64') 
    df['hour'] = df['Timestamp'].dt.hour.astype('int64')
    df['min'] = df['Timestamp'].dt.minute.astype('int64') 

    df['first3'] = df['assessmentItemID'].apply(lambda x : x[2]).astype('int64') 
    df['mid3'] = df['assessmentItemID'].apply(lambda x : x[4:7]).astype('int64') 
    df['last3'] = df['assessmentItemID'].apply(lambda x : x[-3:]).astype('int64') 

    # KnowledgeTag에 따른 정답률
    df = groupby_answer(df, 'KnowledgeTag')
    df = user_groupby_answer(df, 'KnowledgeTag')
    
    df['user_KnowledgeTag_answer'] = df['user_KnowledgeTag_answer'].fillna(0.0)
    df['user_KnowledgeTag_count'] = df['user_KnowledgeTag_count'].fillna(0.0)

    # (first3, mid3)에 따른 count, 정답률
    df = groupby_answer(df, 'first3')
    df = user_groupby_answer(df, 'first3')

    df = groupby_answer(df, 'mid3')
    df = user_groupby_answer(df, 'mid3')

    # (month, weekday, hour)에 따른 count, 정답률
    df = groupby_answer(df, 'month')
    df = user_groupby_answer(df, 'month')

    df = groupby_answer(df, 'weekday')
    df = user_groupby_answer(df, 'weekday')

    df = groupby_answer(df, 'hour')
    df = user_groupby_answer(df, 'hour')

    df['user_hour_count'] = df['user_hour_count'].fillna(0.0)
    df['user_hour_answer'] = df['user_hour_answer'].fillna(0.0)

    # 시험지 내 문제를 푸는데 걸린 시간
    diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    diff = diff.fillna(pd.Timedelta(seconds=0))
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

    df['elapsed'] = diff.shift(-1)

    user_test_groupby_answer = df.groupby(['userID', 'testId']).agg({
        'elapsed' : mean1
    })

    user_test_groupby_answer.columns = ['test_elapsed_mean']
    df = pd.merge(df, user_test_groupby_answer, on=['userID', 'testId'], how='left')
        
    df.loc[df.groupby(['userID', 'testId']).tail(1).index, 'elapsed'] = df['test_elapsed_mean']
    df = df.drop(['test_elapsed_mean'], axis = 1)
    ###

    # 문제를 풀기 전 시점의 (문항, 맞은 문항)의 누적 개수, 누적 정답률
    df["user_beforecorrect"] = (df.groupby("userID")["answerCode"].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)) 
    df["user_beforecount"] = df.groupby("userID")["answerCode"].cumcount() 
    df["user_beforeanswerrate"] = (df['user_beforecorrect'] / df['user_beforecount']).fillna(0.0) 

    df = beforecorrect_tag(df, 'first3')
    df = beforecorrect_tag(df, 'mid3')
    df = beforecorrect_tag(df, 'KnowledgeTag')

    # 문제를 푸는 시점에서 user의 과거, 미래 시점 정답 여부
    for i in range(-2, 3) : 
        if i == 0 : 
            pass
        else : 
            df = shift_num(df, i)
    
    # rolling3_time => 3개 문제 풀떄 걸린 시간의 이동평균 
    df['rolling3_time'] = df.groupby(['userID'])['elapsed'].rolling(3).mean().fillna(0).values
    df.loc[df.groupby(['userID']).head(1).index, 'rolling3_time'] = df.loc[df.groupby(['userID']).head(1).index, 'elapsed']

    for group, group_df in df.groupby('userID'):
        first_two_elapsed_mean = group_df['elapsed'].head(2).mean()
        df.loc[group_df.index[1], 'rolling3_time'] = first_two_elapsed_mean            

    df = df.drop(['Timestamp'], axis = 1)
    
    col = df.columns.tolist()
    new_col = col[:3] + col[4:] + [col[3]]
    df = df[new_col]

    categories = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 'month', 'day', 'weekday', 'hour', 'min', 'first3', 'mid3', 'last3']    
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

