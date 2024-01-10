import os
import argparse
import pandas as pd
import warnings

from args import parse_args
from src.utils import Setting, models_load

from src.data_preprocess.tabnet_data import tabnet_preprocess_data, tabnet_dataloader, tabnet_datasplit

from src.train import train, test

warnings.filterwarnings('ignore')

def main(args) : 
    Setting.seed_everything(args.seed)
    setting = Setting()

    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('tabnet'):
        data = tabnet_dataloader(args)
    
    else : 
        pass
    print(f'--------------- {args.model} Load Data Done!---------------')


    ######################## DATA PREPROCESS
    print(f'--------------- {args.model} Data PREPROCESSING---------------')
    if args.model in ('tabnet'):
        data, cat_dims1, categories = tabnet_preprocess_data(args, data)
    
    else : 
        pass
    print(f'--------------- {args.model} Data PREPROCESSING Done!---------------')

    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model in ('tabnet'):
        X_train, y_train, X_valid, y_valid = tabnet_datasplit(args, data)

    else : 
        pass    
    print(f'--------------- {args.model} Train/Valid Split Done!---------------')

    ######################## MODEL LOAD
    print(f'--------------- {args.model} MODEL LOAD---------------')

    cat_idxs = [ i for i, f in enumerate(data.columns) if f in categories ]
    cat_dims = [ cat_dims1[i] for i in categories ] 

    model = models_load(args, cat_idxs, cat_dims)
    
    
    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    
    model = train(args, model, X_train, y_train, X_valid, y_valid)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = test(args, model, data)

    filename = setting.get_submit_filename(args)
    submission = pd.read_csv(args.data_dir + "sample_submission.csv")
    submission['prediction'] = predicts
    submission.to_csv(filename, index = False)
    print('make csv file !!!', filename)
    
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name = args.model_dir, exist_ok = True)
    os.makedirs(name = args.submit_dir, exist_ok = True)
    main(args = args)