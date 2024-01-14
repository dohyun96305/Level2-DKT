import os
import time
import numpy as np
import random

from .models import *

class Setting:
    @staticmethod
    def seed_everything(seed):
        '''
        [description]
        seed 값을 고정시키는 함수입니다.

        [arguments]
        seed : seed 값
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
    
    def __init__(self):
        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        save_time = now_date + '_' + now_hour.replace(':', '')
        self.save_time = save_time

    def make_dir(self, path):
        '''
        [description]
        경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.

        [arguments]
        path : 경로

        [return]
        path : 경로
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path
    
    def get_submit_filename(self, args, auc, acc):
        '''
        [description]
        submit file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        filename : submit file을 저장할 경로를 반환합니다.
        이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
        '''
        filename = f'./submits/{self.save_time}_{args.model}_auc_{auc:.4f}_acc_{acc:.4f}.csv'
        return filename
    

def models_load(args, cat_idxs, cat_dims):
        '''
        [description]
        입력받은 args 값에 따라 모델을 선택하며, 모델이 존재하지 않을 경우 ValueError를 발생시킵니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델을 선택합니다.
        data : data는 data_loader로 처리된 데이터를 의미합니다.
        '''

        if args.model=='tabnet':
            model = tabnet(args, cat_idxs, cat_dims)

        else:
            raise ValueError('MODEL is not exist')

        return model
