from sklearn.metrics import accuracy_score, roc_auc_score

def train(args, model, X_train, y_train, X_valid, y_valid) : 

    model.fit(
        X_train = X_train, y_train = y_train,
        eval_set = [(X_train, y_train), (X_valid, y_valid)],
        eval_name = ['train', 'valid'],
        eval_metric = ['auc', 'accuracy'],
        max_epochs = args.n_epochs, 
        patience = 20,
        batch_size = args.batch_size, 
        virtual_batch_size = 128,
        num_workers = 0,
        weights = 1,
        drop_last = False,
    )

    auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, -1])
    acc = accuracy_score(y_valid, model.predict(X_valid))

    return model, auc, acc

def test(model, df) : 

    X_test = df[df['answerCode'] == -1].drop(['answerCode'], axis = 1).values
    pred = model.predict_proba(X_test)[:,-1]

    return pred
