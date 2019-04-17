import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from process_dataset import get_vectorized_training_and_test_set
from parameters import *

def get_model(name='NBC'):
    model = None
    if name == 'NBC':
        model = MultinomialNB(alpha=0.1)
    elif name == 'LR':
        model = LogisticRegression(solver='sag')
    elif name == 'LR_CV':
        model = LogisticRegressionCV(cv=5, random_state=0,
                                     max_iter=2,
                                     multi_class='multinomial')
    if IS_DEBUGGING_ON:
        print('model:', name)
    return model

def precision_at_k(y_true, y_pred, k=5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.argsort(y_pred, axis=1)
    y_pred = y_pred[:, ::-1][:, :k]
    arr = [y in s for y, s in zip(y_true, y_pred)]
    return np.mean(arr)

def train_and_test_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_val_pred_proba = model.predict_proba(X_val)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)

    print('Validation set:\nprecision@1 = {}\nprecision@3 = {}\nprecision@5 = {}'.format(
        precision_at_k(y_val, y_val_pred_proba, 1),
        precision_at_k(y_val, y_val_pred_proba, 3),
        precision_at_k(y_val, y_val_pred_proba, 5)))

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    print('Test set:\nprecision@1 = {}\nprecision@3 = {}\nprecision@5 = {}'.format(
        precision_at_k(y_test, y_pred_proba, 1),
        precision_at_k(y_test, y_pred_proba, 3),
        precision_at_k(y_test, y_pred_proba, 5)))
    return model

def main():
    model = get_model(TRADITIONAL_MODEL_NAME)

    (X_train, y_train, X_test, y_test) = get_vectorized_training_and_test_set()
    print('model', model)
    print(f'DATASET_SIZE: {DATASET_SIZE}\nexperiment_name: {experiment_name}')

    validation_set_size = int(VALIDATION_FRACTION * X_train.shape[0])
    X_val = X_train[-validation_set_size:]
    y_val = y_train[-validation_set_size:]
    training_set_size = int(TRAINING_FRACTION * (1-VALIDATION_FRACTION) * X_train.shape[0])
    X_train = X_train[:training_set_size]
    y_train = y_train[:training_set_size]

    print(f'VALIDATION_FRACTION: {VALIDATION_FRACTION}\nvalidation_set_size: {validation_set_size}\nTRAINING_FRACTION: {TRAINING_FRACTION}\ntraining_set_size: {training_set_size}')
    train_and_test_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == '__main__':
    main()
