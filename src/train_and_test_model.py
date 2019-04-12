import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from process_dataset import get_vectorized_training_and_test_set
from parameters import IS_DEBUGGING_ON

def get_model(name='NBC'):
    model = None
    if name == 'NBC':
        model = MultinomialNB(alpha=0.1)
    elif name == 'LR':
        model = LogisticRegression(solver='sag')
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

def train_and_test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # we use precision-at-k metrics to evaluate performance
    # (https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K)

    # print(f'{y_test[:10]}')
    # print(f'{y_pred[:10]}')

    # Got DeprecationWarning.
    # print('precision@1 =', np.mean(y_test == y_pred))
    print('precision@1 =', precision_at_k(y_test, y_pred_proba, 1))
    print('precision@3 =', precision_at_k(y_test, y_pred_proba, 3))
    print('precision@5 =', precision_at_k(y_test, y_pred_proba, 5))

    # RUNNING_KAGGLE_KERNEL == True
    # precision@1 = 0.610528134254689
    # precision@3 = 0.7573692003948668
    # precision@5 = 0.8067670286278381

    # RUNNING_KAGGLE_KERNEL == False
    # precision@1 = 0.7292102665350444
    # precision@3 = 0.8512240868706812
    # precision@5 = 0.8861500493583415

    # precision@1 = 0.615187561697927
    # precision@3 = 0.7615399802566634
    # precision@5 = 0.8105972359328727

def main():
    # model_name = 'NBC'
    model_name = 'LR'
    model = get_model(model_name)

    print('model', model)
    (X_train, y_train, X_test, y_train) = get_vectorized_training_and_test_set()
    train_and_test_model(model, X_train, y_train, X_test, y_train)

if __name__ == '__main__':
    main()
