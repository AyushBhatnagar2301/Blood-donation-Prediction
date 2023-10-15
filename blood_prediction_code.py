import pandas as pd
import numpy as np
blood_dataset = pd.read_csv('C:/Blood donation prediction/archive/transfusion.data')
blood_dataset.head()
blood_dataset.info()
blood_dataset.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)
print(blood_dataset.head(2))
print(blood_dataset.target.value_counts(normalize=True).round(3))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    blood_dataset.drop(columns='target'),
    blood_dataset.target,
    test_size=0.25,
    random_state=42,
    stratify=blood_dataset.target
)
print(X_train.head(3))
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f'{idx}. {transform}')

X_train.var().round(3)

X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

col_to_normalize = 'Monetary (c.c. blood)'

for df_ in [X_train_normed, X_test_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(columns=col_to_normalize, inplace=True)

X_train_normed.var().round(3)

from sklearn import linear_model

logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

logreg.fit(X_train_normed, y_train)

logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')

from operator import itemgetter

sorted(
    [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)],
    key=itemgetter(1),
    reverse=True)