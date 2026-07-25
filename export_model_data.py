"""Export feature importance, out-of-fold predictions and metrics for the Power BI report."""

import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings('ignore')

FEATURES = ['Pclass', 'Sex', 'AgeGroup', 'FareGroup', 'Embarked',
            'FamilySize', 'IsAlone', 'Title', 'Family_Survival']

FEATURE_LABELS = {
    'Pclass': 'Passenger Class',
    'Sex': 'Sex',
    'AgeGroup': 'Age Group',
    'FareGroup': 'Fare Band',
    'Embarked': 'Port of Embarkation',
    'FamilySize': 'Family Size',
    'IsAlone': 'Travelling Alone',
    'Title': 'Social Title',
    'Family_Survival': 'Family Survival',
}

KAGGLE_PUBLIC_SCORE = 0.80861

df = pd.read_csv('train_tratado.csv')
X = df[FEATURES]
y = df['Survived'].astype(int)

model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=1)
model.fit(X, y)

importance = pd.DataFrame({
    'Feature': [FEATURE_LABELS[f] for f in FEATURES],
    'Feature_Raw': FEATURES,
    'Importance': model.feature_importances_.round(4),
}).sort_values('Importance', ascending=False).reset_index(drop=True)
importance['Rank'] = importance.index + 1
importance.to_csv('ml_feature_importance.csv', index=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
predicted = cross_val_predict(model, X, y, cv=cv)
probability = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

predictions = pd.DataFrame({
    'PassengerId': df['PassengerId'],
    'Actual': y,
    'Predicted': predicted,
    'Survival_Probability': probability.round(4),
})
predictions['Is_Correct'] = (predictions['Actual'] == predictions['Predicted']).astype(int)
predictions['Result'] = [
    'True Positive' if a == 1 and p == 1 else
    'True Negative' if a == 0 and p == 0 else
    'False Positive' if a == 0 and p == 1 else
    'False Negative'
    for a, p in zip(predictions['Actual'], predictions['Predicted'])
]
predictions['Result_Order'] = predictions['Result'].map({
    'True Negative': 1, 'False Positive': 2, 'False Negative': 3, 'True Positive': 4})
predictions['Actual_Label'] = predictions['Actual'].map({1: 'Survived', 0: 'Died'})
predictions['Predicted_Label'] = predictions['Predicted'].map({1: 'Survived', 0: 'Died'})
predictions.to_csv('ml_predictions.csv', index=False)

metrics = pd.DataFrame([
    ('Accuracy', accuracy_score(y, predicted), 'Share of passengers classified correctly'),
    ('Precision', precision_score(y, predicted), 'Of those predicted to survive, how many did'),
    ('Recall', recall_score(y, predicted), 'Of those who survived, how many were caught'),
    ('F1 Score', f1_score(y, predicted), 'Balance between precision and recall'),
    ('ROC AUC', roc_auc_score(y, probability), 'Ability to rank survivors above non-survivors'),
    ('Kaggle Public Score', KAGGLE_PUBLIC_SCORE, 'Accuracy on the unseen Kaggle test set'),
], columns=['Metric', 'Value', 'Description'])
metrics['Value'] = metrics['Value'].round(4)
metrics.to_csv('ml_metrics.csv', index=False)

print(importance[['Rank', 'Feature', 'Importance']].to_string(index=False))
print()
print(metrics[['Metric', 'Value']].to_string(index=False))
print()
print(pd.crosstab(predictions['Actual_Label'], predictions['Predicted_Label'],
                  rownames=['Actual'], colnames=['Predicted']))
