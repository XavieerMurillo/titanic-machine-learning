"""Build the treated Titanic dataset used as the Power BI source."""

import warnings

import pandas as pd

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
full_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

full_data['Age'] = full_data['Age'].fillna(full_data['Age'].median())
full_data['Embarked'] = full_data['Embarked'].fillna('S')
full_data['Fare'] = full_data['Fare'].fillna(full_data['Fare'].median())

full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
full_data['IsAlone'] = (full_data['FamilySize'] == 1).astype(int)

full_data['Title'] = full_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
full_data['Title'] = full_data['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
full_data['Title'] = full_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
full_data['Title'] = full_data['Title'].replace('Mme', 'Mrs')
full_data['Title'] = full_data['Title'].map(
    {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}).fillna(0)

full_data.loc[full_data['Age'] <= 12, 'AgeGroup'] = 0
full_data.loc[(full_data['Age'] > 12) & (full_data['Age'] <= 18), 'AgeGroup'] = 1
full_data.loc[(full_data['Age'] > 18) & (full_data['Age'] <= 50), 'AgeGroup'] = 2
full_data.loc[full_data['Age'] > 50, 'AgeGroup'] = 3
full_data['AgeGroup'] = full_data['AgeGroup'].astype(int)

full_data['Sex'] = full_data['Sex'].replace({'female': 1, 'male': 0})
full_data['Embarked'] = full_data['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
full_data['FareGroup'] = pd.qcut(full_data['Fare'], 4, labels=[0, 1, 2, 3])

full_data['Surname'] = full_data['Name'].apply(lambda name: name.split(',')[0].strip())
full_data['Family_ID'] = full_data['Surname'] + '_' + full_data['Fare'].astype(str)
full_data['Family_Survival'] = 0.5

for _, group in full_data.groupby('Family_ID'):
    if len(group) == 1:
        continue
    for index in group.index:
        relatives = group.drop(index)['Survived'].dropna()
        if len(relatives) == 0:
            continue
        if relatives.mean() == 1.0:
            full_data.loc[index, 'Family_Survival'] = 1.0
        elif relatives.mean() == 0.0:
            full_data.loc[index, 'Family_Survival'] = 0.0

train_set = full_data[full_data['Survived'].notnull()].copy()
train_set['Survived'] = train_set['Survived'].astype(int)

train_set = train_set[[
    'PassengerId', 'Survived', 'Pclass', 'Name', 'Surname', 'Sex', 'Age', 'AgeGroup',
    'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Ticket', 'Fare', 'FareGroup', 'Cabin',
    'Embarked', 'Title', 'Family_ID', 'Family_Survival',
]]

train_set.to_csv('train_tratado.csv', index=False)

print(f"train_tratado.csv written: {train_set.shape[0]} rows, {train_set.shape[1]} columns")
