import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

filename = 'data/bank/bank-full.csv'
df = pd.read_csv(filename, sep=';')

X = df.drop(columns='deposit') 
y = df['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, stratify=y, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_name = 'data/bank/train.csv'
test_name = 'data/bank/test.csv'

train_data.to_csv(train_name, index=0)
test_data.to_csv(test_name, index=0)