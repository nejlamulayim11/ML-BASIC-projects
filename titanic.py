import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Veri okuma
data = pd.read_csv("train.csv")

# Eksik değerleri doldurma ve gereksiz sütunları kaldırma
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].map({'C':0,'Q':1,'S':2}).fillna(2)
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data = data.drop(columns=['Name','Ticket','Cabin'])

# Özellik ve hedef
X = data.drop(columns=['Survived'])
y = data['Survived']

# Eğitim-test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model kurma
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Titanic Doğruluk Oranı:", accuracy_score(y_test, y_pred))
