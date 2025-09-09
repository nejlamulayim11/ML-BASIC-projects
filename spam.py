import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Veri okuma
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1','v2']]  # Kaggle dataset sütunları
data.columns = ['label','text']

# Özellik ve hedef
X = data['text']
y = data['label']

# Metin verisini sayısallaştırma
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Eğitim-test bölme
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Spam Tespiti Doğruluk Oranı:", accuracy_score(y_test, y_pred))
