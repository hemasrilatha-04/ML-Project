import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


nltk.download('stopwords')

# load dataset
df = pd.read_csv("data/fake_job_postings.csv")
print(df.head())

# keep only needed columns
df['text'] = df['description'].fillna('')
df = df[['text', 'fraudulent']]

# -------- CLEAN TEXT --------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['text'] = df['text'].apply(clean_text)
print(df['text'].head())

# -------- BALANCE DATA --------
df_real = df[df.fraudulent == 0]
df_fake = df[df.fraudulent == 1]

df_fake_upsampled = resample(df_fake,
                            replace=True,
                            n_samples=len(df_real),
                            random_state=42)

df = pd.concat([df_real, df_fake_upsampled]).sample(frac=1)
print(df)


sns.countplot(x=df['fraudulent'])
plt.title("Class Distribution (Real vs Fake Jobs)")
plt.xlabel("0 = Real, 1 = Fake")
plt.ylabel("Count")
plt.show()

# -------- VECTORIZE --------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['fraudulent']
print("✅ Text Vectorization Complete!")
print("Sample Vectorized Text Shape:", X[0].shape)



# -------- SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# -------- TRAIN MODEL --------
model = XGBClassifier(
    scale_pos_weight=len(df_real)/len(df_fake),
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("✅ Model Training Complete!")
print("Sample Prediction:", model.predict(X_test[0]))

# -------- EVALUATE --------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -------- SAVE MODEL --------
