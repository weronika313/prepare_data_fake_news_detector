import pandas as pd
from nltk.corpus import stopwords
import gensim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if (
            token not in gensim.parsing.preprocessing.STOPWORDS
            and len(token) > 2
            and token not in stop_words
        ):
            result.append(token)

    return result


df_fake = pd.read_csv("input/Fake.csv")
df_true = pd.read_csv("input/True.csv")

df_true["target"] = 1
df_fake["target"] = 0
df = pd.concat([df_true, df_fake]).reset_index(drop=True)
df["original"] = df["title"] + " " + df["text"]

stop_words = stopwords.words("english")
stop_words.extend(["from", "subject", "re", "edu", "use"])

df.subject = df.subject.replace(
    {"politics": "PoliticsNews", "politicsNews": "PoliticsNews"}
)

df["clean_final"] = df["original"].apply(preprocess)
df["clean_joined_final"] = df["clean_final"].apply(lambda x: " ".join(x))

X_train, X_test, y_train, y_test = train_test_split(
    df.clean_joined_final, df.target, test_size=0.25, random_state=0
)
vec_train = CountVectorizer().fit(X_train)
X_vec_train = vec_train.transform(X_train)
X_vec_test = vec_train.transform(X_test)
model = LogisticRegression(C=3)
model.fit(X_vec_train, y_train)
predicted_value = model.predict(X_vec_test)
accuracy_value = roc_auc_score(y_test, predicted_value)

filename = "fake_news_model.pkl"
joblib.dump(model, filename)
