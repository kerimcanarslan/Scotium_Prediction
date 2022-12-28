# pip install yellowbrick
# pip install verstack
# conda install lightgbm

import numpy as np
import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from verstack import LGBMTuner

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################
# Adım 1- Veri setlerini okutunuz
################################

df1 = pd.read_csv("vbo_format/datasets/scotium/scoutium_attributes.csv", sep=";")
df2 = pd.read_csv("vbo_format/datasets/scotium/scoutium_potential_labels.csv", sep=";")
df1.head()
df2.head()


################################
# Adım 2 - Veri setlerini birleştiriniz
################################

df = pd.merge(df1, df2, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'], how='left')
df.head()
df.shape

################################
# Adım 3 - position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız
################################

df = df.loc[~(df["position_id"]==1)]

################################
# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur
################################

df["potential_label"].value_counts()
df = df.loc[~(df["potential_label"] == "below_average")]

################################
# Adım 5 - Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz
################################
# her bir satırda 1 oyuncu olacak.

# Görev 1-) İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
# “attribute_value” olacak şekilde pivot table’ı oluşturunuz.

table1 = pd.pivot_table(df, values="attribute_value", index=["player_id", "position_id", "potential_label"], columns=["attribute_id"])
table1

# Görev 2-) Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

table1 = table1.reset_index()
table1.columns.name = "index"
type(str(table1[4322].name))

for col in table1.columns:
    if type(table1[col].name) != str:
        table1[col].name = str(table1[col].name)

table1.head()
type(table1[4322].name)

################################
# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz
################################

labelencoder = LabelEncoder()
table1["potential_label"] = labelencoder.fit_transform(table1["potential_label"])
table1.head()

################################
# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız
################################

table1.info()

num_cols = [col for col in table1.columns if table1[col].nunique()>7]
num_cols = [col for col in table1.columns if "player_id" not in col]
num_cols = num_cols[2:]

################################
# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız
################################

df = table1.copy()
df.head()

scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(scaled, columns=df[num_cols].columns)
df.head()


################################
# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız
################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(eval_metric='logloss'), xgboost_params)]


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss')),
                   # ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

    knn_params = {"n_neighbors": range(2, 50)}

    cart_params = {'max_depth': range(1, 20),
                   "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [8, 15, None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [15, 20],
                 "n_estimators": [200, 300]}

    xgboost_params = {"learning_rate": [0.1, 0.01],
                      "max_depth": [5, 8],
                      "n_estimators": [100, 200],
                      "colsample_bytree": [0.5, 1]}

    lightgbm_params = {"learning_rate": [0.01, 0.1],
                       "n_estimators": [300, 500],
                       "colsample_bytree": [0.7, 1]}

    classifiers = [('KNN', KNeighborsClassifier(), knn_params),
                   ("CART", DecisionTreeClassifier(), cart_params),
                   ("RF", RandomForestClassifier(), rf_params),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
                   ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)

base_models(X, y, scoring="roc_auc")
base_models(X, y, scoring="f1")
base_models(X, y, scoring="accuracy")

hyperparameter_optimization(X, y, cv=3, scoring="roc_auc")

