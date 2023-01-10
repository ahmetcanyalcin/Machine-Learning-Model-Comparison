################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("D:\VBO Veri Bilimi Okulu Bootcamp Eğitimi\Veri Bilimi için Pyhton Dersi çalışmaları\Veri Setleri\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

df.head()


################################################
# Random Forests
################################################

#İlerleyen sürede fonksiyonların "otomatik" ön tanımlı değerlenirini kullanmak istemeyebiliriz. Yani kullanıcı olarak parametreleri kendi bilgi dahilimizde ayarlamak istiyoruz.Bu işleme hiper parametre optimizasyon denir.
#Bu sebeple istediğimiz tüm kombinasyonları bir grid'in içine yazıp işliyoruz


rf_model = RandomForestClassifier(random_state=17)

# Random forest'ın özelliklerini aşağıda belirliyoruz.

rf_params = {"max_depth": [5, 8, None], #derinlik ne kadar dallanma olacağını ifade eder.
             "max_features": [3, 5, 7, "auto"], #bölünmelerde göz önünde bulundurulması gereken değişken sayılarını ifade eder
             "min_samples_split": [2, 5, 8, 15, 20], #Daha fazla bölünme olmaması için tanımlanır. Yani dallandıktan sonra 2 tane kalırsa daha fazla bölünme yapmayacaktır. veya 5 tane kaldıktan sonra gibi.
             "n_estimators": [100, 200, 500]} #Ağaç sayısını ifade eder. Örneğin 100 tane rastgele ağaç oluşturur.

# cv = cross-validation parametresidir  n_jobs = işlemci kullanımı içindir ki -1 pc'nin işlemcisini %100'e çıkarır. yani aşağıda bulunan rf_best_grid nesnesi modelin fitlenmesi için kullanılıyor. Ar

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

#rf_best_grid nesnesi fit edilen modeldir aslında. İçerisinde en iyi paremetrelere karşılık bir model nesnesini tutar. Tahmin yapılmak istenirse bu kullanılabilir.

rf_best_grid.best_params_

rf_best_grid.best_score_

#en iyi model değeri 0.777404295051354'dir


rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

#rf_best_grid sonucuna göre fonksiyonda max derinliği ön tanımlı parametre olan none'ı kullanmış , 5 değişkenle devam etmiş, Daha sonra 8 tane kaldıktan sonra bölünmeyi bırakmış ve son olarak 500 tane rastgele ağaç kullanmıştır.

#'max_depth': None,
#'max_features': 5,
#'min_samples_split': 8,
#'n_estimators': 500}


#Final modelimizin hatasını, test setiniin accuracy değerini ve f1 değerini aşağıda işliyoruz.
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()          # 0.766848940533151
cv_results['test_f1'].mean()                # 0.6447777811143756
cv_results['test_roc_auc'].mean()           # 0.8271054131054132

# Model Performans Result
# 0.8271054131054132

################################################
# GBM Model
################################################

#GBM için de aynı işlemleri yapıyoruz.

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_params = {"learning_rate": [0.01, 0.1],  #y-yhat 'den gelir. Her ağacın katkısını learning rate kadar küçültür.
              "max_depth": [3, 8],           #maximum derinlik
              "n_estimators": [500, 1000],   #random forest'da ağaç sayısını ifade ederken GBM'de iterasyon sayısıdır.
              "subsample": [1, 0.5, 0.7]}    #

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

#{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.7}

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()      #0.7474367737525631
cv_results['test_f1'].mean()            # 0.6192760703700929
cv_results['test_roc_auc'].mean()       #0.8120113960113959

# Result
# 0.8298774928774929

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()      # 0.7474367737525631
cv_results['test_f1'].mean()            #0.6192760703700929
cv_results['test_roc_auc'].mean()       #0.8120113960113959

#Result of xgboost_best_grid.best_params_
# {'colsample_bytree': 0.5,
#  'learning_rate': 0.001,
#  'max_depth': 5,
#  'n_estimators': 500}

# Result
#0.8297977207977209



################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

#Result
#{'colsample_bytree': 1, 'learning_rate': 0.01, 'n_estimators': 300}

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()      #0.7474367737525631
cv_results['test_f1'].mean()            #0.6192760703700929
cv_results['test_roc_auc'].mean()      #0.8120113960113959

# 0.821051282051282

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  #0.7474367737525631
cv_results['test_f1'].mean()        # 0.6192760703700929
cv_results['test_roc_auc'].mean()   #0.8120113960113959

#Result
#0.8120113960113959


################################################
# Feature Importance
################################################

# Ağaç yöntemlerin hemen hemen hepsinde ortak olan bir özelliktir. Bu özelliği kullanarak kendi fonksiyonumuzu yazarak veleri göreselleştirebiliriz.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)


################################
# Analyzing Model Complexity with Learning Curves
################################


################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()



rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],  # ağaç sayısını ifade eder
                 ["max_features", [3, 5, 7, "auto"]],      #
                 ["min_samples_split", [2, 5, 8, 15, 20]], #
                 ["n_estimators", [10, 50, 100, 200, 500]]] #

rf_model = RandomForestClassifier(random_state=17)


for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])





