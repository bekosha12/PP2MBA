import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import scorecardpy as sc
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Загрузка данных
train_data = pd.read_csv('C:/Users/Admin/Desktop/Case/train.csv')
test_data = pd.read_csv('C:/Users/Admin/Desktop/Case/test.csv')

# Разделение данных на признаки и целевую переменную
X = train_data.drop(['target'], axis=1)
y = train_data['target']

# Аугментация данных с использованием RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Используем автоматическую биннингизацию и расчет WOE для обучающего набора
bins_train = sc.woebin(train_data, y='target', x=X_resampled.columns.tolist(), stop_limit=0.1)
train_woe = sc.woebin_ply(train_data, bins_train)
test_woe = sc.woebin_ply(test_data, bins_train)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Аугментация данных с использованием RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled_under, y_resampled_under = undersampler.fit_resample(X_train, y_train)

# Аугментация данных с использованием RandomOverSampler
X_resampled_over, y_resampled_over = oversampler.fit_resample(X_train, y_train)

# Создание базовых моделей LightGBM и XGBoost с регуляризацией (L1, L2)
model_lgb = lgb.LGBMClassifier(reg_alpha=0.1, reg_lambda=0.1)
model_xgb = xgb.XGBClassifier(reg_alpha=0.1, reg_lambda=0.1)

# Поиск оптимальных гиперпараметров для LightGBM с использованием RandomUnderSampler
param_grid_lgb = {'num_leaves': [15, 31, 50], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2]}
grid_search_lgb = GridSearchCV(model_lgb, param_grid=param_grid_lgb, cv=5, scoring='accuracy')
grid_search_lgb.fit(X_resampled_under, y_resampled_under)
best_params_lgb = grid_search_lgb.best_params_
model_lgb.set_params(**best_params_lgb)

# Поиск оптимальных гиперпараметров для XGBoost с использованием RandomOverSampler
param_grid_xgb = {'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [50, 100, 200]}
grid_search_xgb = GridSearchCV(model_xgb, param_grid=param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_resampled_over, y_resampled_over)
best_params_xgb = grid_search_xgb.best_params_
model_xgb.set_params(**best_params_xgb)

# Обучение моделей LightGBM и XGBoost на всем обучающем наборе данных
model_lgb.fit(X_resampled, y_resampled)
model_xgb.fit(X_resampled, y_resampled)

# Получение прогнозов базовых моделей на тестовом наборе данных
predictions_lgb_test = model_lgb.predict_proba(X_test)[:, 1]
predictions_xgb_test = model_xgb.predict_proba(X_test)[:, 1]

# Формирование признаков для мета-модели на тестовом наборе данных
meta_features_test = np.column_stack((predictions_lgb_test, predictions_xgb_test))

# Создание мета-модели (случайный лес)
meta_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Обучение мета-модели на тестовом наборе данных
meta_model.fit(meta_features_test, y_test)  # Вероятно здесь подходит y_test, так как это наш набор для обучения мета-модели

# Вывод AUC-PR для LightGBM, XGBoost и Meta Model
precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, predictions_lgb_test)
auc_pr_lgb = auc(recall_lgb, precision_lgb)
print("AUC-PR для LightGBM:", auc_pr_lgb)

precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, predictions_xgb_test)
auc_pr_xgb = auc(recall_xgb, precision_xgb)
print("AUC-PR для XGBoost:", auc_pr_xgb)

predictions_meta_test = meta_model.predict_proba(meta_features_test)[:, 1]
precision_meta, recall_meta, _ = precision_recall_curve(y_test, predictions_meta_test)
auc_pr_meta = auc(recall_meta, precision_meta)
print("AUC-PR для Meta Model:", auc_pr_meta)

# Вывод AUC-ROC для LightGBM, XGBoost и Meta Model
auc_roc_lgb = roc_auc_score(y_test, predictions_lgb_test)
print("AUC-ROC для LightGBM:", auc_roc_lgb)

auc_roc_xgb = roc_auc_score(y_test, predictions_xgb_test)
print("AUC-ROC для XGBoost:", auc_roc_xgb)

auc_roc_meta = roc_auc_score(y_test, predictions_meta_test)
print("AUC-ROC для Meta Model:", auc_roc_meta)

# Вывод графика AUC-PR для LightGBM, XGBoost и Meta Model
plt.figure(figsize=(10, 8))
plt.plot(recall_lgb, precision_lgb, label=f'LightGBM (AUC-PR = {auc_pr_lgb:.2f})', color='blue')
plt.plot(recall_xgb, precision_xgb, label=f'XGBoost (AUC-PR = {auc_pr_xgb:.2f})', color='red')
plt.plot(recall_meta, precision_meta, label=f'Meta Model (AUC-PR = {auc_pr_meta:.2f})', color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Вывод графика ROC Curve для LightGBM, XGBoost и Meta Model
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, predictions_lgb_test)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions_xgb_test)
fpr_meta, tpr_meta, _ = roc_curve(y_test, predictions_meta_test)

plt.figure(figsize=(10, 8))
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC-ROC = {auc_roc_lgb:.2f})', color='blue')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC-ROC = {auc_roc_xgb:.2f})', color='red')
plt.plot(fpr_meta, tpr_meta, label=f'Meta Model (AUC-ROC = {auc_roc_meta:.2f})', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Проверка баланса классов для обучающей модели
print("\nБаланс классов для обучающей модели:")
print(pd.Series(y_resampled).value_counts(normalize=True))

# Проверка баланса классов для тестовой модели
print("\nБаланс классов для тестовой модели:")
print(pd.Series(y_test).value_counts(normalize=True))
uuuuu=test_data.drop(['customer_id'],axis=1)
rtrt=model_lgb.predict_proba(uuuuu)
popo=pd.DataFrame({'customer_id': test_data['customer_id']})
popo["PD"]=rtrt[:,1]
popo.to_csv('uspeh2.csv')
# Вывод Accuracy для LightGBM, XGBoost и Meta Model
accuracy_lgb = accuracy_score(y_test, model_lgb.predict(X_test))
accuracy_xgb = accuracy_score(y_test, model_xgb.predict(X_test))
accuracy_meta = accuracy_score(y_test, meta_model.predict(meta_features_test))

print("\nAccuracy для LightGBM:", accuracy_lgb)
print("Accuracy для XGBoost:", accuracy_xgb)
print("Accuracy для Meta Model:", accuracy_meta)

# Получение значений WOE для всех признаков
woe_values = pd.DataFrame(bins_train)

# Создание списка для хранения уникальных значений WOE
unique_woe_values = []

# Проход по каждому столбцу и получение уникальных значений WOE
for column in woe_values.columns:
    unique_woe_values.extend(woe_values[column].unique())

# Отсортировать уникальные значения WOE в порядке убывания
unique_woe_values.sort(reverse=True)

# Вывод значений WOE по убыванию
print("Значения WOE по убыванию:")
for woe in unique_woe_values:
    print(woe)