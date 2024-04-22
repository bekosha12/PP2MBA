import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, cross_val_score
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

# Применение oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Применение undersampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)

# Разделение данных на обучающий и тестовый наборы (70% - обучающая модель, 30% - тестовая модель)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Используем автоматическую биннингизацию и расчет WOE
bins = sc.woebin(train_data, y='target', x=X_resampled.columns.tolist(), stop_limit=0.1)
train_woe = sc.woebin_ply(train_data, bins)
test_woe = sc.woebin_ply(test_data, bins)

# Создание базовых моделей LightGBM и XGBoost с регуляризацией
model_lgb = lgb.LGBMClassifier(reg_alpha=0.1, reg_lambda=0.1)
model_xgb = xgb.XGBClassifier(reg_alpha=0.1, reg_lambda=0.1)

# Поиск оптимальных гиперпараметров для LightGBM
param_grid_lgb = {'num_leaves': [15, 31, 50], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2]}
grid_search_lgb = GridSearchCV(model_lgb, param_grid=param_grid_lgb, cv=5, scoring='accuracy')
grid_search_lgb.fit(X_train, y_train)
best_params_lgb = grid_search_lgb.best_params_
model_lgb.set_params(**best_params_lgb)

# Поиск оптимальных гиперпараметров для XGBoost
param_grid_xgb = {'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [50, 100, 200]}
grid_search_xgb = GridSearchCV(model_xgb, param_grid=param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)
best_params_xgb = grid_search_xgb.best_params_
model_xgb.set_params(**best_params_xgb)

# Обучение моделей LightGBM и XGBoost
model_lgb.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)

# Получение прогнозов базовых моделей на тестовом наборе данных
predictions_lgb_test = model_lgb.predict_proba(X_test)[:, 1]
predictions_xgb_test = model_xgb.predict_proba(X_test)[:, 1]

# Формирование признаков для мета-модели на тестовом наборе данных
meta_features_test = np.column_stack((predictions_lgb_test, predictions_xgb_test))

# Создание мета-модели (случайный лес)
meta_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
meta_model.fit(meta_features_test, y_test)  

# Получение прогнозов мета-модели
meta_predictions_test = meta_model.predict_proba(meta_features_test)[:, 1]

# Вычисление метрик для мета-модели
accuracy_meta = accuracy_score(y_test, meta_model.predict(meta_features_test))
auc_roc_meta = roc_auc_score(y_test, meta_predictions_test)
precision_meta, recall_meta, _ = precision_recall_curve(y_test, meta_predictions_test)
auc_pr_meta = auc(recall_meta, precision_meta)

# Вывод метрик для мета-модели
print("\nMetrics for Meta Model:")
print("Accuracy:", accuracy_meta)
print("AUC-ROC:", auc_roc_meta)
print("AUC-PR:", auc_pr_meta)

# Кросс-валидация для оценки AUC-PR и AUC-ROC для LightGBM и XGBoost
predictions_lgb_cv = cross_val_predict(model_lgb, X_train, y_train, cv=5, method='predict_proba')[:, 1]
precision_lgb_cv, recall_lgb_cv, _ = precision_recall_curve(y_train, predictions_lgb_cv)
auc_pr_lgb_cv = auc(recall_lgb_cv, precision_lgb_cv)
auc_roc_lgb_cv = roc_auc_score(y_train, predictions_lgb_cv)

predictions_xgb_cv = cross_val_predict(model_xgb, X_train, y_train, cv=5, method='predict_proba')[:, 1]
precision_xgb_cv, recall_xgb_cv, _ = precision_recall_curve(y_train, predictions_xgb_cv)
auc_pr_xgb_cv = auc(recall_xgb_cv, precision_xgb_cv)
auc_roc_xgb_cv = roc_auc_score(y_train, predictions_xgb_cv)

print("\nAUC-PR для LightGBM (CV):", auc_pr_lgb_cv)
print("AUC-ROC для LightGBM (CV):", auc_roc_lgb_cv)
print("\nAUC-PR для XGBoost (CV):", auc_pr_xgb_cv)
print("AUC-ROC для XGBoost (CV):", auc_roc_xgb_cv)

# Построение графиков Precision-Recall Curve для LightGBM, XGBoost и Meta Model
plt.figure(figsize=(8, 6))
plt.plot(recall_lgb_cv, precision_lgb_cv, label=f'LightGBM (AUC-PR = {auc_pr_lgb_cv:.2f})', color='blue')
plt.plot(recall_xgb_cv, precision_xgb_cv, label=f'XGBoost (AUC-PR = {auc_pr_xgb_cv:.2f})', color='red')
plt.plot(recall_meta, precision_meta, label=f'Meta Model (AUC-PR = {auc_pr_meta:.2f})', color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков ROC Curve для LightGBM, XGBoost и Meta Model
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, predictions_lgb_test)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions_xgb_test)
fpr_meta, tpr_meta, _ = roc_curve(y_test, meta_predictions_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC-ROC = {auc_roc_lgb_cv:.2f})', color='blue')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC-ROC = {auc_roc_xgb_cv:.2f})', color='red')
plt.plot(fpr_meta, tpr_meta, label=f'Meta Model (AUC-ROC = {auc_roc_meta:.2f})', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Проверка баланса классов для обучающей модели
print("\nBalance of classes for the training model:")
print(pd.Series(y_train).value_counts(normalize=True))

# Проверка баланса классов для тестовой модели
print("\nBalance of classes for the testing model:")
print(pd.Series(y_test).value_counts(normalize=True))

# Сохранение pd значений по XGBoost в отдельный csv
xgb_pd_values = pd.DataFrame({'customer_id': test_data['customer_id'], 'pd_value': predictions_xgb_test})
xgb_pd_values.to_csv('xgboost_pd_values.csv', index=False)

# Сохранение pd значений по LightGBM в отдельный csv
lgb_pd_values = pd.DataFrame({'customer_id': test_data['customer_id'], 'pd_value': predictions_lgb_test})
lgb_pd_values.to_csv('lightgbm_pd_values.csv', index=False)

# Вывод Accuracy для LightGBM, XGBoost и Meta Model на тестовом наборе данных
accuracy_lgb = accuracy_score(y_test, model_lgb.predict(X_test))
accuracy_xgb = accuracy_score(y_test, model_xgb.predict(X_test))

print("\nAccuracy for LightGBM:", accuracy_lgb)
print("Accuracy for XGBoost:", accuracy_xgb)
print("Accuracy for Meta Model:", accuracy_meta)