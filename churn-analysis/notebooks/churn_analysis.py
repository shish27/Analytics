"""
Customer Churn Analysis — Telco Customer Churn (IBM, Kaggle)
Полный пайплайн: очистка -> EDA -> feature engineering -> модель -> выводы.
"""

import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------------
df = pd.read_csv(r"C:/Users/Bossy/Downloads/Анализ оттока клиентов/churn-analysis/data/telco_churn.csv")

# ------------------------------------------------------------------
# 2. Загрузка в SQLite для SQL-части анализа (см. sql/churn_analysis.sql)
# ------------------------------------------------------------------
conn = sqlite3.connect("churn.db")
df.to_sql("customers", conn, if_exists="replace", index=False)

# пример: выполнить один из SQL-запросов прямо из Python и получить результат
contract_churn = pd.read_sql(
    """
    SELECT Contract,
           COUNT(*) AS total,
           ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate
    FROM customers GROUP BY Contract ORDER BY churn_rate DESC
    """,
    conn,
)
print("Отток по типу контракта:\n", contract_churn, "\n")

# ------------------------------------------------------------------
# 3. Очистка данных
# ------------------------------------------------------------------
# TotalCharges хранится как строка и содержит пробелы у новых клиентов (tenure=0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df = df.drop_duplicates(subset="customerID")
df["Churn_flag"] = (df["Churn"] == "Yes").astype(int)

print(f"Всего клиентов: {len(df)}")
print(f"Доля оттока: {df['Churn_flag'].mean():.1%}\n")

# ------------------------------------------------------------------
# 4. EDA — визуализации
# ------------------------------------------------------------------
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.barplot(
    data=df.groupby("Contract")["Churn_flag"].mean().reset_index(),
    x="Contract", y="Churn_flag", ax=axes[0, 0],
)
axes[0, 0].set_title("Отток по типу контракта")
axes[0, 0].set_ylabel("Доля оттока")

sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", ax=axes[0, 1], bins=30)
axes[0, 1].set_title("Распределение tenure (месяцев с компанией)")

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0])
axes[1, 0].set_title("Ежемесячный платёж vs отток")

sns.barplot(
    data=df.groupby("TechSupport")["Churn_flag"].mean().reset_index(),
    x="TechSupport", y="Churn_flag", ax=axes[1, 1],
)
axes[1, 1].set_title("Отток в зависимости от TechSupport")
axes[1, 1].set_ylabel("Доля оттока")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.close()
print("Сохранено: eda_overview.png\n")

# ------------------------------------------------------------------
# 5. Feature engineering
# ------------------------------------------------------------------
df["tenure_segment"] = pd.cut(
    df["tenure"], bins=[-1, 12, 24, 48, 200],
    labels=["0-12", "13-24", "25-48", "48+"],
)
df["CLV_proxy"] = df["tenure"] * df["MonthlyCharges"]

# Отбрасываем ID и исходный текстовый Churn, кодируем категориальные признаки
feature_df = df.drop(columns=["customerID", "Churn", "tenure_segment"])
feature_df = pd.get_dummies(feature_df, drop_first=True)

X = feature_df.drop(columns=["Churn_flag"])
y = feature_df["Churn_flag"]

# ------------------------------------------------------------------
# 6. Train/test split + масштабирование
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 7. Модели: Logistic Regression (baseline, интерпретируемая) + Random Forest
# ------------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)

# ------------------------------------------------------------------
# 8. Оценка качества
# ------------------------------------------------------------------
for name, model, X_te in [
    ("Logistic Regression", log_reg, X_test_scaled),
    ("Random Forest", rf, X_test),
]:
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Стался", "Ушёл"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}\n")

# Матрица ошибок для Random Forest
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, display_labels=["Остался", "Ушёл"])
plt.title("Матрица ошибок — Random Forest")
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ROC-кривая
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC-кривая — Random Forest")
plt.savefig("roc_curve.png", dpi=150)
plt.close()
print("Сохранены: confusion_matrix.png, roc_curve.png\n")

# ------------------------------------------------------------------
# 9. Интерпретация: важность признаков
# ------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Топ-10 факторов, влияющих на отток:\n", importances.head(10), "\n")

plt.figure(figsize=(8, 6))
importances.head(10).sort_values().plot(kind="barh")
plt.title("Топ-10 факторов оттока (Random Forest feature importance)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Сохранено: feature_importance.png")

# ------------------------------------------------------------------
# 10. Автоматическая сводка выводов — собирает уже посчитанные выше
#     переменные в читаемый текстовый отчёт (conclusions.md).
#     Это не готовый бизнес-анализ, а "скелет" с реальными цифрами:
#     интерпретацию (почему так и что делать) всё равно добавляет
#     человек в README, но числа в тексте гарантированно совпадают
#     с тем, что реально выдал код на этом прогоне.
# ------------------------------------------------------------------
overall_churn_rate = df["Churn_flag"].mean()

# Отток по типу контракта — берём из уже посчитанного contract_churn (шаг 2)
contract_max = contract_churn.iloc[0]  # уже отсортировано по убыванию churn_rate
contract_min = contract_churn.iloc[-1]

# ROC-AUC уже вычислен на шаге 8, пересчитаем переменной для отчёта
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_roc_auc = roc_auc_score(y_test, rf_proba)

# Recall для класса "Churn" отдельно (из confusion matrix)
rf_pred = rf.predict(X_test)
tp = ((rf_pred == 1) & (y_test == 1)).sum()
fn = ((rf_pred == 0) & (y_test == 1)).sum()
fp = ((rf_pred == 1) & (y_test == 0)).sum()
recall_churn = tp / (tp + fn)
precision_churn = tp / (tp + fp)

top5_features = importances.head(5)

conclusions = f"""# Автоматическая сводка результатов

## Ключевые цифры (посчитано на этом прогоне)

- Общий отток по датасету: {overall_churn_rate:.1%}
- Наибольший отток: {contract_max['Contract']} ({contract_max['churn_rate']}%)
- Наименьший отток: {contract_min['Contract']} ({contract_min['churn_rate']}%)
- ROC-AUC модели (Random Forest): {rf_roc_auc:.2f}
- Recall по классу "Churn": {recall_churn:.1%} (модель находит {tp} из {tp+fn} реально уходящих клиентов)
- Precision по классу "Churn": {precision_churn:.1%}

## Топ-5 факторов оттока (Random Forest feature importance)

{chr(10).join(f"{i+1}. {name} — {score:.3f}" for i, (name, score) in enumerate(top5_features.items()))}


"""

with open("conclusions.md", "w", encoding="utf-8") as f:
    f.write(conclusions)

print(conclusions)
print("Сохранено: conclusions.md")
