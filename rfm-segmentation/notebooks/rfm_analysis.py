"""
RFM-сегментация клиентской базы — Online Retail II (UCI/Kaggle)
Полный пайплайн: очистка -> расчёт RFM -> сегментация -> визуализация -> выводы.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------------
# 1. Загрузка данных и первичная диагностика
# ------------------------------------------------------------------
df = pd.read_csv(r"E:\Пет-проекты\RFM\rfm-segmentation\data\online_retail_II.csv", encoding="ISO-8859-1")

print("=" * 60)
print("ШАГ 1: ПЕРВИЧНАЯ ДИАГНОСТИКА")
print("=" * 60)

print("\nФорма датасета:", df.shape)

print("\nТипы данных и заполненность столбцов:")
df.info()

print("\nПропуски по столбцам:")
print(df.isnull().sum())


print("\nСтатистика по числовым столбцам (.describe()):")
print(df[["Quantity", "Price"]].describe())
# Здесь видно min у Quantity и Price в минусе

print("\nСколько строк с некорректными значениями:")
print("  Quantity <= 0:", (df["Quantity"] <= 0).sum())
print("  Price <= 0:", (df["Price"] <= 0).sum())

print("\nПримеры строк с отрицательным Quantity (первые 5):")
print(df[df["Quantity"] < 0][["Invoice", "StockCode", "Description", "Quantity"]].head())
# Гипотеза: это возвраты. Проверяем — у таких строк Invoice начинается на 'C' (cancellation)?

neg_qty = df[df["Quantity"] < 0]
pct_starts_with_c = neg_qty["Invoice"].astype(str).str.startswith("C").mean()
print(f"\nДоля строк с отрицательным Quantity, где Invoice начинается на 'C': {pct_starts_with_c:.1%}")
# Если доля близка к 100% — гипотеза подтверждена: отрицательное количество = отменённый заказ,
# и удаление по префиксу 'C' закроет почти все эти случаи автоматически

print("\nПримеры строк с отрицательной Price (какие это товары):")
print(df[df["Price"] < 0]["StockCode"].value_counts().head())
# Проверяем: это реальные товарные позиции или служебные коды
# (в этом датасете часто встречаются коды вроде 'B' — Bank Charges, 'D' — Discount,
# 'M'/'m' — Manual — то есть не товарные строки, а технические корректировки)

print("\nПроверка дубликатов строк:")
print("Полных дублей:", df.duplicated().sum())

# ------------------------------------------------------------------
# 2. Очистка данных:
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 2: ОЧИСТКА")
print("=" * 60)

rows_before = df.shape[0]

# 2.1 — убираем строки без Customer ID: без него невозможно привязать
# транзакцию к клиенту, а RFM считается именно на уровне клиента
df = df.dropna(subset=["Customer ID"])
print(f"После удаления строк без Customer ID: {df.shape[0]} строк (было {rows_before})")

# 2.2 — убираем отменённые заказы (Invoice начинается на 'C' — cancellation),
# это подтверждено проверкой выше: такие Invoice почти всегда совпадают
# со строками отрицательного Quantity
rows_before_cancel = df.shape[0]
df = df[~df["Invoice"].astype(str).str.startswith("C")]
print(f"После удаления отменённых заказов (Invoice начинается на 'C'): "
      f"{df.shape[0]} строк (убрано {rows_before_cancel - df.shape[0]})")

# 2.3 — проверяем, что осталось из некорректных значений ПОСЛЕ удаления отмен —
# не дублируем ли мы уже сделанную работу следующим фильтром
remaining_neg_qty = (df["Quantity"] <= 0).sum()
remaining_neg_price = (df["Price"] <= 0).sum()
print(f"После удаления отмен осталось: Quantity<=0: {remaining_neg_qty}, Price<=0: {remaining_neg_price}")
# Если эти числа не нулевые — значит, есть ещё случаи, не связанные с отменами
# (например, служебные корректировочные записи), и следующий фильтр оправдан,
# а не является избыточным дублированием предыдущего шага

# 2.4 — убираем оставшиеся некорректные строки (служебные записи вроде Discount/
# Bank Charges, обнаруженные на шаге диагностики, а не отменённые заказы —
# те уже убраны на шаге 2.2)
rows_before_filter = df.shape[0]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
print(f"После удаления оставшихся некорректных строк: "
      f"{df.shape[0]} строк (убрано {rows_before_filter - df.shape[0]})")

# 2.5 — проверка на полные дубликаты строк (могли остаться после всех фильтров)
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Удалено дублирующихся строк: {duplicates}")

# Приводим дату к datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Считаем сумму по строке
df["TotalPrice"] = df["Quantity"] * df["Price"]

print(f"\nИТОГО после очистки: {df.shape[0]} строк "
      f"({100 * df.shape[0] / rows_before:.1f}% от исходных {rows_before}), "
      f"{df['Customer ID'].nunique()} уникальных клиентов")

# ------------------------------------------------------------------
# 3. Расчёт RFM-метрик на уровне клиента
# ------------------------------------------------------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalPrice", "sum"),
).reset_index()

print("\nRFM-таблица (первые строки):\n", rfm.head())
print("\nСтатистика по метрикам:\n", rfm[["Recency", "Frequency", "Monetary"]].describe())

# ------------------------------------------------------------------
# 4. Присвоение баллов по квартилям (1-4)
# ------------------------------------------------------------------
# Recency: чем МЕНЬШЕ дней с последней покупки — тем ЛУЧШЕ (выше балл)
rfm["R_score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1]).astype(int)

# Frequency и Monetary: чем БОЛЬШЕ — тем лучше (выше балл)
# rank(method="first") нужен, чтобы избежать ошибки qcut при большом числе одинаковых значений
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"], q=4, labels=[1, 2, 3, 4]).astype(int)

rfm["RFM_Score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
rfm["RFM_Sum"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

# ------------------------------------------------------------------
# 5. Присвоение названий сегментов
# ------------------------------------------------------------------
def assign_segment(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New Customers"
    elif r >= 3 and f <= 2 and m >= 3:
        return "Promising"
    elif r == 2 and f >= 3:
        return "At Risk"
    elif r <= 2 and f >= 4 and m >= 4:
        return "Cant Lose Them"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    elif r == 2 and f <= 2:
        return "Hibernating"
    else:
        return "Need Attention"


rfm["Segment"] = rfm.apply(assign_segment, axis=1)

# ------------------------------------------------------------------
# 6. Сводная таблица по сегментам
# ------------------------------------------------------------------
segment_summary = (
    rfm.groupby("Segment")
    .agg(
        Клиентов=("Customer ID", "count"),
        Средний_Recency=("Recency", "mean"),
        Средний_Frequency=("Frequency", "mean"),
        Суммарная_выручка=("Monetary", "sum"),
    )
    .round(1)
    .sort_values("Суммарная_выручка", ascending=False)
)
segment_summary["% клиентов"] = (
    100 * segment_summary["Клиентов"] / segment_summary["Клиентов"].sum()
).round(1)
segment_summary["% выручки"] = (
    100 * segment_summary["Суммарная_выручка"] / segment_summary["Суммарная_выручка"].sum()
).round(1)

print("\n=== Сводка по сегментам ===\n")
print(segment_summary)

segment_summary.to_csv("segment_summary.csv", encoding="utf-8-sig")
print("\nСохранено: segment_summary.csv")

# ------------------------------------------------------------------
# 7. Визуализация
# ------------------------------------------------------------------
sns.set_style("whitegrid")

# 7.1 — распределения R, F, M
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(rfm["Recency"], bins=40, ax=axes[0], color="steelblue")
axes[0].set_title("Распределение Recency (дней)")

sns.histplot(rfm["Frequency"], bins=40, ax=axes[1], color="seagreen")
axes[1].set_title("Распределение Frequency (заказов)")
axes[1].set_xlim(0, rfm["Frequency"].quantile(0.99))

sns.histplot(rfm["Monetary"], bins=40, ax=axes[2], color="indianred")
axes[2].set_title("Распределение Monetary (сумма)")
axes[2].set_xlim(0, rfm["Monetary"].quantile(0.99))

plt.tight_layout()
plt.savefig("images/rfm_distributions.png", dpi=150)
plt.close()
print("Сохранено: images/rfm_distributions.png")

# 7.2 — размер сегментов
plt.figure(figsize=(9, 6))
order = segment_summary.sort_values("Клиентов", ascending=True).index
sns.barplot(
    x=segment_summary.loc[order, "Клиентов"],
    y=order,
    color="steelblue",
)
plt.title("Количество клиентов по сегментам")
plt.xlabel("Клиентов")
plt.ylabel("")
plt.tight_layout()
plt.savefig("images/segment_sizes.png", dpi=150)
plt.close()
print("Сохранено: images/segment_sizes.png")

# 7.3 — вклад сегментов в выручку
plt.figure(figsize=(9, 6))
order_rev = segment_summary.sort_values("Суммарная_выручка", ascending=True).index
sns.barplot(
    x=segment_summary.loc[order_rev, "Суммарная_выручка"],
    y=order_rev,
    color="indianred",
)
plt.title("Вклад сегментов в общую выручку")
plt.xlabel("Выручка")
plt.ylabel("")
plt.tight_layout()
plt.savefig("images/segment_revenue.png", dpi=150)
plt.close()
print("Сохранено: images/segment_revenue.png")

# ------------------------------------------------------------------
# 8. Автоматическая сводка выводов 
# ------------------------------------------------------------------
top_segment_by_revenue = segment_summary.index[0]
top_segment_pct_revenue = segment_summary.iloc[0]["% выручки"]
top_segment_pct_customers = segment_summary.iloc[0]["% клиентов"]

lost_or_hibernating = segment_summary.loc[
    segment_summary.index.isin(["Lost", "Hibernating"]), "% клиентов"
].sum()

conclusions = f"""# Автоматическая сводка результатов RFM-анализа

Всего клиентов после очистки: {rfm['Customer ID'].nunique()}

## Ключевые цифры

- Сегмент с наибольшим вкладом в выручку: {top_segment_by_revenue}
  ({top_segment_pct_revenue}% выручки при {top_segment_pct_customers}% клиентов)
- Доля клиентов в статусе "потерянные/спящие" (Lost + Hibernating): {lost_or_hibernating:.1f}%

## Таблица по сегментам

{segment_summary.to_markdown()}


"""

with open("conclusions.md", "w", encoding="utf-8") as f:
    f.write(conclusions)

print(conclusions)
print("Сохранено: conclusions.md")

"""
# Проверка гипотезы: связан ли "горб" в Recency (380-420 дней) со структурой
# датасета из двух периодов (клиенты, активные только в 2009-2010)

# 1. Смотрим, в каком году была ПОСЛЕДНЯЯ покупка каждого клиента
last_purchase_year = df.groupby("Customer ID")["InvoiceDate"].max().dt.year
print("Распределение клиентов по году последней покупки:")
print(last_purchase_year.value_counts().sort_index())

# 2. Смотрим, у каких клиентов Recency попадает именно в диапазон "горба" (380-420 дней)
hump_customers = rfm[(rfm["Recency"] >= 380) & (rfm["Recency"] <= 420)]
print(f"\nКлиентов в диапазоне Recency 380-420: {len(hump_customers)}")

# 3. Проверяем год последней покупки именно для этих клиентов
hump_last_year = df[df["Customer ID"].isin(hump_customers["Customer ID"])] \
    .groupby("Customer ID")["InvoiceDate"].max().dt.year
print("\nГод последней покупки у клиентов из 'горба':")
print(hump_last_year.value_counts())

# 4. Для сравнения — распределение дат последней покупки по всей базе
plt.figure(figsize=(10, 4))
df.groupby("Customer ID")["InvoiceDate"].max().hist(bins=50)
plt.title("Распределение дат ПОСЛЕДНЕЙ покупки по всем клиентам")
plt.xlabel("Дата")
plt.ylabel("Количество клиентов")
plt.savefig("images/last_purchase_date_check.png", dpi=150)
plt.close()
"""