# Урок 9. Агрегации и сегментация данных

## 1. Цель занятия

Освоить методы агрегации и сегментации данных и научиться:

* агрегировать данные по группам;
* рассчитывать метрики;
* строить сводные таблицы;
* разбивать объекты на сегменты;
* интерпретировать результаты.

## 2. Зачем нужна агрегация

Зачастую требуется работать не с отдельными строками, а перейти к обобщённым показателям. 

Пример данных:

```python
import pandas as pd

df = pd.DataFrame({
    "client_id": [1, 2, 1, 3, 2, 1],
    "region": ["Москва", "СПб", "Москва", "Казань", "СПб", "Москва"],
    "product": ["A", "B", "B", "A", "A", "C"],
    "revenue": [100, 200, 150, 300, 250, 400]
})
```

Агрегация:

```python
df.groupby("client_id")["revenue"].sum()
```

## 3. Группировка данных (groupby)

groupby – основной инструмент агрегации в pandas.

Общий шаблон:

```python
df.groupby("признак")["колонка"].агрегация()
```

### 3.1 Группировка по одному признаку

```python
df.groupby("client_id")["revenue"].sum()
```

Позволяет получить метрику по каждому объекту.

### 3.2 По нескольким признакам

```python
df.groupby(["region", "product"])["revenue"].sum()
```

Позволяет анализировать структуру данных (например, внутри региона по товарам).

### 3.3 Сброс индекса

```python
result = df.groupby("client_id")["revenue"].sum().reset_index()
```

Нужен, чтобы получить обычную таблицу.

### 3.4 Сортировка

```python
df.groupby("client_id")["revenue"].sum().sort_values(ascending=False)
```

Используется для ранжирования (например, топ-клиенты).

### 3.5 Фильтрация

```python
df[df["region"] == "Москва"].groupby("product")["revenue"].sum()
```

Сначала фильтр, потом группировка.

### 3.6 Подсчёт количества

```python
df.groupby("client_id")["revenue"].count()
df.groupby("client_id").size()
```

* count() – без NaN
* size() – все строки

## 4. Агрегатные функции

После группировки применяются функции, которые считают метрики.

Основные:

```python
sum()
mean()
median()
min()
max()
count()
nunique()
std()
```

### 4.1 Одна функция

```python
df.groupby("client_id")["revenue"].mean()
```

Показывает среднее значение внутри группы.

### 4.2 Несколько функций

```python
df.groupby("client_id")["revenue"].agg(["sum", "mean", "max"])
```

Позволяет получить несколько метрик сразу.

### 4.3 По нескольким колонкам

```python
df.groupby("client_id").agg({
    "revenue": ["sum", "mean"],
    "product": "nunique"
})
```

Можно считать разные метрики по разным признакам.

### 4.4 Переименование

```python
df.groupby("client_id").agg(
    total_revenue=("revenue", "sum"),
    avg_revenue=("revenue", "mean"),
    unique_products=("product", "nunique")
)
```

Делает результат читаемым.

### 4.5 Кастомные функции

```python
df.groupby("client_id")["revenue"].agg(lambda x: x.max() - x.min())
```

Позволяет считать собственные метрики.

## 5. Многоуровневая агрегация

Группировка по нескольким признакам.

```python
df.groupby(["region", "product"])["revenue"].sum()
```

Создаёт иерархическую структуру (MultiIndex).

### 5.1 Сброс структуры

```python
df.groupby(["region", "product"])["revenue"].sum().reset_index()
```

Преобразует в обычную таблицу.

### 5.2 Разворот (unstack)

```python
df.groupby(["region", "product"])["revenue"].sum().unstack()
```

Строки – регионы, колонки – товары.

Используется, когда нужно анализировать данные в нескольких измерениях.

## 6. Pivot-таблицы

Pivot – это расширение groupby с удобным представлением.

```python
pd.pivot_table(
    df,
    index="region",
    columns="product",
    values="revenue",
    aggfunc="sum"
)
```

Позволяет сразу получить таблицу вида «как в Excel».

### 6.1 Заполнение пропусков

```python
pd.pivot_table(
    df,
    index="region",
    columns="product",
    values="revenue",
    aggfunc="sum",
    fill_value=0
)
```

### 6.2 Итоги

```python
pd.pivot_table(
    df,
    index="region",
    columns="product",
    values="revenue",
    aggfunc="sum",
    margins=True
)
```

Добавляет общие итоги.

## 7. Сегментация

Сегментация – это разбиение объектов на группы по метрикам.

Сначала агрегируем:

```python
agg = df.groupby("client_id").agg(
    total_revenue=("revenue", "sum"),
    transactions=("revenue", "count")
)
```

### 7.1 Простая сегментация

```python
agg["segment"] = agg["total_revenue"].apply(
    lambda x: "high" if x >= 400 else "low"
)
```

### 7.2 По нескольким условиям

```python
def segment(row):
    if row["total_revenue"] >= 400 and row["transactions"] >= 3:
        return "VIP"
    elif row["total_revenue"] >= 300:
        return "middle"
    else:
        return "low"

agg["segment"] = agg.apply(segment, axis=1)
```

### 7.3 Квантиль

```python
agg["segment"] = pd.qcut(
    agg["total_revenue"],
    q=3,
    labels=["low", "medium", "high"]
)
```

Делит на группы с равным количеством объектов.

## 8. ABC-анализ

Метод сегментации по вкладу в результат.

```python
df = df.sort_values(by="revenue", ascending=False)

df["share"] = df["revenue"] / df["revenue"].sum()
df["cum_share"] = df["share"].cumsum()

def abc_class(x):
    if x <= 0.8:
        return "A"
    elif x <= 0.95:
        return "B"
    else:
        return "C"

df["ABC"] = df["cum_share"].apply(abc_class)
```

* A – основной вклад
* B – средний
* C – малый

## 9. RFM-анализ

Сегментация клиентов по поведению:

* R – давность покупки
* F – частота
* M – сумма

### 9.1 Расчёт

```python
current_date = df["date"].max()

rfm = df.groupby("client_id").agg(
    recency=("date", lambda x: (current_date - x.max()).days),
    frequency=("date", "count"),
    monetary=("revenue", "sum")
)
```

### 9.2 Скоры

```python
rfm["R"] = pd.qcut(rfm["recency"], q=3, labels=[3,2,1])
rfm["F"] = pd.qcut(rfm["frequency"], q=3, labels=[1,2,3])
rfm["M"] = pd.qcut(rfm["monetary"], q=3, labels=[1,2,3])
```

### 9.3 Сегмент

```python
rfm["RFM"] = (
    rfm["R"].astype(str) +
    rfm["F"].astype(str) +
    rfm["M"].astype(str)
)
```

## 10. Биннинг

Перевод чисел в категории.

### 10.1 pd.cut

```python
df["segment"] = pd.cut(
    df["revenue"],
    bins=[0, 200, 500, 1000],
    labels=["low", "medium", "high"]
)
```

### 10.2 pd.qcut

```python
df["segment"] = pd.qcut(
    df["revenue"],
    q=3,
    labels=["low", "medium", "high"]
)
```

* cut – интервалы
* qcut – равное количество объектов

## 11. Интерпретация сегментов

После сегментации нужно сравнить группы.

```python
df.groupby("segment").agg(
    clients=("client_id", "count"),
    avg_revenue=("revenue", "mean")
)
```

Позволяет понять различия между сегментами и сделать выводы.
