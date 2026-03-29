# Урок 7. Введение в Python и Google Colab

## 1. Цель занятия

Освоить основы использования Python в аналитике и научиться:
* запускать код в среде Google Colab;
* работать с базовыми типами данных;
* выполнять вычисления;
* использовать табличные структуры данных;
* загружать данные из файлов;
* проводить первичный анализ данных (EDA).

## 2. Python

Python представляет собой инструмент для автоматизации и масштабирования анализа данных. В отличие от Excel, который ориентирован на ручную работу, Python позволяет формализовать вычисления, обрабатывать большие объемы данных, воспроизводить анализ и снижать вероятность ошибок.

### Пример

```python
revenue = 100000
cost = 70000

profit = revenue - cost
margin = profit / revenue

print("Прибыль:", profit)
print("Маржинальность:", round(margin, 2))
```

При увеличении объема данных и числа операций Excel становится неэффективным. Например, в случае, если необходимо обработать множество файлов. В Excel потребуется ручное объединение, в связи с чем возрастет риск ошибок.

В Python это может быть реализовано так:

```python
import pandas as pd
import glob

files = glob.glob("data/*.csv")

df_list = []
for f in files:
    df = pd.read_csv(f)
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)

print(df_all.shape)
```

В Python данные обрабатываются как структурированные объекты.

```python
import pandas as pd

df = pd.DataFrame({
    "revenue": [100, 200, 150],
    "cost": [70, 120, 90]
})

df["profit"] = df["revenue"] - df["cost"]
df["margin"] = df["profit"] / df["revenue"]

print(df)
```

## 3. Google Colab

[Google Colab](https://colab.research.google.com/) – облачная среда для работы с Python. Она:

* не требует установки;
* работает через браузер;
* выполняет код на удаленном сервере;
* поддерживает аналитические библиотеки.

### Пример

```python
print("Hello, world!")
```

## 4. Работа с ноутбуком

### 4.1 Типы ячеек

* Code – выполнение Python-кода
* Text – пояснения

### 4.2 Порядок выполнения

Результат зависит от последовательности выполнения.

```python
b = a + 10  # ошибка, если a не определена
```

### 4.3 Состояние среды

Переменные сохраняются между ячейками:

```python
a = 5
b = a + 10
print(b)
```

### 4.4 Перезапуск среды

После перезапуска все переменные удаляются.

```python
x = 100
print(x)
```

После restart:

```
NameError
```

## 5. Базовый синтаксис Python

### 5.1 Переменные

```python
revenue = 100000
cost = 70000
profit = revenue - cost
```

### 5.2 Типы данных

```python
a = 10        # int
b = 3.5       # float
name = "A"    # str
flag = True   # bool
```

### 5.3 Списки

```python
revenues = [100, 200, 150]

print(revenues[0])
print(sum(revenues))

revenues.append(300)
```

### 5.4 Словари

```python
data = {
    "revenue": 100,
    "cost": 70
}

print(data["revenue"])
```

### 5.5 Список словарей

```python
data = [
    {"region": "Москва", "revenue": 100},
    {"region": "СПб", "revenue": 150}
]
```

## 6. Введение в pandas

pandas – библиотека для работы с табличными данными.

### 6.1 Создание DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    "region": ["Москва", "СПб"],
    "revenue": [100, 200],
    "cost": [70, 120]
})
```

### 6.2 Работа со столбцами

```python
df["profit"] = df["revenue"] - df["cost"]
df["margin"] = df["profit"] / df["revenue"]
```

### 6.3 Агрегации

```python
df["revenue"].sum()
df["revenue"].mean()
```

### 6.4 Фильтрация

```python
df[df["revenue"] > 150]
```

### 6.5 Сортировка

```python
df.sort_values(by="revenue", ascending=False)
```

## 7. Загрузка данных

### 7.1 CSV

```python
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", sep=";")
```

### 7.2 Excel

```python
df = pd.read_excel("data.xlsx")
```

### 7.3 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/data.csv")
```

### 7.4 Проверка данных

```python
print(df.shape)
print(df.columns)
print(df.dtypes)
```

## 8. Разведочный анализ данных (EDA)

EDA – этап изучения данных без их изменения.

### 8.1 Просмотр данных

```python
df.head()
df.head(10)
```

### 8.2 Информация о данных

```python
df.info()
```

### 8.3 Описательная статистика

```python
df.describe()
```

### 8.4 Выбор колонок

```python
df["revenue"]
df[["revenue", "cost"]]
```

### 8.5 Фильтрация

```python
df[df["revenue"] > 100]

df[(df["revenue"] > 100) & (df["cost"] < 80)]
```

### 8.6 Категориальный анализ

```python
df["region"].unique()
df["region"].value_counts()
```
