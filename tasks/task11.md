# Домашнее задание №11

## Модели классификации в Python

**Цель:** построить модель классификации, оценить качество предсказания классов и интерпретировать результат.

## 1. Создание ноутбука

Откройте Google Colab:
[https://colab.new](https://colab.new)

Создайте новый ноутбук.

## 2. Подготовка среды

```python id="pzcu12"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.dummy import DummyClassifier
```

## 3. Исходные данные

```python id="01m9di"
np.random.seed(42)

df = pd.DataFrame({
    "visits": np.random.randint(1, 30, 150),
    "purchases": np.random.randint(0, 10, 150),
    "days_from_last_visit": np.random.randint(1, 90, 150)
})

df["left"] = (
    ((df["visits"] < 10) & (df["purchases"] < 3)) |
    (df["days_from_last_visit"] > 60)
).astype(int)

df.head()
```

Переменные:

* `visits` – количество визитов клиента;
* `purchases` – количество покупок;
* `days_from_last_visit` – число дней с последнего визита;
* `left` – факт ухода клиента.

Целевая переменная – `left`.

Значения целевой переменной:

* `0` – клиент не ушел;
* `1` – клиент ушел.

## 4. Первичный анализ данных

Выполните проверку структуры данных.

```python id="nrd4jd"
df.info()
```

```python id="lkg162"
df.describe()
```

```python id="rz35z7"
df.isna().sum()
```

Ответьте кратко:

1. Есть ли пропуски?
2. Все ли признаки имеют числовой тип?
3. Сколько строк в датасете?

## 5. Распределение классов

Проверьте, сколько объектов относится к каждому классу.

```python id="d667nc"
df["left"].value_counts()
```

Посчитайте доли классов.

```python id="6aj9mx"
df["left"].value_counts(normalize=True)
```

Постройте график распределения классов.

```python id="6zgl6h"
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="left"
)

plt.title("Распределение классов")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.grid(True)
plt.show()
```

Ответьте кратко: классы распределены равномерно или есть дисбаланс?

## 6. Визуализация зависимости

Постройте график связи количества визитов и давности последнего визита.

```python id="6dzylx"
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="visits",
    y="days_from_last_visit",
    hue="left"
)

plt.title("Поведение клиентов и факт ухода")
plt.xlabel("Количество визитов")
plt.ylabel("Дней с последнего визита")
plt.grid(True)
plt.show()
```

Сделайте краткий вывод: отличаются ли клиенты классов `0` и `1` визуально.

## 7. Формирование признаков и целевой переменной

```python id="gm8pww"
X = df[[
    "visits",
    "purchases",
    "days_from_last_visit"
]]

y = df["left"]
```

Проверьте размеры:

```python id="btjp9s"
print(X.shape)
print(y.shape)
```

## 8. Разделение данных на train и test

```python id="6h3t4r"
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
```

Параметр `stratify=y` сохраняет соотношение классов в train и test.

## 9. Обучение модели

Используйте логистическую регрессию.

```python id="q781im"
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

## 10. Построение прогноза

```python id="rfaqqq"
y_pred = model.predict(X_test)
```

Создайте таблицу сравнения факта и прогноза.

```python id="lam4a9"
result = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
})

result.head()
```

## 11. Вероятности классов

Рассчитайте вероятность ухода клиента.

```python id="w58aw3"
y_proba = model.predict_proba(X_test)[:, 1]

result = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "probability_left": y_proba
})

result.head()
```

`probability_left` показывает оцененную моделью вероятность класса `1`.

## 12. Оценка качества модели

Рассчитайте основные метрики классификации.

```python id="dl13mx"
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1-score:", round(f1, 2))
```

Кратко объясните:

* что показывает Accuracy;
* что показывает Precision;
* что показывает Recall;
* что показывает F1-score.

## 13. Матрица ошибок

```python id="br7t1p"
cm = confusion_matrix(y_test, y_pred)

cm
```

Постройте тепловую карту.

```python id="a1ck6s"
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Матрица ошибок")
plt.xlabel("Прогноз")
plt.ylabel("Факт")
plt.show()
```

Ответьте кратко:

1. Сколько объектов модель классифицировала правильно?
2. Сколько клиентов модель ошибочно отнесла к ушедшим?
3. Сколько ушедших клиентов модель не нашла?

## 14. Classification report

Выведите полный отчет по классификации.

```python id="crq82i"
print(classification_report(y_test, y_pred))
```

Сделайте краткий вывод: по какому классу качество выше.

## 15. Интерпретация коэффициентов

Выведите коэффициенты модели.

```python id="up9k0u"
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
})

coef_df
```

Отсортируйте коэффициенты по модулю.

```python id="p90e4z"
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

coef_df.sort_values(
    by="abs_coefficient",
    ascending=False
)
```

Ответьте кратко:

1. Какой признак сильнее всего влияет на прогноз?
2. Какие признаки повышают вероятность ухода?
3. Какие признаки снижают вероятность ухода?

## 16. Сравнение с базовой моделью

Базовая модель всегда предсказывает самый частый класс.

```python id="ll6ftw"
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("Accuracy базовой модели:", round(baseline_accuracy, 2))
print("Accuracy логистической регрессии:", round(accuracy, 2))
```

Сделайте вывод: дает ли логистическая регрессия более точный результат, чем базовая модель.

## 17. Прогноз для нового клиента

Сделайте прогноз для нового клиента.

```python id="xk32q9"
new_client = pd.DataFrame({
    "visits": [5],
    "purchases": [1],
    "days_from_last_visit": [50]
})

new_prediction = model.predict(new_client)
new_probability = model.predict_proba(new_client)[:, 1]

print("Прогнозный класс:", new_prediction[0])
print("Вероятность ухода:", round(new_probability[0], 2))
```

Кратко объясните полученный результат.

## 18. Итоговые вопросы

Ответьте письменно:

1. Какая переменная является целевой?
2. Какие признаки использовались для классификации?
3. Есть ли дисбаланс классов?
4. Какое значение Accuracy получилось?
5. Какое значение Recall получилось?
6. Лучше ли модель логистической регрессии, чем базовая модель?
7. Какие ограничения есть у построенной модели?

## Формат сдачи

* ссылка на Google Colab;
* ноутбук выполняется сверху вниз без ошибок;
* рассчитаны Accuracy, Precision, Recall и F1-score;
* построена матрица ошибок;
* выполнен прогноз для нового клиента;
* добавлены краткие выводы.
