# Домашнее задание №10

## Регрессионные модели в Python

**Цель:** построить регрессионную модель, оценить качество прогноза и интерпретировать результаты.

## 1. Создание ноутбука

Откройте Google Colab:
[https://colab.new](https://colab.new)

Создайте новый ноутбук.

## 2. Подготовка среды

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## 3. Исходные данные

```python
np.random.seed(42)

df = pd.DataFrame({
    "advertising_budget": np.random.randint(10, 100, 120),
    "visitors": np.random.randint(500, 3000, 120),
    "discount": np.random.randint(0, 30, 120)
})

df["revenue"] = (
    2000
    + df["advertising_budget"] * 80
    + df["visitors"] * 5
    - df["discount"] * 30
    + np.random.normal(0, 1200, 120)
)

df.head()
```

Переменные:

* `advertising_budget` – рекламный бюджет;
* `visitors` – количество посетителей;
* `discount` – размер скидки;
* `revenue` – выручка.

Целевая переменная – `revenue`.

## 4. Первичный анализ данных

Выполните проверку структуры данных.

```python
df.info()
```

```python
df.describe()
```

```python
df.isna().sum()
```

Ответьте кратко:

1. Есть ли пропуски?
2. Все ли признаки имеют числовой тип?
3. Сколько строк в датасете?

## 5. Анализ взаимосвязей

Рассчитайте корреляции.

```python
corr = df.corr(numeric_only=True)

corr
```

Постройте тепловую карту.

```python
plt.figure(figsize=(7, 5))

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues"
)

plt.title("Корреляция между переменными")
plt.show()
```

Ответьте кратко: какой признак сильнее всего связан с выручкой?

## 6. Визуализация зависимости

Постройте связь рекламного бюджета и выручки.

```python
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="advertising_budget",
    y="revenue"
)

plt.title("Связь рекламного бюджета и выручки")
plt.xlabel("Рекламный бюджет")
plt.ylabel("Выручка")
plt.grid(True)
plt.show()
```

Сделайте краткий вывод о характере зависимости.

## 7. Формирование признаков и целевой переменной

```python
X = df[[
    "advertising_budget",
    "visitors",
    "discount"
]]

y = df["revenue"]
```

Проверьте размеры:

```python
print(X.shape)
print(y.shape)
```

## 8. Разделение данных на train и test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
```

## 9. Обучение модели

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

## 10. Построение прогноза

```python
y_pred = model.predict(X_test)
```

Создайте таблицу сравнения факта и прогноза.

```python
result = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
})

result.head()
```

## 11. Оценка качества модели

```python
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 2))
```

Кратко объясните:

* что показывает MAE;
* что показывает RMSE;
* что показывает R².

## 12. Визуализация факта и прогноза

```python
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Фактическая выручка")
plt.ylabel("Прогнозная выручка")
plt.title("Факт и прогноз")
plt.grid(True)
plt.show()
```

Сделайте вывод: насколько близко прогнозные значения расположены к фактическим.

## 13. Интерпретация коэффициентов

```python
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})

coef_df
```

Отсортируйте коэффициенты по модулю.

```python
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

coef_df.sort_values(
    by="abs_coefficient",
    ascending=False
)
```

Ответьте кратко:

1. Какой признак имеет наибольший коэффициент по модулю?
2. Какие признаки увеличивают прогноз выручки?
3. Какие признаки уменьшают прогноз выручки?

## 14. Сравнение с базовой моделью

Базовая модель прогнозирует среднее значение выручки из обучающей выборки.

```python
baseline_pred = np.full(
    shape=len(y_test),
    fill_value=y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

print("MAE базовой модели:", round(baseline_mae, 2))
print("MAE линейной регрессии:", round(mae, 2))
```

Сделайте вывод: дает ли линейная регрессия более точный прогноз, чем базовая модель.

## 15. Прогноз для нового наблюдения

```python
new_data = pd.DataFrame({
    "advertising_budget": [75],
    "visitors": [2200],
    "discount": [15]
})

new_prediction = model.predict(new_data)

print("Прогноз выручки:", round(new_prediction[0], 2))
```

Кратко объясните полученный прогноз.

## 16. Итоговые вопросы

Ответьте письменно:

1. Какая переменная является целевой?
2. Какие признаки использовались для прогноза?
3. Какая средняя ошибка модели по MAE?
4. Лучше ли модель линейной регрессии, чем базовая модель?
5. Какой признак сильнее всего влияет на прогноз?
6. Какие ограничения есть у построенной модели?

## Формат сдачи

* ссылка на Google Colab;
* ноутбук выполняется сверху вниз без ошибок;
* рассчитаны MAE, RMSE и R²;
* построен график факта и прогноза;
* выполнен прогноз для нового наблюдения;
* добавлены краткие выводы.
