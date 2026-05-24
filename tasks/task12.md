# Домашнее задание №12

## Временные ряды и SARIMA в Python

**Цель:** построить прогноз временного ряда, оценить качество прогноза и сравнить простую модель с SARIMA.

## 1. Создание ноутбука

Откройте Google Colab:
[https://colab.new](https://colab.new)

Создайте новый ноутбук.

## 2. Подготовка среды

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
```

## 3. Исходные данные

```python
np.random.seed(42)

dates = pd.date_range(
    start="2021-01-01",
    periods=48,
    freq="MS"
)

trend = np.arange(48) * 1500

seasonality = 8000 * np.sin(
    2 * np.pi * np.arange(48) / 12
)

noise = np.random.normal(0, 3000, 48)

revenue = 100000 + trend + seasonality + noise

df = pd.DataFrame({
    "date": dates,
    "revenue": revenue
})

df.head()
```

Переменные:

* `date` – дата наблюдения;
* `revenue` – выручка за месяц.

Целевая переменная – `revenue`.

## 4. Первичный анализ данных

Проверьте структуру данных.

```python
df.info()
```

```python
df.describe()
```

```python
df.isna().sum()
```

Проверьте временной диапазон.

```python
print("Начало периода:", df["date"].min())
print("Конец периода:", df["date"].max())
```

Отсортируйте данные по дате.

```python
df = df.sort_values("date").reset_index(drop=True)
```

Ответьте кратко:

1. Сколько наблюдений в датасете?
2. Есть ли пропуски?
3. Какая периодичность у данных?

## 5. Визуализация временного ряда

```python
plt.figure(figsize=(10, 5))

plt.plot(
    df["date"],
    df["revenue"],
    marker="o"
)

plt.title("Динамика выручки по месяцам")
plt.xlabel("Дата")
plt.ylabel("Выручка")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Ответьте кратко:

1. Есть ли общий тренд?
2. Есть ли сезонные колебания?
3. Есть ли резкие отклонения?

## 6. Скользящее среднее

Рассчитайте скользящее среднее за 3 месяца.

```python
df["rolling_mean_3"] = df["revenue"].rolling(
    window=3
).mean()

df.head()
```

Постройте график.

```python
plt.figure(figsize=(10, 5))

plt.plot(
    df["date"],
    df["revenue"],
    marker="o",
    label="Факт"
)

plt.plot(
    df["date"],
    df["rolling_mean_3"],
    marker="o",
    label="Скользящее среднее"
)

plt.title("Выручка и скользящее среднее")
plt.xlabel("Дата")
plt.ylabel("Выручка")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Сделайте краткий вывод: помогает ли скользящее среднее увидеть тренд.

## 7. Создание признаков для простой модели

Создайте лаговые признаки.

```python
df["revenue_lag_1"] = df["revenue"].shift(1)
df["revenue_lag_2"] = df["revenue"].shift(2)
df["revenue_lag_3"] = df["revenue"].shift(3)

df["month"] = df["date"].dt.month

df.head()
```

Удалите строки с пропусками после создания лагов.

```python
df_model = df.dropna().copy()

df_model.head()
```

## 8. Формирование признаков и целевой переменной

```python
features = [
    "revenue_lag_1",
    "revenue_lag_2",
    "revenue_lag_3",
    "rolling_mean_3",
    "month"
]

X = df_model[features]
y = df_model["revenue"]

print(X.shape)
print(y.shape)
```

## 9. Разделение данных на train и test

Во временных рядах данные делятся по времени, без случайного перемешивания.

```python
train_size = int(len(df_model) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

dates_test = df_model["date"].iloc[train_size:]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
```

## 10. Обучение простой модели

```python
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)
```

Постройте прогноз.

```python
linear_pred = linear_model.predict(X_test)
```

Создайте таблицу факта и прогноза.

```python
linear_result = pd.DataFrame({
    "date": dates_test,
    "actual": y_test,
    "predicted": linear_pred
})

linear_result.head()
```

## 11. Оценка качества простой модели

```python
linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)

linear_mape = np.mean(
    np.abs((y_test - linear_pred) / y_test)
) * 100

print("Linear MAE:", round(linear_mae, 2))
print("Linear RMSE:", round(linear_rmse, 2))
print("Linear MAPE:", round(linear_mape, 2), "%")
```

Кратко объясните:

* что показывает MAE;
* что показывает RMSE;
* что показывает MAPE.

## 12. Визуализация простой модели

```python
plt.figure(figsize=(10, 5))

plt.plot(
    linear_result["date"],
    linear_result["actual"],
    marker="o",
    label="Факт"
)

plt.plot(
    linear_result["date"],
    linear_result["predicted"],
    marker="o",
    label="Прогноз Linear Regression"
)

plt.title("Факт и прогноз простой модели")
plt.xlabel("Дата")
plt.ylabel("Выручка")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Сделайте краткий вывод: насколько прогноз близок к фактическим значениям.

## 13. Базовая модель

Базовая модель прогнозирует значение предыдущего месяца.

```python
baseline_pred = X_test["revenue_lag_1"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

print("MAE базовой модели:", round(baseline_mae, 2))
print("MAE простой ML-модели:", round(linear_mae, 2))
```

Сделайте вывод: лучше ли простая ML-модель, чем прогноз по предыдущему месяцу.

## 14. Подготовка данных для SARIMA

Для SARIMA дата должна быть индексом.

```python
df_ts = df[["date", "revenue"]].copy()

df_ts = df_ts.set_index("date")

df_ts = df_ts.asfreq("MS")

df_ts.head()
```

Проверьте пропуски.

```python
df_ts.isna().sum()
```

## 15. Разделение ряда для SARIMA

Последние 6 месяцев используйте для проверки прогноза.

```python
test_size = 6

train_ts = df_ts.iloc[:-test_size]
test_ts = df_ts.iloc[-test_size:]

print("Train:", train_ts.shape)
print("Test:", test_ts.shape)
```

## 16. Обучение SARIMA

```python
sarima_model = SARIMAX(
    train_ts["revenue"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_result = sarima_model.fit(disp=False)
```

Используется модель:

```text
SARIMA(1, 1, 1)(1, 1, 1, 12)
```

Здесь `12` означает годовую сезонность для месячных данных.

## 17. Прогноз SARIMA

```python
sarima_pred = sarima_result.forecast(
    steps=len(test_ts)
)

sarima_result_df = pd.DataFrame({
    "date": test_ts.index,
    "actual": test_ts["revenue"],
    "predicted": sarima_pred
})

sarima_result_df.head()
```

## 18. Оценка качества SARIMA

```python
sarima_mae = mean_absolute_error(
    sarima_result_df["actual"],
    sarima_result_df["predicted"]
)

sarima_rmse = np.sqrt(
    mean_squared_error(
        sarima_result_df["actual"],
        sarima_result_df["predicted"]
    )
)

sarima_mape = np.mean(
    np.abs(
        (sarima_result_df["actual"] - sarima_result_df["predicted"])
        / sarima_result_df["actual"]
    )
) * 100

print("SARIMA MAE:", round(sarima_mae, 2))
print("SARIMA RMSE:", round(sarima_rmse, 2))
print("SARIMA MAPE:", round(sarima_mape, 2), "%")
```

## 19. Визуализация SARIMA

```python
plt.figure(figsize=(10, 5))

plt.plot(
    train_ts.index,
    train_ts["revenue"],
    label="Train"
)

plt.plot(
    test_ts.index,
    test_ts["revenue"],
    marker="o",
    label="Факт"
)

plt.plot(
    sarima_result_df["date"],
    sarima_result_df["predicted"],
    marker="o",
    label="Прогноз SARIMA"
)

plt.title("Прогноз SARIMA")
plt.xlabel("Дата")
plt.ylabel("Выручка")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Сделайте краткий вывод: насколько хорошо SARIMA воспроизводит тестовый период.

## 20. Сравнение моделей

Сравните базовую модель, простую ML-модель и SARIMA.

```python
comparison = pd.DataFrame({
    "model": [
        "Baseline",
        "Linear Regression",
        "SARIMA"
    ],
    "MAE": [
        baseline_mae,
        linear_mae,
        sarima_mae
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, baseline_pred)),
        linear_rmse,
        sarima_rmse
    ],
    "MAPE": [
        np.mean(np.abs((y_test - baseline_pred) / y_test)) * 100,
        linear_mape,
        sarima_mape
    ]
})

comparison
```

Отсортируйте модели по MAE.

```python
comparison.sort_values("MAE")
```

Ответьте кратко:

1. Какая модель имеет минимальную MAE?
2. Какая модель имеет минимальную RMSE?
3. Какая модель имеет минимальную MAPE?
4. Какая модель лучше подходит для данного ряда?

## 21. Прогноз на будущий период

Обучите SARIMA на всем ряде и постройте прогноз на 6 месяцев вперед.

```python
final_sarima_model = SARIMAX(
    df_ts["revenue"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_sarima_result = final_sarima_model.fit(disp=False)

future_steps = 6

future_forecast = final_sarima_result.forecast(
    steps=future_steps
)

future_dates = pd.date_range(
    start=df_ts.index[-1] + pd.DateOffset(months=1),
    periods=future_steps,
    freq="MS"
)

future_df = pd.DataFrame({
    "date": future_dates,
    "forecast": future_forecast.values
})

future_df
```

Постройте график истории и будущего прогноза.

```python
plt.figure(figsize=(10, 5))

plt.plot(
    df_ts.index,
    df_ts["revenue"],
    marker="o",
    label="История"
)

plt.plot(
    future_df["date"],
    future_df["forecast"],
    marker="o",
    label="Прогноз"
)

plt.title("Прогноз выручки на будущий период")
plt.xlabel("Дата")
plt.ylabel("Выручка")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 22. Итоговые вопросы

Ответьте письменно:

1. Почему временной ряд нельзя случайно перемешивать перед разделением на train и test?
2. Какие признаки использовались в простой ML-модели?
3. Что означает сезонный период `12` в SARIMA?
4. Какая модель показала наименьшую ошибку?
5. Лучше ли SARIMA, чем базовая модель?
6. Какие ограничения есть у построенного прогноза?

## Формат сдачи

* ссылка на Google Colab;
* ноутбук выполняется сверху вниз без ошибок;
* построен график временного ряда;
* рассчитаны MAE, RMSE и MAPE;
* построен прогноз SARIMA;
* выполнено сравнение моделей;
* добавлены краткие выводы.
