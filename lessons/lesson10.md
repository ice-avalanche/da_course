# Урок 10. Визуализация данных

## 1. Цель занятия

Освоить методы визуализации данных и научиться:

* выбирать тип графика под задачу;
* строить графики в Python;
* анализировать распределения и взаимосвязи;
* визуализировать категории;
* оформлять графики;
* избегать ошибок.

## 2. matplotlib

matplotlib – базовая библиотека визуализации, обеспечивающая полный контроль над построением графиков.

```python
import matplotlib.pyplot as plt
```

### 2.1 Линейный график

Линейный график используется для анализа динамики показателя во времени или по упорядоченной оси.

```python
plt.plot(df["month"], df["revenue"])
```

### 2.2 Bar plot

Столбчатая диаграмма применяется для сравнения значений между категориями.

```python
plt.bar(df["month"], df["revenue"])
```

### 2.3 Scatter plot

Диаграмма рассеяния используется для анализа взаимосвязи между двумя числовыми переменными.

```python
plt.scatter(df["revenue"], df["revenue"])
```

### 2.4 Histogram

Гистограмма показывает распределение значений и частоту их появления.

```python
plt.hist(df["revenue"], bins=5)
```

### 2.5 Несколько линий

Позволяет сравнивать динамику нескольких показателей на одном графике.

```python
plt.plot(df["month"], df["revenue"], label="revenue")
plt.legend()
```

### 2.6 Area plot

Площадной график подчеркивает накопление или вклад показателя во времени.

```python
plt.fill_between(df["month"], df["revenue"])
```

### 2.7 Step plot

Используется для дискретных изменений показателя.

```python
plt.step(df["month"], df["revenue"])
```

## 3. seaborn

seaborn – надстройка над matplotlib, упрощающая построение статистических графиков.

```python
import seaborn as sns
```

### 3.1 Line plot

Позволяет автоматически учитывать агрегирование и доверительные интервалы.

```python
sns.lineplot(data=df, x="month", y="revenue")
```

### 3.2 Категории (hue)

Позволяет разбить данные на группы и сравнить их поведение.

```python
df["region"] = ["Москва","СПб","Москва","СПб"]

sns.lineplot(data=df, x="month", y="revenue", hue="region")
```

### 3.3 Bar plot

Отображает агрегированные значения (по умолчанию — среднее).

```python
sns.barplot(data=df, x="month", y="revenue")
```

### 3.4 Violin plot

Показывает распределение с учетом плотности.

```python
sns.violinplot(data=df, x="region", y="revenue")
```

### 3.5 Pairplot

Позволяет быстро исследовать зависимости между всеми числовыми признаками.

```python
sns.pairplot(df)
```

## 4. Распределения

Анализ распределений позволяет выявлять структуру данных и выбросы.

### 4.1 Histogram

```python
sns.histplot(df["revenue"], bins=10, kde=True)
```

Дополнительно отображает оценку плотности распределения.

### 4.2 Boxplot

```python
sns.boxplot(x=df["revenue"])
```

Показывает медиану, квартильный размах и выбросы.

### 4.3 По категориям

```python
sns.boxplot(data=df, x="region", y="revenue")
```

Позволяет сравнивать распределения между группами.

### 4.4 KDE plot

```python
sns.kdeplot(df["revenue"])
```

Используется для сглаженной оценки плотности.

## 5. Взаимосвязи

Используются для выявления зависимостей между переменными.

### 5.1 Scatter

```python
sns.scatterplot(data=df, x="revenue", y="revenue")
```

### 5.2 С категориями

```python
sns.scatterplot(data=df, x="revenue", y="revenue", hue="region")
```

Позволяет анализировать структуру зависимостей внутри групп.

### 5.3 Регрессия

```python
sns.regplot(data=df, x="revenue", y="revenue")
```

Добавляет линию тренда.

### 5.4 Корреляции

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
```

Отображает силу линейной связи между признаками.

## 6. Категориальные данные

Используются для анализа различий между группами.

### 6.1 Bar plot

```python
sns.barplot(data=df, x="region", y="revenue")
```

### 6.2 Count

```python
sns.countplot(data=df, x="region")
```

Показывает количество наблюдений.

### 6.3 С hue

```python
sns.barplot(data=df, x="region", y="revenue", hue="month")
```

### 6.4 Сортировка

```python
agg = df.groupby("region")["revenue"].sum().sort_values().reset_index()
sns.barplot(data=agg, x="region", y="revenue")
```

Позволяет ранжировать категории.

## 7. Настройка графиков

Настройка повышает читаемость и интерпретируемость визуализации.

```python
plt.figure(figsize=(10,5))

plt.plot(
    df["month"],
    df["revenue"],
    marker="o",
    linestyle="--",
    linewidth=2
)

plt.title("Выручка по месяцам", fontsize=14)
plt.xlabel("Месяц", fontsize=12)
plt.ylabel("Выручка", fontsize=12)

plt.xticks(rotation=45)
plt.yticks(fontsize=10)

plt.grid(True)

plt.legend(["Revenue"])

plt.tight_layout()
plt.show()
```

Дополнительные настройки позволяют более точно управлять отображением графика и акцентировать внимание на ключевых элементах данных.

```python
plt.style.use("ggplot")     # стиль графика

plt.xlim(0, 10)             # границы по оси X
plt.ylim(0, 500)            # границы по оси Y

plt.axhline(y=200)          # горизонтальная линия
plt.axvline(x=2)            # вертикальная линия

plt.annotate(
    "max value",
    xy=(3, 400),
    xytext=(2, 450),
    arrowprops=dict(arrowstyle="->")
)
```
