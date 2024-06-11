# СЕРВИС ПРОГНОЗИРОВАНИЯ СТОИМОСТИ АКЦИЙ КРУПНЕЙШИХ КОМПАНИЙ ММВБ
## 1. Обработка данных финансовых рынков и анализ фундаментальных показателей:
- Используется веб-скрейпинг для загрузки списка тикеров топ-10 компаний ММВБ, загружаются котировки с использованием [API Московской Биржи (MOEX)](https://www.moex.com/ru/), дивидендные истории компаний и квартальные отчеты компаний с веб-сайта [smart-lab.ru](https://smart-lab.ru/).
   
- Вычисляются наиболее значимые фундаментальные метрики, их корреляция с капитализацией компании через нормализованный RMSE(среднеквадратичная ошибка) как способность модели их прогнозировать.
  
- Производится прогнозирование дохода компаний и стоимости компании на следующие 10 кварталов и затем для итогового выбора данные резюмируются в одну таблицу.


## 2. **Визуализация данных в Streamlit:**

- [**Интерактивный выбор акции**](#): выпадающие списки акций, слайдеры регулировки диапазона дат, отображение доходности при наведении курсора в любой точке времени.
  
- [**Доверительный диапазон прогноза**](#): график с нижней и верхней границей прогноза  
  
- [**Индивидуально расчитанные корреляции**](#): между наиболее значимыми фундаментальными показателями и стоимостью, дополнительно представлен коэфициент прогнозирования, история и прогнозируемые выплаты дивидендов 

  <div style="display: flex;">
    <img src="img/exmp/10.png" width="800" height="400">
    <img src="img/exmp/11.png" width="800" height="400">
    <img src="img/exmp/12.png" width="820" height="340">
</div>

### Структура проекта:

- **src/**: Исходный код проекта
  - `data_loader.py`: Загрузка данных о ценах акций, прогнозах и дивидендах.
  - `plot_loader.py`: Отображение графиков цен и прогнозов.
  - `income_loader.py`: Отображение данных о доходах и метриках.
  - `utils.py`: Вспомогательные функции для обработки данных и визуализации.
  - `main.py`: Основной скрипт для работы с Streamlit.

- **data/**: Хранение загруженных данных
  - `<ticker>_h_<start>-<stop>.csv`: Котировки для каждого тикера.
  - `<ticker>_dividend_table.csv`: Дивидендная история.
  - `<ticker>_income.csv`: Квартальные отчеты.

- **export/streamlit/**: Экспорт результатов анализа
  - `<ticker>_forecast.csv`: Прогнозы для каждого тикера.
  - `<ticker>_dividend_table.csv`: Дивиденды для каждого тикера.
  - `<ticker>_income.csv`: Квартальные отчеты.
  - `<ticker>_forcast_dividend.csv`: Прогнозы дивидендов.
  - `<ticker>_correlation_table.csv`: Корреляция метрик.
  - `00_sums_data.csv`: Общие данные для анализа.

- **img/**: Изображения для боковой панели.

- `requirements.txt`: Зависимости проекта.
- `README.md`: Описание проекта и инструкции.

### Используемые библиотеки
- [Python](https://www.python.org/) анализа данных и предобработки
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) веб-скрейпинга
- [Pandas](https://pandas.pydata.org/) обработка и анализ данных
- [Matplotlib](https://matplotlib.org/), [Plotly Express]() и [Seaborn](https://seaborn.pydata.org/) визуализация данных
- [Prophet](https://facebook.github.io/prophet/) для прогнозирования временных рядов
- [Streamlit](https://streamlit.io/) для создания веб-приложения (если используется)

### Установка и запуск:
1. Склонируйте репозиторий:

```bash
git clone git@github.com:AlexanderGithubProfile/STOCK_PREDICTION.git
```

2. Установите необходимые зависимости, выполнив команду:
```bash
pip install -r requirements.txt
```
3. Установите Streamlit:

```bash
pip install streamlit
```
4. Запустите скрипт или приложение для обработки данных и анализа финансовых рынков:
```bash
streamlit run src/main.py
```
После запуска приложения откроется веб-интерфейс, где вы сможете выбрать акцию из доступного списка и ознакомиться с прогнозом стоимости, а также другими параметрами.

