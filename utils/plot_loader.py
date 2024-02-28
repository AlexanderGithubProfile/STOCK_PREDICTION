import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def plot_loader(TICKER, data, forecast, income):
    """
    Функция для загрузки и отображения графиков.

    Parameters:
    TICKER (str): Код тикера компании.
    data (pandas.DataFrame): Данные о ценах акций компании.

    Returns:
    None
    """
    # Разделение экрана на колонки
    left, right = st.columns([3, 2])

    # Отображение значений доходностей рядом с графиком
    with right:
        # Дополнительная обработка данных для отображения
        data_ = pd.read_csv('export/streamlit/00_sums_data.csv')
        data_.name = data_.name.apply(str.upper)
        data_.set_index('name', inplace=True)

        # Прогнозная доходность
        st.markdown(f"""<div style=' 
                            font-size: 20px;  
                            padding-left: 300px; 
                            text-align: left;
                            '><br><br><br>прогнозная доходность <span
                            style='font-size: 100px;
                            padding-left: 30px;
                            '>{int(data_.loc[TICKER]['%_change'] .round()):+d}%</span>
                            </div>""", 
                            unsafe_allow_html=True)

        # Нижняя граница риска
        st.markdown(f"""<div style='
                            font-size: 20px; 
                            padding-left: 300px; 
                            text-align: left;
                            '>при оценке границы риска <span 
                            style='font-size: 100px;
                            padding-left: 20px;'
                            >{int(data_.loc[TICKER]['potential_risk'].round()):+d}%</span>
                            </div>""", 
                            unsafe_allow_html=True)

    # Основной график
    with left:
        # Фильтрация данных для отображения последней части линии прогноза
        forecast_filtered = forecast[(forecast['ds'] >= '2022-02-20') 
                                & (forecast['ds'] <= forecast['ds'].max())]

        # Создание графика
        fig3 = go.Figure(data=[
                # Прогноз
                go.Scatter(x=forecast_filtered['ds'], 
                            y=forecast_filtered['yhat'], 
                            mode='lines',
                            name='Прогноз',  
                            line=dict(color='orange', width=4.5)),

                # Фактические значения 
                go.Scatter(x=data['begin'],  
                            y=data['open'], 
                            mode='lines', 
                            name='Факт. знач.',
                            line=dict(color='rgba(82, 126, 233, 0.8)', width=1)),

                # Верхняя граница прогноза
                go.Scatter(x=forecast_filtered['ds'], 
                            y=forecast_filtered['yhat_upper'], 
                            mode='lines',
                            showlegend=False,
                            fill='tonexty',
                            fillcolor='rgba(186, 191, 204, 0.05)', 
                            line=dict(color='rgba(186, 191, 204, 0.15)')),

                # Нижняя граница прогноза
                go.Scatter(x=forecast_filtered['ds'], 
                            y=forecast_filtered['yhat_lower'],
                            name='Гран.прогн.', 
                            mode='lines', 
                            fill='tonexty',
                            fillcolor='rgba(186, 191, 204, 0.05)',  
                            line=dict(color='rgba(186, 191, 204, 0.15)'))
        ])

        # Настройка макета графика
        fig3.update_layout(title=dict(
                            text=TICKER,
                            font=dict(size=60)),
                            xaxis_title='Дата',
                            yaxis_title='Цена',
                            height=600,
                            width=1150,
                            xaxis_rangeslider_visible=True
                            ) 

        # Настройка стартового положения слайдера
        fig3.update_xaxes(rangeslider=dict(visible=True), 
                                            range=['2021-05-20', 
                                            forecast['ds'].max()])

        # Калибровка осей
        fig3.update_yaxes(range=[forecast['yhat'].min()-50, forecast['yhat'].max()+50])
        fig3.update_xaxes(range=['2021-05-01', '2025-05-01'])

        # Добавление штрих-пунктира к последней предсказанной цене акции
        last_pred_price = forecast_filtered['yhat'].iloc[-1]
        last_predicted_date = forecast_filtered['ds'].max()
        # Горизонтальная линия
        fig3.add_shape(type="line",
                            x0=data['begin'].min(), y0=last_pred_price,
                            x1=forecast_filtered['ds'].max(), y1=last_pred_price,
                            opacity=0.5,  # Прозрачность
                            line=dict(color="grey", dash="dashdot"),
                            name='Последняя предсказанная цена')
        # Вертикальная линия
        fig3.add_shape(type="line",
                            x0=last_predicted_date, y0=0,
                            x1=last_predicted_date, y1=last_pred_price,
                            line=dict(color="grey", dash="dashdot"),
                            opacity=0.5,  # Прозрачность
                            name='Последняя дата предсказания')

        # Табличка с финальной ценой прогноза и стрелка
        fig3.add_annotation(
                            x=last_predicted_date, 
                            y=last_pred_price,
                            text=f'({last_predicted_date.strftime("%b %d, %Y")}, {int(last_pred_price)}) Прогноз {int(((last_pred_price/ income.iloc[-2:].y.mean() - 1)*100).round()):+d}%',
                            showarrow=True,
                            arrowhead=1,
                            ax=0,  
                            ay=-40,  
                            arrowcolor='orange',  
                            font=dict(color="black", size=16),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1.4,
                            borderpad=10,
                            align="right"
                            )
        # Отображение графика
        st.plotly_chart(fig3)
