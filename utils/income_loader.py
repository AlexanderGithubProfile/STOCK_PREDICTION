import streamlit as st
import plotly.graph_objects as go
import numpy as np

def income_loader(forecast, income, forecast_income):
    """
    Функция для загрузки и отображения графиков дохода компании.

    Args:
        TICKER (str): Тикер компании.
        forecast (pd.DataFrame): DataFrame с прогнозом дохода компании.
        income (pd.DataFrame): DataFrame с фактическими данными дохода компании.

    Returns:
        None
    """
    # Разделение страницы на две колонки
    left, right = st.columns([3, 2])
    
    # Отображение значений P/E рядом с графиком
    with right:
        st.markdown(f"""<div style='font-size: 20px; 
                    padding-left: 300px; 
                    text-align: left;
                    '><br><br><br>P/E при будущих доходах и текущей цене P/E<span 
                    style='font-size: 100px;
                    padding-left: 30px;
                    '>{np.round(income['Капитализация, млрд руб'].iloc[-1] / (forecast_income.iloc[-4:].yhat.sum()),2):.2f}</span></div>""", 
                    unsafe_allow_html=True)
        st.markdown(f"""<div style='font-size: 20px; 
                    padding-left: 300px; 
                    text-align: left;
                    '>P/E при текущей цене акции и доходе P/E<span 
                    style='font-size: 100px;
                    padding-left: 20px;
                    '>{np.round(income['Капитализация, млрд руб']
                    .iloc[-1]/income.y.iloc[-4:]
                    .sum()):.2f}</span></div>""", 
                    unsafe_allow_html=True)
    
    # Отображение графика прогноза дохода
    with left:    
        fig4 = go.Figure(data=[
            # Предсказанные значения
            go.Scatter(
                x=forecast_income['ds'], 
                y=forecast_income['yhat'],
                mode='lines',  
                name='Прогноз',
                marker_color='orange'
            ),
            # График цен акций
            go.Bar(
                x=forecast_income['ds'],  # Временные метки
                y=income['y'],   # Фактические значения
                name='Факт. знач.',
                marker=dict(color='rgba(82, 126, 233, 0.8)'),
            ),
            # Врехняя граница прогноза
            go.Scatter(
                x=forecast_income['ds'], 
                y=forecast_income['yhat_upper'], 
                mode='lines',
                showlegend=False,
                fill='tonexty',
                fillcolor='rgba(186, 191, 204, 0.05)', 
                line=dict(color='rgba(186, 191, 204, 0.1)')
            ),
            # Нижняя граница прогноза
            go.Scatter(
                x=forecast_income['ds'], 
                y=forecast_income['yhat_lower'],
                name='Гран.прогн.', 
                mode='lines', 
                fill='tonexty',
                fillcolor='rgba(186, 191, 204, 0.05)',  
                line=dict(color='rgba(186, 191, 204, 0.1)')
            )
        ])
        # Общие настройки размера и подпись осей
        fig4.update_layout(
                title=dict(text='Прогноз дохода компании', font=dict(size=30)),
                xaxis_title='Дата',
                yaxis_title='Цена',
                height=600,
                width=1150,
                xaxis_rangeslider_visible=True
        ) 
        # Фильтр отображения прогнозной части
        fig4.update_yaxes(
                range=[forecast_income['yhat'].min()-50, 
                        max(max(income['y']), 
                        forecast_income['yhat'].max()+50)]
        )
        
        # Добавление горизонтальной линии к последнему предсказанному значению
        last_predicted_price = forecast_income['yhat'].iloc[-1]
        last_predicted_date = forecast_income['ds'].max()
        fig4.add_shape(type="line",
                x0=income['ds'].min(), 
                y0=last_predicted_price,
                x1=forecast['ds'].max(), 
                y1=last_predicted_price,
                opacity=0.5,  # Прозрачность
                line=dict(color="grey", dash="dashdot"),
                name='Последняя предсказанная цена'
                        )
        # Добавление вертикальной линии к последнему предсказанному значению
        fig4.add_shape(
                type="line",
                x0=last_predicted_date, 
                y0=0,
                x1=last_predicted_date, 
                y1=last_predicted_price,
                line=dict(color="grey", dash="dashdot"),
                opacity=0.5,  # Прозрачность
                name='Последняя дата предсказания'
        )
        # Бирка на графике с резульатами в последний день прогноза
        fig4.add_annotation(
                x=last_predicted_date,
                y=last_predicted_price,
                text=f'({last_predicted_date.strftime("%b %d, %Y")}, {int(last_predicted_price)} Прогноз {int(((last_predicted_price/ income.iloc[-2:].y.mean() - 1)*100).round()):+d}%)',
                showarrow=True,
                arrowhead=1,
                ax=0,  # Нет смещения по оси x
                ay=-40,  # Смещение вниз от точки
                arrowcolor='orange',  # Оранжевый цвет стрелки
                font=dict(color="black", size=18),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1.4,
                borderpad=10,
                align="right"
        )
        # Отображение графика в Streamlit
        return st.plotly_chart(fig4)
