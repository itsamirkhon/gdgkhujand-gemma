import streamlit as st
import pandas as pd
import plotly.express as px
from llama_cpp import Llama
import os
import io 

model_path = "ПУТЬ_К_ВАШЕЙ_МОДЕЛИ"


if not os.path.exists(model_path):
    st.error(f"Ошибка: Файл модели не найден по пути: {model_path}")
    st.info("Пожалуйста, замените 'ПУТЬ_К_ВАШЕЙ_МОДЕЛИ_GGUF' на фактический путь к вашему файлу .gguf")
    st.stop() 

@st.cache_resource
def load_gemma_model(model_path):
    """Загружает модель Gemma."""
    print("Загрузка модели Gemma...") 
    try:
 
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
        print("Модель загружена.")
        return llm
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()

llm = load_gemma_model(model_path)

def analyze_data_with_gemma(dataframe_info, user_question, model):
    """Отправляет информацию о датафрейме и вопрос пользователя модели Gemma."""

    prompt = f"""
    Based on the following dataset structure:
    {dataframe_info}

    The user is asking the following question or requesting insights about this data:
    "{user_question}"

    Please provide a helpful, clear, and concise response focusing specifically on understanding the data, potential relationships between columns, and actionable ideas for visualization based on the dataset structure. Avoid conversational filler.

    Response:
    """

    print(f"Анализ данных с Gemma (вопрос пользователя)...") 

    output = model(
        prompt,
        max_tokens=1000, 
        stop=["\n\n"], 
        temperature=0.7, 
        echo=False
    )

    generated_text = output["choices"][0]["text"]
    return generated_text.strip()

if 'gemma_analysis_result' not in st.session_state:
    st.session_state.gemma_analysis_result = None

st.title("AI Powered Data Visualizer с Gemma")

st.write("Загрузите ваш CSV файл, чтобы начать анализ и визуализацию с помощью локальной модели Gemma.")

uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

dataframe = None 

if uploaded_file is not None:
    try:

        dataframe = pd.read_csv(uploaded_file)
        st.success("Файл успешно загружен!")
        st.subheader("Первые 5 строк данных:")
        st.dataframe(dataframe.head())
        st.subheader("Информация о столбцах:")

        buffer = io.StringIO()
        dataframe.info(buf=buffer, verbose=True, memory_usage=False, show_counts=False)
        cols_info_str = buffer.getvalue() 

        st.text(cols_info_str)

        st.subheader("Анализ данных с помощью Gemma")
        st.write("Задайте вопрос о данных или попросите идеи для анализа.")
        user_data_question = st.text_area("Ваш вопрос:", key="data_analysis_question", height=100) 

        col_analyze, col_clear = st.columns([1, 1]) 

        with col_analyze:
            if st.button("Получить анализ от Gemma"):
                if user_data_question:
                     with st.spinner("Gemma анализирует данные..."):
                         st.session_state.gemma_analysis_result = analyze_data_with_gemma(cols_info_str, user_data_question, llm)
                else:
                     st.warning("Пожалуйста, введите ваш вопрос или запрос для анализа.")

        with col_clear:
             if st.button("Очистить анализ"):
                 st.session_state.gemma_analysis_result = None
                 st.rerun() 

        if st.session_state.gemma_analysis_result:
             st.subheader("Ответ Gemma:")
             st.markdown(st.session_state.gemma_analysis_result)

        st.subheader("Построение графика вручную")
        all_columns = dataframe.columns.tolist()
        x_axis_options = [None] + all_columns
        y_axis_options = [None] + all_columns

        x_axis = st.selectbox("Выберите столбец для оси X:", x_axis_options, key="x_axis_manual_select")
        y_axis = st.selectbox("Выберите столбец для оси Y:", y_axis_options, key="y_axis_manual_select")

        plot_type = st.selectbox(
            "Выберите тип графика:",
            ("Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Box Plot", "Violin Plot", "Area Chart", "Pie Chart", "Density Heatmap", "Sunburst", "Treemap"),
            key="plot_type_manual_select"
        )

        if st.button("Построить выбранный график"):
            try:
                st.subheader("Ваш график:")
                if plot_type == "Scatter Plot" and x_axis and y_axis:
                    fig = px.scatter(dataframe, x=x_axis, y=y_axis, title=f"{plot_type} ({x_axis} vs {y_axis})")
                    st.plotly_chart(fig)
                elif plot_type == "Line Plot" and x_axis and y_axis:
                     fig = px.line(dataframe, x=x_axis, y=y_axis, title=f"{plot_type} ({x_axis} vs {y_axis})")
                     st.plotly_chart(fig)
                elif plot_type == "Bar Chart" and x_axis:
                     if y_axis and y_axis in dataframe.columns:
                          fig = px.bar(dataframe, x=x_axis, y=y_axis, title=f"{plot_type} ({x_axis} vs {y_axis})")
                     else: 
                          counts = dataframe[x_axis].value_counts().reset_index()
                          counts.columns = [x_axis, 'Count']
                          fig = px.bar(counts, x=x_axis, y='Count', title=f"Количество по {x_axis}")
                     st.plotly_chart(fig)
                elif plot_type == "Histogram" and x_axis:
                     fig = px.histogram(dataframe, x=x_axis, title=f"{plot_type} of {x_axis}")
                     st.plotly_chart(fig)
                elif plot_type == "Box Plot" and x_axis:
                     fig = px.box(dataframe, x=x_axis, y=y_axis, title=f"{plot_type} ({x_axis}" + (f" vs {y_axis})" if y_axis else ")"))
                     st.plotly_chart(fig)
                elif plot_type == "Violin Plot" and x_axis:
                     fig = px.violin(dataframe, x=x_axis, y=y_axis, title=f"{plot_type} ({x_axis}" + (f" vs {y_axis})" if y_axis else ")"))
                     st.plotly_chart(fig)
                elif plot_type == "Area Chart" and x_axis and y_axis:
                     fig = px.area(dataframe, x=x_axis, y=y_axis, title=f"{plot_type} ({x_axis} vs {y_axis})")
                     st.plotly_chart(fig)
                elif plot_type == "Pie Chart" and x_axis:
                     if y_axis and y_axis in dataframe.columns:
                          fig = px.pie(dataframe, values=y_axis, names=x_axis, title=f"{plot_type} ({y_axis} by {x_axis})")
                     else: 
                          counts = dataframe[x_axis].value_counts().reset_index()
                          counts.columns = [x_axis, 'Count']
                          fig = px.pie(counts, values='Count', names=x_axis, title=f"Распределение по {x_axis}")
                     st.plotly_chart(fig)
                elif plot_type == "Density Heatmap" and x_axis and y_axis:
                     fig = px.density_heatmap(dataframe, x=x_axis, y=y_axis, title=f"{plot_type} ({x_axis} vs {y_axis})")
                     st.plotly_chart(fig)
                elif plot_type == "Sunburst" and x_axis:
                     fig = px.sunburst(dataframe, path=[x_axis], title=f"{plot_type} ({x_axis})")
                     st.plotly_chart(fig)
                elif plot_type == "Treemap" and x_axis:
                     fig = px.treemap(dataframe, path=[x_axis], title=f"{plot_type} ({x_axis})")
                     st.plotly_chart(fig)
                else:
                     st.warning(f"Для графика типа '{plot_type}' требуется выбор соответствующих столбцов.")


            except Exception as e:
                st.error(f"Ошибка при построении графика: {e}")

    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")

else:
    st.info("Пожалуйста, загрузите CSV файл, чтобы начать.")

st.markdown("---")
st.write("Это демонстрация возможностей локальной модели Gemma для анализа данных и помощи в визуализации.")

