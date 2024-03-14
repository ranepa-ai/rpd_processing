import streamlit as st
import pandas as pd
import ast
from graphviz import Digraph
from PIL import Image
import matplotlib.pyplot as plt
import json


def assign_colors(values):
    sorted_values = sorted(values) 
    min_values = sorted_values[:5] 
    max_values = sorted_values[-5:]  

    color_high = '#E67300'
    color_medium = '#FFE14D'
    color_low = '#80FF66'

    color_map = [] 
    for value in values:
        if value in max_values:
            color_map.append(color_high)
        elif value in min_values:
            color_map.append(color_low)
        else:
            color_map.append(color_medium)

    return color_map

def find_last_smaller(sem_list, second):
    result = None
    for num in reversed(sem_list):
        if num < second:
            result = num
            break
    return result

def create_nodes_and_edges(df, column_name, legacy=False):
    color_mapping = assign_colors(df[column_name])

    # Создание словарей для узлов и рёбер по кластерам
    nodes_dict = {}
    edges_dict = {}

    for sem in range(1, 9):
        edges_dict[sem] = []
        nodes_dict[sem] = [[f'Семестр_номер_{sem}', '#FFFFFF']]
        if sem > 1:
            edges_dict[sem].append((f'Семестр_номер_{sem-1}', f'Семестр_номер_{sem}'))

    for index, row in df.iterrows():
        for sem_i, sem in enumerate(row['Семестр']):
            node_name = f'{row["Дисциплина"]}_{sem}'
            node_color = color_mapping[index]

            nodes_dict[sem].append([node_name, node_color])
            if sem_i == 0 and len(row['Реализуется после']) != 0 and legacy:
                for second in row['Реализуется после'][:2]:
                    if second in set(df['Дисциплина']):
                      sem_list = df[df['Дисциплина'] == second]['Семестр'].iloc[0]
                      sem_sec = find_last_smaller(sem_list, sem)
                      if sem_sec:
                        edges_dict[sem].append((f'{second}_{sem_sec}', node_name))
            elif sem_i > 0:
                prev_sem = row['Семестр'][sem_i - 1]
                prev_node_name = f'{row["Дисциплина"]}_{prev_sem}'
                edges_dict[sem].append((prev_node_name, node_name))

    return nodes_dict, edges_dict

df = pd.read_csv('preproc_data.csv')
df['Семестр'][df['Семестр'].isna()]='1'
df['Семестр'] = df['Семестр'].apply(lambda x: sorted([int(j) if int(j) < 9 else 8 for j in x.split(', ')]))


#df['Семестр'] = df['Семестр'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['Реализуется после'] = df['Реализуется после'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#df['Семестр'] = df['Семестр'].apply(lambda x: sorted([int(i) for i in x]))
df['Направленность'] = df['Направленность'].apply(lambda x: x[1:-1])

uniqueness_df = pd.read_csv('unique.csv')  # Путь к вашему файлу unique.csv

# Объединяем DataFrame по общей колонке 'Полный индекс дисциплины'
df = pd.merge(df, uniqueness_df[['Полный индекс дисциплины', 'uniqueness']], on='Полный индекс дисциплины', how='left')
df_all = df
# Создание интерактивных виджетов для выбора фильтров
qualification = st.selectbox('Выберите квалификацию:', df['Квалификация'].unique())
year_options = df[df['Квалификация'] == qualification]['Год набора'].unique()
year = st.selectbox('Выберите год набора:', year_options, index=0 if year_options.size > 0 else None)
education_form_options = df[(df['Квалификация'] == qualification) & (df['Год набора'] == year)]['Форма обучения'].unique()
education_form = st.selectbox('Выберите форму обучения:', education_form_options, index=0 if education_form_options.size > 0 else None)
direction_options = df[(df['Квалификация'] == qualification) & (df['Год набора'] == year) & (df['Форма обучения'] == education_form)]['Направленность'].unique()
direction = st.selectbox('Выберите направленность:', direction_options, index=0 if direction_options.size > 0 else None)
mask = st.selectbox('Выберите маску:', ['Компетенции (уникальность в пределах всех РПД)', 'Важность скиллов со стороны рынка', 'Важность академического материала', 'NLP-анализ текстов программ'])
leg = st.selectbox('Показывать прешествующие дисциплины (тестовый режим):', [False, True])

d_dict = {'Компетенции (уникальность в пределах всех РПД)':'Competence_TFIDF', 'Важность скиллов со стороны рынка':'Training_Outcomes_TFIDF', 'Важность академического материала':'Topics_TFIDF', 'NLP-анализ текстов программ':'uniqueness'}
# Кнопка подтверждения выбранных фильтров
# Применение выбранных фильтров к DataFrame
filtered_df = df[(df['Квалификация'] == qualification) & 
                    (df['Год набора'] == year) & 
                    (df['Форма обучения'] == education_form) & 
                    (df['Направленность'] == direction)]

df = filtered_df.drop(['Unnamed: 0'], axis=1)
df.drop_duplicates(subset='Дисциплина', keep='first', inplace=True)
#df.drop_duplicates(subset='РПД', keep='first', inplace=True)
df = df.reset_index(drop=True)
df['Семестр'] = df['Семестр'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['Реализуется после'] = df['Реализуется после'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['Семестр'] = df['Семестр'].apply(lambda x: sorted([int(i) for i in x]))
df['Дисциплина'] = df['Дисциплина'].apply(lambda x: x[1:-1])

if df.empty:
    st.write('Нет данных для выбранных параметров.')
else:
    st.write(df)

nodes_dict, edges_dict = create_nodes_and_edges(df, d_dict[mask], leg)
# Создание объекта графа
g = Digraph('G', format='pdf', filename=f'{"first"}_cluster.gv', engine='dot')

# Установка общих параметров графа
g.attr(rankdir='LR', ranksep='0.2', nodesep='0.2', margin="1.5,0.5",splines='line')
# Проход по словарям узлов и рёбер
for sem, nodes in nodes_dict.items():
    # Создание подграфа для каждого семестра
    with g.subgraph(name=f'cluster_{sem}') as c:
        # Установка стилей для подграфа
        c.attr(rank = 'same', style='filled', color='lightgrey', label=f'Semester #{sem}', fontsize='16',rankdir='LR')
        # Установка стилей для узлов в подграфе
        # Добавление узлов в подграф

        c.node_attr.update(style='filled', shape='box3d', fontsize='6', width='0.5', height='0.2')

        for node in nodes[:1]:
            c.node(node[0], fillcolor=node[1], fontsize='12', area='5')

        # Добавление рёбер в подграф
        for edge in edges_dict[sem][:1]:
            c.edge(*edge)


        # Добавление узлов в подграф
        for node in nodes[1:]:
            c.node(node[0], fillcolor=node[1])

        # Добавление рёбер в подграф
        for edge in edges_dict[sem][1:]:
            c.edge(*edge)

g.view()
st.graphviz_chart(g, use_container_width=True)

with open(f'{"first"}_cluster.gv.pdf', 'rb') as f:
    pdf_bytes = f.read()
st.download_button(label='Увеличить PDF', data=pdf_bytes, file_name='first_cluster.gv.pdf', mime='application/pdf')



fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i, (col, label) in enumerate(d_dict.items()):
    ax = axs[i // 2, i % 2]
    filtered_df[label].plot(kind='hist', ax=ax, alpha=0.5, color='red')
    ax.set_xlabel(col)
    ax.set_ylabel('Частота')
    mean_value = round(filtered_df[label].mean(), 4)
    ax.axvline(mean_value, color='red', linestyle='--', label=f'Среднее: {mean_value}')
    mean_value = round(df_all[label].mean(), 4)
    ax.axvline(mean_value, color='blue', linestyle='--', label=f'Все данные: {mean_value}')
    ax.legend()
st.pyplot(fig)

df2 = pd.read_csv('preproc_data_2.csv')
df_2 = pd.read_json('common_sim.json').T.reset_index()
df_2['РПД'] = df_2['index']
df2 = pd.merge(df2, df_2, on='РПД', how='left')
df2['Направленность'] = df2['Направленность'].apply(lambda x: x[1:-1])
df2['Дисциплина'] = df2['Дисциплина'].apply(lambda x: x[1:-1])

# Получить список уникальных индексов из df
unique_indices = df['Полный индекс дисциплины'].unique()

# Вывести selectbox для выбора уникального индекса
selected_index = st.selectbox("Наиболее схожие дисциплины из других направлений, на основе анализа РПД", unique_indices)

# Получить РПД из df2, соответствующие выбранному индексу
relevant_df2 = df2[df2['Полный индекс дисциплины'] == selected_index].iloc[0]['2 наиболее схожие РПД']

# Если есть РПД для выбранного индекса, вывести их
if relevant_df2:
    relevant_rpd_info = df2[df2['РПД'].isin(relevant_df2)][['РПД', 'Направленность', 'Дисциплина']]
    st.write(relevant_rpd_info)
else:
    st.write("Для выбранной дисциплины нет соответствующих РПД.")

st.write("Словарь nodes_dict:")
st.json(json.dumps(nodes_dict))
st.write("Словарь edges_dict:")
st.json(json.dumps(edges_dict))