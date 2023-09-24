import pandas as pd
import os
import openai
import re
import argparse

parser = argparse.ArgumentParser(description='Dataframes Transformer')
parser.add_argument('--source', type=str, help='Input dir for source df')
parser.add_argument('--template', type=str, help='Input dir for template df')
parser.add_argument('--target', type=str, help='Output dir for result df', default='target.csv')
args = parser.parse_args()

openai.api_key = os.getenv("OPEN_AI_APIKEY")
# загружаем данные
df_to_convert = pd.read_csv(args.source)
template = pd.read_csv(args.template)

# генерируем список примеров данных из первой строки датафрейма, из которых модель подходящие значения
options = []
for i in df_to_convert[:1].values.tolist()[0]:
    options.append(str(i))
options = ', '.join(options)

# контекст для модели
system_prompt = """You are a data analyst. 
Your goal is to choose from the proposed list the option that is most similar to the data from the example. 
Option data and example data may contain names, dates and other kind of data. 
Return only the selected option."""


def generate_prompt(ex: str, op: str) -> str:
    """
    Функция генерирует текст запроса в модель.
    :param ex: Примеры данных
    :param op: Список опций для выбора
    :return: Объединенную строку с примерами и опциями.
    """
    return f"""Example data: {ex}.
    
List of options: {op}"""


# цикл запросов к модели, в ходе которого формируется список ответов модели.
result_values = []
for i_col in template.columns:
    examples = ', '.join(template[i_col].astype('str').tolist())
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generate_prompt(examples, options)},
        ])
    result_values.append(response['choices'][0]['message']['content'])

# получаем индексы целевых столбцов на основании ответов модели
result_columns = []
for i in result_values:
    idx = options.split(', ').index(i)
    result_columns.append(df_to_convert.columns[idx])

# делаем срез столбцов и формируем итоговый датафрейм
result_df = df_to_convert[result_columns]
result_df.columns = template.columns


system_prompt2 = """You are a data scientist analyzing two tables. Both tables contain the same data, but may differ in their format. 
When analyzing data you use the Pandas library.
Your goal is to write a Python finction that converts data format from the first table to the data format from the second table. 
You can only change data types and replace or remove punctuation. 
Python function should be like convert_data(df, column_name).
If the data doesn't need processing, return 'convert_data(df, column_name): return df'.
"""

def generate_prompt_for_func(data1, data2, col_name):
    return f"""Data from the first table: {data2}
    
Data from the second table: {data1}

The name of column is {col_name}

Return only the function without importing libraries, don't add any extra description, text and markdown syntax.
"""


for i_col in result_df.columns:
    examples = ', '.join(template[i_col].sample(8).astype('str').tolist())
    data_to_analyse = ', '.join(result_df[i_col].sample(8).astype('str').tolist())
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
        {"role": "system", "content": system_prompt2},
        {"role": "user", "content": generate_prompt_for_func(examples, data_to_analyse, i_col)}],
        temperature=0)
    response_code = response['choices'][0]['message']['content']
    exec(response_code)
    result_df = convert_data(result_df, i_col)

# сохраняем результат. Значение по умолчанию - текущая директория
result_df.to_csv(args.target, index=False)
