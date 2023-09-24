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

# переводим все значения в строки, чтобы в дальнейшем избежать ошибок
df_to_convert = df_to_convert.astype('str')
template = template.astype('str')

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
    examples = ', '.join(template[i_col].tolist())
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

# проводим обработку данных для приведения их к формату датафрейма-образца
result_df['Date'] = pd.to_datetime(result_df['Date']).apply(lambda x: x.strftime('%m-%d-%Y'))
result_df['PolicyNumber'] = result_df['PolicyNumber'].apply(lambda x: re.sub(r'[^A-Z0-9]', '', x))
result_df['Premium'] = result_df['Premium'].apply(lambda x: int(float(x)))

# сохраняем результат. Значение по умолчанию - текущая директория
result_df.to_csv(args.target, index=False)
