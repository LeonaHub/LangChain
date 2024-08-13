import pandas as pd
import matplotlib.pyplot as plt

file_path = 'summary3_scores.csv' 
data = pd.read_csv(file_path)

data['model_type'] = data['filename'].apply(lambda x: 'langchain' if 'langchain' in x else 'openai')

langchain_rows = data[data['model_type'] == 'langchain']
openai_rows = data[data['model_type'] == 'openai']

langchain_rows = langchain_rows.reset_index(drop=True)
openai_rows = openai_rows.reset_index(drop=True)

rouge_diff = pd.DataFrame()
for metric in [
    'rouge_1_recall', 'rouge_1_precision', 'rouge_1_f_score',
    'rouge_2_recall', 'rouge_2_precision', 'rouge_2_f_score',
    'rouge_l_recall', 'rouge_l_precision', 'rouge_l_f_score'
]:
    rouge_diff[f'{metric}_diff'] = langchain_rows[metric] - openai_rows[metric]

better_langchain = (rouge_diff > 0).sum()
better_openai = (rouge_diff < 0).sum()

plt.figure(figsize=(12, 6))

langchain_bars = better_langchain.plot(kind='bar', color='green', width=0.4, position=1, label='LangChain')

for i in langchain_bars.patches:
    plt.text(i.get_x() + i.get_width() / 2, i.get_height(), f'{int(i.get_height())}', 
             ha='center', va='bottom')

openai_bars = better_openai.plot(kind='bar', color='blue', width=0.4, position=0, label='OpenAI')

for i in openai_bars.patches:
    plt.text(i.get_x() + i.get_width() / 2, i.get_height(), f'{int(i.get_height())}', 
             ha='center', va='bottom')

plt.title('Document Count Comparison (LangChain vs OpenAI)')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()
