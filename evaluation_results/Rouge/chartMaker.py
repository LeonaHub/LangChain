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

mean_diffs = rouge_diff.mean()

plt.figure(figsize=(12, 6))
bars = mean_diffs.plot(kind='bar', color=['red' if x < 0 else 'green' for x in mean_diffs])
plt.title('Mean Difference In ROUGE Scores (LangChain vs OpenAI)')
plt.ylabel('Mean Difference')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45, ha='right')

for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{bar.get_height():.4f}', ha='center', va='bottom')

plt.show()
