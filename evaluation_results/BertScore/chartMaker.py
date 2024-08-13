import pandas as pd
import matplotlib.pyplot as plt

file_path = 'summary1_scores.csv'

data = pd.read_csv(file_path)

data['doc_id'] = data['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

pivot_data = data.pivot_table(index='doc_id', columns='model_type', values=[
    'bert_score_precision', 'bert_score_recall', 'bert_score_f1'])

for metric in ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']:
    pivot_data[f'{metric}_diff'] = pivot_data[(metric, 'langchain')] - pivot_data[(metric, 'openai')]

plt.figure(figsize=(12, 6))
mean_diffs = pivot_data[[f'{metric}_diff' for metric in ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']]].mean()
bars = mean_diffs.plot(kind='bar', color=['green' if x > 0 else 'red' for x in mean_diffs])
plt.title('Mean Differences in BERT Scores (LangChain-enhanced GPT 3.5 vs OpenAI GPT 3.5)')
plt.ylabel('Mean Difference')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45, ha='right')


for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{bar.get_height():.4f}', ha='center', va='bottom')

plt.show()

better_langchain = (pivot_data[[f'{metric}_diff' for metric in ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']]] > 0).sum()
better_openai = (pivot_data[[f'{metric}_diff' for metric in ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']]] < 0).sum()


plt.figure(figsize=(12, 6))
bar_width = 0.4
bar_positions = range(len(better_langchain))

bars_langchain = plt.bar([p + bar_width for p in bar_positions], better_langchain, width=bar_width, color='green', label='LangChain')

bars_openai = plt.bar(bar_positions, better_openai, width=bar_width, color='blue', label='OpenAI')

plt.title('Document Count Comparison (LangChain-enhanced GPT 3.5 vs OpenAI GPT 3.5)')
plt.ylabel('Number of Documents')
plt.xticks([p + bar_width / 2 for p in bar_positions], better_langchain.index, rotation=45, ha='right')
plt.legend()

for bar in bars_langchain:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{int(bar.get_height())}', ha='center', va='bottom')

for bar in bars_openai:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{int(bar.get_height())}', ha='center', va='bottom')

plt.show()
