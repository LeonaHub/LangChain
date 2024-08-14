import pandas as pd
import matplotlib.pyplot as plt

file_path = 'summaries3_scores.csv'
data = pd.read_csv(file_path)

data['doc_id'] = data['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

pivot_data = data.pivot_table(index='doc_id', columns='model_type', values=[
    'rouge_we_3_p', 'rouge_we_3_r', 'rouge_we_3_f'])

mean_langchain = pivot_data.xs('langchain', axis=1, level=1).mean()
mean_openai = pivot_data.xs('openai', axis=1, level=1).mean()

plt.figure(figsize=(12, 6))

labels = ['ROUGE WE Precision', 'ROUGE WE Recall', 'ROUGE WE F1 Score']
x = range(len(labels))

plt.bar(x, mean_langchain, width=0.4, label='LangChain', color='green', align='center')
plt.bar([p + 0.4 for p in x], mean_openai, width=0.4, label='OpenAI', color='blue', align='center')

for i, v in enumerate(mean_langchain):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
for i, v in enumerate(mean_openai):
    plt.text(i + 0.4, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

plt.xticks([p + 0.2 for p in x], labels)
plt.ylabel('Mean ROUGE WE Score')
plt.title('Comparison of ROUGE WE Scores (LangChain vs OpenAI)')
plt.legend()

plt.show()
