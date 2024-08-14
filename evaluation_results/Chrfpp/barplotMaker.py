import pandas as pd
import matplotlib.pyplot as plt

file_path = 'summaries3_scores.csv'  
data = pd.read_csv(file_path)

data['doc_id'] = data['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

pivot_data = data.pivot_table(index='doc_id', columns='model_type', values='chrf_score')

plt.figure(figsize=(12, 6))

bars = pivot_data.mean().plot(kind='bar', color=['green', 'blue'])

plt.title('Comparison of Mean CHRF Scores (LangChain vs OpenAI)')
plt.ylabel('Mean CHRF Score')
plt.xlabel('Model Type')

for bar in bars.patches:
    height = bar.get_height()
    bars.annotate(f'{height:.4f}',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3),  
                  textcoords="offset points",
                  ha='center', va='bottom')

plt.show()
