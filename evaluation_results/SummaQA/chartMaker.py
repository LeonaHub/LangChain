import pandas as pd
import matplotlib.pyplot as plt

file_path = 'summaries3_scores.csv'  
data = pd.read_csv(file_path)


data['doc_id'] = data['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

pivot_data = data.pivot_table(index='doc_id', columns='model_type', values=[
    'summaqa_avg_prob', 'summaqa_avg_fscore'])

mean_scores = pivot_data.mean()

plt.figure(figsize=(12, 6))
bars_avg_prob = plt.bar(['LangChain - Avg Prob', 'OpenAI - Avg Prob'], mean_scores['summaqa_avg_prob'], color=['green', 'blue'])
bars_avg_fscore = plt.bar(['LangChain - Avg FScore', 'OpenAI - Avg FScore'], mean_scores['summaqa_avg_fscore'], color=['green', 'blue'])

plt.title('Comparison of Mean SummaQA Scores (LangChain vs OpenAI)')
plt.ylabel('Mean SummaQA Score')
plt.xticks(rotation=45, ha='right')

for bar in bars_avg_prob + bars_avg_fscore:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}',
             ha='center', va='bottom')

plt.show()
