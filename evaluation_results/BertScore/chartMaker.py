import pandas as pd
import matplotlib.pyplot as plt

# Load the provided CSV file
file_path = 'summary3_scores.csv'

data = pd.read_csv(file_path)

# Extract document ID from filename for easier comparison
data['doc_id'] = data['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

# Pivot the data for comparison between langchain and openai models
pivot_data = data.pivot_table(index='doc_id', columns='model_type', values=[
    'bert_score_precision', 'bert_score_recall', 'bert_score_f1'])

# Calculate the difference between langchain and openai for each metric
for metric in ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']:
    pivot_data[f'{metric}_diff'] = pivot_data[(metric, 'langchain')] - pivot_data[(metric, 'openai')]

# Generate plots to compare the performance

# Plotting the average difference for each metric
plt.figure(figsize=(12, 6))
mean_diffs = pivot_data[[f'{metric}_diff' for metric in ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']]].mean()
mean_diffs.plot(kind='bar', color=['green' if x > 0 else 'red' for x in mean_diffs])
plt.title('Mean Differences in BERT Scores (LangChain-enhanced GPT 3.5 vs OpenAI GPT 3.5)')
plt.ylabel('Mean Difference')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45, ha='right')
plt.show()

# Count how many documents favored LangChain or OpenAI for each metric
better_langchain = (pivot_data[[f'{metric}_diff' for metric in ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']]] > 0).sum()
better_openai = (pivot_data[[f'{metric}_diff' for metric in ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']]] < 0).sum()

# Plotting the document count comparison
plt.figure(figsize=(12, 6))
better_langchain.plot(kind='bar', color='green', width=0.4, position=1, label='LangChain')
better_openai.plot(kind='bar', color='blue', width=0.4, position=0, label='OpenAI')
plt.title('Document Count Comparison (LangChain-enhanced GPT 3.5 vs OpenAI GPT 3.5)')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()
