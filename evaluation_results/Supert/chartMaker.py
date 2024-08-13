import pandas as pd
import matplotlib.pyplot as plt

summaries_scores = pd.read_csv('summaries3_scores.csv')
abstracts_scores = pd.read_csv('Abstracts_scores.csv')

summaries_scores['doc_id'] = summaries_scores['filename'].apply(lambda x: x.split('_')[1].split('.')[0])
abstracts_scores['doc_id'] = abstracts_scores['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

merged_scores = summaries_scores.merge(abstracts_scores[['doc_id', 'supert_score']], on='doc_id', suffixes=('_model', '_reference'))

merged_scores['supert_diff'] = merged_scores['supert_score_model'] - merged_scores['supert_score_reference']

langchain_scores = merged_scores[merged_scores['model_type'] == 'langchain']
openai_scores = merged_scores[merged_scores['model_type'] == 'openai']

# Calculate mean differences
mean_diff_langchain = langchain_scores['supert_diff'].mean()
mean_diff_openai = openai_scores['supert_diff'].mean()

# Plotting the mean differences
plt.figure(figsize=(10, 6))
bars = plt.bar(['LangChain', 'OpenAI'], [mean_diff_langchain, mean_diff_openai], color=['green', 'blue'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')

plt.title('Mean SUPERT Score Difference (Model vs Reference)')
plt.ylabel('Mean SUPERT Score Difference')
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

print(f"Mean SUPERT Score Difference for LangChain: {mean_diff_langchain}")
print(f"Mean SUPERT Score Difference for OpenAI: {mean_diff_openai}")
