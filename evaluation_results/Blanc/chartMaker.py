import pandas as pd
import matplotlib.pyplot as plt

summaries_scores = pd.read_csv('summaries3_scores.csv')
abstracts_scores = pd.read_csv('Abstracts_scores.csv')

summaries_scores['doc_id'] = summaries_scores['filename'].apply(lambda x: x.split('_')[1].split('.')[0])
abstracts_scores['doc_id'] = abstracts_scores['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

merged_scores = summaries_scores.merge(abstracts_scores[['doc_id', 'blanc_score']], on='doc_id', suffixes=('_model', '_reference'))

merged_scores['blanc_diff'] = merged_scores['blanc_score_model'] - merged_scores['blanc_score_reference']

langchain_scores = merged_scores[merged_scores['filename'].str.contains('langchain')]
openai_scores = merged_scores[merged_scores['filename'].str.contains('openai')]

mean_diff_langchain = langchain_scores['blanc_diff'].mean()
mean_diff_openai = openai_scores['blanc_diff'].mean()

plt.figure(figsize=(10, 6))
bars = plt.bar(['LangChain', 'OpenAI'], [mean_diff_langchain, mean_diff_openai], color=['green', 'blue'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')

plt.title('Mean BLANC Score Difference (Model vs Reference)')
plt.ylabel('Mean BLANC Score Difference')
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

print(f"Mean BLANC Score Difference for LangChain: {mean_diff_langchain}")
print(f"Mean BLANC Score Difference for OpenAI: {mean_diff_openai}")
