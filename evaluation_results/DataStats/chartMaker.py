import pandas as pd
import matplotlib.pyplot as plt

# Load the provided CSV file
file_path = 'summary3_scores.csv'  
data = pd.read_csv(file_path)

# Extract document ID from filename for easier comparison
data['doc_id'] = data['filename'].apply(lambda x: x.split('_')[1].split('.')[0])

# Pivot the data for comparison between langchain and openai models
pivot_data = data.pivot_table(index='doc_id', columns='model_type', values=[
    'coverage', 'density', 'compression', 'percentage_novel_1-gram', 'percentage_novel_2-gram'])

# Calculate the difference between langchain and openai for each metric
for metric in ['coverage', 'density', 'compression', 'percentage_novel_1-gram', 'percentage_novel_2-gram']:
    pivot_data[f'{metric}_diff'] = pivot_data[(metric, 'langchain')] - pivot_data[(metric, 'openai')]

# Generate plots to compare the performance

# Plotting the average difference for each metric
plt.figure(figsize=(12, 6))
mean_diffs = pivot_data[[f'{metric}_diff' for metric in ['coverage', 'density', 'compression', 'percentage_novel_1-gram', 'percentage_novel_2-gram']]].mean()
bars = mean_diffs.plot(kind='bar', color=['green' if x > 0 else 'red' for x in mean_diffs])
plt.title('Mean Differences in DataStats (LangChain-enhanced GPT 3.5 vs OpenAI GPT 3.5)')
plt.ylabel('Mean Difference')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45, ha='right')

# Annotate each bar with the data value
for bar in bars.patches:
    height = bar.get_height()
    bars.annotate(f'{height:.2f}',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom')

plt.show()

# Count how many documents favored LangChain or OpenAI for each metric
better_langchain = (pivot_data[[f'{metric}_diff' for metric in ['coverage', 'density', 'compression', 'percentage_novel_1-gram', 'percentage_novel_2-gram']]] > 0).sum()
better_openai = (pivot_data[[f'{metric}_diff' for metric in ['coverage', 'density', 'compression', 'percentage_novel_1-gram', 'percentage_novel_2-gram']]] < 0).sum()

# Plotting the document count comparison
plt.figure(figsize=(12, 6))
bars_langchain = better_langchain.plot(kind='bar', color='green', width=0.4, position=1, label='LangChain')
bars_openai = better_openai.plot(kind='bar', color='blue', width=0.4, position=0, label='OpenAI')
plt.title('Document Count Comparison (LangChain-enhanced GPT 3.5 vs OpenAI GPT 3.5)')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45, ha='right')
plt.legend()

# Annotate each bar with the data value
for bar in bars_langchain.patches + bars_openai.patches:
    height = bar.get_height()
    bars_langchain.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

plt.show()
