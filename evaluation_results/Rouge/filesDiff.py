import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'summary3_scores.csv' 
data = pd.read_csv(file_path)

# Determine model type based on filename
data['model_type'] = data['filename'].apply(lambda x: 'langchain' if 'langchain' in x else 'openai')

# Separate the rows for LangChain and OpenAI models
langchain_rows = data[data['model_type'] == 'langchain']
openai_rows = data[data['model_type'] == 'openai']

# Reset indices for alignment
langchain_rows = langchain_rows.reset_index(drop=True)
openai_rows = openai_rows.reset_index(drop=True)

# Calculate ROUGE differences
rouge_diff = pd.DataFrame()
for metric in [
    'rouge_1_recall', 'rouge_1_precision', 'rouge_1_f_score',
    'rouge_2_recall', 'rouge_2_precision', 'rouge_2_f_score',
    'rouge_l_recall', 'rouge_l_precision', 'rouge_l_f_score'
]:
    rouge_diff[f'{metric}_diff'] = langchain_rows[metric] - openai_rows[metric]

# Count how many documents favored LangChain or OpenAI for each metric
better_langchain = (rouge_diff > 0).sum()
better_openai = (rouge_diff < 0).sum()

# Plotting the document count comparison with annotations
plt.figure(figsize=(12, 6))

# Plot LangChain bars
langchain_bars = better_langchain.plot(kind='bar', color='green', width=0.4, position=1, label='LangChain')

# Annotate LangChain bars
for i in langchain_bars.patches:
    plt.text(i.get_x() + i.get_width() / 2, i.get_height(), f'{int(i.get_height())}', 
             ha='center', va='bottom')

# Plot OpenAI bars
openai_bars = better_openai.plot(kind='bar', color='blue', width=0.4, position=0, label='OpenAI')

# Annotate OpenAI bars
for i in openai_bars.patches:
    plt.text(i.get_x() + i.get_width() / 2, i.get_height(), f'{int(i.get_height())}', 
             ha='center', va='bottom')

plt.title('Document Count Comparison (LangChain vs OpenAI)')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()
