import pandas as pd
import matplotlib.pyplot as plt

# Function to load data and generate bar plot
def generate_bleu_diff_barplot(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Pivot the data for comparison
    pivot_data = data.pivot_table(index='doc_id', columns='model_type', values='bleu_score').reset_index()

    # Calculate the difference between langchain and the base model (openai)
    pivot_data['bleu_diff'] = pivot_data['langchain'] - pivot_data['openai']

    # Sort the data by the difference from largest to smallest
    sorted_heatmap_data = pivot_data[['doc_id', 'bleu_diff']].set_index('doc_id').sort_values(by='bleu_diff', ascending=False)

    # Count the number of positive and negative differences
    positive_count = (sorted_heatmap_data['bleu_diff'] > 0).sum()
    negative_count = (sorted_heatmap_data['bleu_diff'] < 0).sum()

    # Generate the bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sorted_heatmap_data)), sorted_heatmap_data['bleu_diff'], color=sorted_heatmap_data['bleu_diff'].apply(lambda x: 'red' if x > 0 else 'blue'))

    # Add a horizontal line at y=0
    plt.axhline(0, color='black', linewidth=0.8)

    # Mark the transition point from positive to negative
    for i in range(1, len(sorted_heatmap_data)):
        if sorted_heatmap_data['bleu_diff'].iloc[i] < 0 and sorted_heatmap_data['bleu_diff'].iloc[i-1] > 0:
            plt.axvline(i-0.5, color='green', linestyle='--')
            break

    # Add labels
    plt.title('BLEU Score Differences (LangChain-enhanced GPT 3.5 vs GPT 3.5)')
    plt.xlabel('Documents (sorted by BLEU difference)')
    plt.ylabel('BLEU Score Difference')
    plt.xticks([])  

    # Display counts of positive and negative differences
    plt.text(len(sorted_heatmap_data) - 1, sorted_heatmap_data['bleu_diff'].max(), f'Positive Count: {positive_count}', ha='right', color='red')
    plt.text(len(sorted_heatmap_data) - 1, sorted_heatmap_data['bleu_diff'].min(), f'Negative Count: {negative_count}', ha='right', color='blue')

    plt.show()

file_path = 'summary3_scores.csv'
generate_bleu_diff_barplot(file_path)
