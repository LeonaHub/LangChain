from datasets import load_dataset

dataset = load_dataset('../arxiv_dataset_script.py', data_dir='../data', trust_remote_code=True)

print(dataset['train'][0])
# print(dataset['train'].info)

print(dataset['train'].features)
print(len(dataset['train']))
2515829
