'''
Support a bunch of datasets for classification tasks.
Implement restaurant_sup and laptop_sup dataset for Aspect Based Sentiment Analysis(ABSA).
Implement acl_sup dataset for citation intent classification.
Implement agnews_sup dataset.
Implement the few shot version of the above tasks: named restaurant_fs, laptop_fs, acl_fs, agnews_fs.
Implement the aggregation of dataset.
'''

import datasets
import json
import pandas as pd

def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    dataset = None

    # your code for preparing the dataset...
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]
    
    all_datasets = []
    label_offset = 0
    for name in dataset_name:
        # Implement restaurant_sup and laptop_sup dataset for Aspect Based Sentiment Analysis(ABSA).
        if name.startswith('restaurant') or name.startswith('laptop'):
            label_dict = {'positive': 1, 'negative': 0, 'neutral': 2}
            if name.startswith('restaurant'):
                with open('SemEval14-res/train.json', 'r') as f:
                    train_data = json.load(f)
                    train_data_list = [{'text':value['term'] + sep_token + value['sentence'], 'label':label_dict[value['polarity']]} for value in train_data.values()]
                with open('SemEval14-res/test.json', 'r') as f:
                    test_data = json.load(f)
                    test_data_list = [{'text':value['term'] + sep_token + value['sentence'], 'label':label_dict[value['polarity']]} for value in test_data.values()]
            elif name.startswith('laptop'):
                with open('SemEval14-laptop/train.json', 'r') as f:
                    train_data = json.load(f)
                    train_data_list = [{'text':value['term'] + sep_token + value['sentence'], 'label':label_dict[value['polarity']]} for value in train_data.values()]
                with open('SemEval14-lap/test.json', 'r') as f:
                    test_data = json.load(f)
                    test_data_list = [{'text':value['term'] + sep_token + value['sentence'], 'label':label_dict[value['polarity']]} for value in test_data.values()]
            
            train_dataset = datasets.Dataset.from_dict(train_data_list)
            test_dataset = datasets.Dataset.from_dict(test_data_list)
            dataset = datasets.DatasetDict({'train':train_dataset, 'test':test_dataset})


        # Implement acl_sup dataset for citation intent classification.
        elif name.startswith('acl'):
            label_list = {'background': 0, 'method': 1, 'result': 2}
            with open('ACL-ARC/train.jsonl', 'r', encoding='utf-8') as f:
                train_data = [json.loads(line) for line in f]
                train_data_list = [{'text':str(value['sectionName']) + sep_token + str(value['string']), 'label':label_list[value['label']]} for value in train_data]
            with open('ACL-ARC/test.jsonl', 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f]
                test_data_list = [{'text':str(value['sectionName']) + sep_token + str(value['string']), 'label':label_list[value['label']]} for value in test_data]

            train_dataset_df = pd.DataFrame(train_data_list)
            test_dataset_df = pd.DataFrame(test_data_list)
            train_dataset = datasets.Dataset.from_pandas(train_dataset_df)
            test_dataset = datasets.Dataset.from_pandas(test_dataset_df)

            dataset = datasets.DatasetDict({'train':train_dataset, 'test':test_dataset})
            

        # Implement agnews_sup dataset.
        elif name.startswith('agnews'):
            dataset = datasets.load_dataset('ag_news', split='test')
            dataset = dataset.train_test_split(test_size=0.1, seed=2022)
            dataset = dataset.rename_column('description', 'text')

        if name.endswith('_fs'):
            dataset['train'] = dataset['train'].select(range(128))
            dataset['test'] = dataset['test'].select(range(128))

        dataset = dataset.map(lambda example: {'label': example['label'] + label_offset})
        all_datasets.append(dataset)
        label_offset += len(set(dataset['train']['label']))

    train_datasets = [d['train'] for d in all_datasets]
    test_datasets = [d['test'] for d in all_datasets]
    train_dataset = datasets.concatenate_datasets(train_datasets)
    test_dataset = datasets.concatenate_datasets(test_datasets)
    dataset = datasets.DatasetDict({'train':train_dataset, 'test':test_dataset})


    return dataset
