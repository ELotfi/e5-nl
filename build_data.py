from datasets import load_dataset
import os


SYN_TRAIN_FILE =  'synret_50k.jsonl' #'syn_ret_nl.jsonl'
TASK_TYPES = {'sl':'', 'ls':'-no_in_batch_neg', 'sts':'', 'll':'', 'ss':''}
os.makedirs('data', exist_ok=True)


def _transform(sample):
	sample['pos'] = [sample['pos']]
	sample['neg'] = [sample['neg']]
	return sample


raw_dataset = load_dataset('Ehsanl/SynRet', data_files=SYN_TRAIN_FILE)['train'].rename_column('q', 'query')
print('Building the dataset ...')
for task, task_suffix in TASK_TYPES.items():
	task_dataset = raw_dataset.filter(lambda x: x['task_type']== task)
	task_dataset = task_dataset.map(_transform).remove_columns(['task_type', 'task_desc'])
	task_dataset.to_json(f'data/{task}{task_suffix}.jsonl')

