from datasets import load_dataset
import os
from argparse import ArgumentParser


SYN_TRAIN_FILE =  'syn_ret_nl.jsonl' #'syn_ret_nl.jsonl'
SYN_TASK_TYPES = {'sl':'', 'ls':'-no_in_batch_neg', 'sts':'', 'll':'', 'ss':''}
OLD_DATASETS = {
	"HotpotQA-NL": {'id': "Ehsanl/hpqa_nl_trip", 'ratio':1, 'suf':''},
	"FEVER-NL": {'id':"Ehsanl/fv_nl_trip", 'ratio':1, 'suf':''},
	"MSMARCO-NL": {'id':"Ehsanl/msm_nl_trip", 'ratio':.6, 'suf':''},
	#"NQ-NL": ("clips/beir-nl-nq",1),
	"SQuAD-NL": {'id':"Ehsanl/sq_nl_trip", 'ratio':1, 'suf':'-no_in_batch_neg'},
	"Quora-NL": {'id':"Ehsanl/qr_nl_trip", 'ratio':.3, 'suf':''}
}

CNV_DATASETS = {
	"CHAT-NL": {'id': "BramVanroy/dutch_chat_datasets", 'ratio':1, 'suf':''},
	"HERMES-NL": {'id':"BramVanroy/Openhermes-2.5-dutch-46k-format", 'ratio':1, 'suf':''},
	"ULTRA-NL": {'id':"BramVanroy/ultrachat_200k_dutch", 'ratio':1, 'suf':''}
}

os.makedirs('data', exist_ok=True)


def _transform(sample):
	sample['pos'] = [sample['pos']]
	sample['neg'] = [sample['neg']]
	return sample


def process_dataset_old(data_conf):
	data_id, ratio, suffix = data_conf





def main(args):
	print('Building the datasets ...')
	if args.use_syn_data:
		raw_dataset = load_dataset('Ehsanl/SynRet', data_files=SYN_TRAIN_FILE)['train'].rename_column('q', 'query')
		for task, task_suffix in SYN_TASK_TYPES.items():
			task_dataset = raw_dataset.filter(lambda x: x['task_type']== task)
			task_dataset = task_dataset.map(_transform, ).remove_columns(['task_type', 'task_desc'])
			task_dataset.to_json(f'data/syn_{task}{task_suffix}.jsonl')
	if args.use_old_data:
		for data_name, (data_id, ratio, suffix) in OLD_DATASETS.items():		
			dataset = load_dataset(data_id)['train'].shuffle()
			processed_dataset = dataset.select(range(int(len(dataset)*ratio)))
			processed_dataset.to_json(f'data/old_{data_name}{suffix}.jsonl')
	if args.use_cnv_data:
		for data_name, (data_id, ratio, suffix) in OLD_DATASETS.items():
			pass





if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--use_syn_data', default=True)
	parser.add_argument('--use_old_data', default=True)
	parser.add_argument('--use_cnv_data', default=True)
	args = parser.parse_args()
	main(args)