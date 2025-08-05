from datasets import load_dataset
import os
from argparse import ArgumentParser
from functools import partial


SYN_TRAIN_FILE =  'syn_ret_nl.jsonl' #'syn_ret_nl.jsonl'
SYN_TASK_TYPES = {'sl':'', 'ls':'-no_in_batch_neg', 'sts':'', 'll':'', 'ss':''}
OLD_DATASETS = {
	"HotpotQA-NL": {'id': "Ehsanl/Ret-nl", 'ratio':1, 'suf':'', 'config':'hpqa'},
	"FEVER-NL": {'id':"Ehsanl/fv_nl_trip", 'ratio':1, 'suf':'' },
	"MSMARCO-NL": {'id':"Ehsanl/msm_nl_trip", 'ratio':.5, 'suf':''},
	#"NQ-NL": ("clips/beir-nl-nq",1),
	"SQuAD-NL": {'id':"Ehsanl/sq_nl_trip", 'ratio':1, 'suf':'-no_in_batch_neg'},
	"Quora-NL": {'id':"Ehsanl/qr_nl_trip", 'ratio':.3, 'suf':''}
}

CNV_DATASETS = {
	"QA3-NL": {'id': "Ehsanl/qa3_nl_trip", 'ratio':1, 'suf':''},
	"ULTRA-NL": {'id':"Ehsanl/ultrac_nl_trip", 'ratio':1, 'suf':''}
}

os.makedirs('data', exist_ok=True)


prompts={
	"HotpotQA-NL": "Given a multi-hop question in Dutch, retrieve Dutch documents that can help answer the question",
	"FEVER-NL": "Given a claim in Dutch, retrieve Dutch documents that support or refute the claim",
	"SQuAD-NL": "Given a question in Dutch, retrieve Dutch Wikipedia passages that answer the question",
	"MSMARCO-NL": "Given a web search query in Dutch, retrieve relevant Dutch passages that answer the query",
	"Quora-NL": "Given a question in Dutch, retrieve Dutch questions that are semantically equivalent to the given question"
}


def add_prompts(query, task_desc:str, data_type, dataset_name=None):
	if data_type == 'syn':
		if task_desc == '': task_desc = 'Given the following text, find semantically similar texts.'
		task_desc = task_desc.replace('Given a', 'Given the following').replace('Identifying', 'Identify') + ' (in Dutch)'
	elif data_type == 'old':
		task_desc = prompts[dataset_name]

	query = f'Instruct: {task_desc} \n Query:{query}'
	return query


def main(args):
	is_llm = any([k in args.model for k in ['Qwen', 'Mistral', 'EuroLLM']]) if args.is_llm is None else args.is_llm
	print(is_llm)

	def _transform_syn(sample):
		if is_llm: sample['query'] = add_prompts(sample['query'], sample['task_desc'], 'syn')
		sample['pos'] = [sample['pos']]
		sample['neg'] = [sample['neg']]
		return sample

	def _add_prompt(sample, dataset_name):
		sample['query'] = add_prompts(sample['query'], None, data_type='old', dataset_name=dataset_name)
		return sample
	
	print('Building the datasets ...')
	if args.use_syn_data:
		raw_dataset = load_dataset('Ehsanl/SynRet', data_files=SYN_TRAIN_FILE, token=args.token)['train'].rename_column('q', 'query')
		for task, task_suffix in SYN_TASK_TYPES.items():
			task_dataset = raw_dataset.filter(lambda x: x['task_type']== task)
			task_dataset = task_dataset.map(_transform_syn).remove_columns(['task_type', 'task_desc'])
			task_dataset.to_json(f'data/syn_{task}{task_suffix}.jsonl')
	if args.use_old_data:
		for data_name, flds in OLD_DATASETS.items():
			data_id, ratio, suffix = flds['id'], flds['ratio'], flds['suf']	
			data_dir = flds.get('config', 'data')
			dataset = load_dataset(data_id, data_dir=data_dir, split='train').shuffle()
			if ratio < 1: dataset = dataset.select(range(int(len(dataset)*ratio)))
			if is_llm: 
				tasked_prompt = partial(_add_prompt, dataset_name=data_name)
				dataset = dataset.map(tasked_prompt)
			dataset.to_json(f'data/old_{data_name}{suffix}.jsonl')
	if args.use_cnv_data:
		for data_name, flds in CNV_DATASETS.items():
			data_id, ratio, suffix = flds['id'], flds['ratio'], flds['suf']
			dataset = load_dataset(data_id)['train'].shuffle()
			if ratio < 1: dataset = dataset.select(range(int(len(dataset)*ratio)))
			dataset.to_json(f'data/cnv_{data_name}{suffix}.jsonl')




if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--model', default=None)
	parser.add_argument('--use_syn_data', default=False)
	parser.add_argument('--use_old_data', default=False)
	parser.add_argument('--use_cnv_data', default=False)
	parser.add_argument('--token', default=None)
	parser.add_argument('--is_llm', default=None)
	args = parser.parse_args()
	main(args)