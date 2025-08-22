from datasets import load_dataset
import os
from argparse import ArgumentParser
from functools import partial


SYN_TRAIN_FILE =  'syn_ret_nl.jsonl' #'syn_ret_nl.jsonl'
SYN_TASKS = {'sl':{'suf':'', 'type':'retrieval'}, 
			 'ls': {'suf':'-no_in_batch_neg', 'type':'classification'}, 
			 'sts':{'suf':'', 'type':'symmetric_sts'}, 
			 'll':{'suf':'', 'type':'symmetric_clustering'}, 
			 'ss':{'suf':'', 'type':'symmetric_clustering'}
			}
OLD_DATASETS_ = {
	"HotpotQA-NL": {'id': "Ehsanl/Ret-nl", 'ratio':1, 'suf':'', 'config':'hpqa'},
	"FEVER-NL": {'id':"Ehsanl/Ret-nl", 'ratio':1, 'suf':'', 'config':'fevr' },
	"MSMARCO-NL": {'id':"Ehsanl/msm_nl_trip", 'ratio':.6, 'suf':''},
	#"NQ-NL": ("clips/beir-nl-nq",1),
	"SQuAD-NL": {'id':"Ehsanl/sq_nl_trip", 'ratio':1, 'suf':'-no_in_batch_neg'},
	"Quora-NL": {'id':"Ehsanl/qr_nl_trip", 'ratio':.3, 'suf':''}
}

OLD_DATASETS = {
	"HotpotQA-NL": {'id': "Ehsanl/RetNLMined", 'config':'hpqa', 'ratio':1, 'suf':'', 'group_size':8, 'type':'retrieval'},
	"FEVER-NL": {'id':"Ehsanl/RetNLMined", 'config':'fevr', 'ratio':1, 'suf':'' , 'group_size':8, 'type':'retrieval'},
	"MSMARCO-NL": {'id':"Ehsanl/RetNLMined",'config':'mrco', 'ratio':.6, 'suf':'', 'group_size':8, 'type':'retrieval'}
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

	def _add_prompt(sample, dataset_name):
		sample['query'] = add_prompts(sample['query'], None, data_type='old', dataset_name=dataset_name)
		return sample
	
	print('Building the datasets ...')
	if args.use_syn_data:
		group_size = 2
		raw_dataset = load_dataset('Ehsanl/SynRetRr', data_dir='rranked', token=args.token)['train'].rename_column('q', 'query')
		print(len(raw_dataset))
		if args.filter_by_dpn: raw_dataset = raw_dataset.filter(lambda x:((x['task_type']=='ls') or (x['pos_scores'][0] >= .1) and (x['pos_scores'][0] - x['neg_scores'][0] <= args.dpn_thresh)))
		print(len(raw_dataset))
		for task, task_conf in SYN_TASKS.items():
			task_suffix, task_type = task_conf['suf'], task_conf['type']
			task_dataset = raw_dataset.filter(lambda x: x['task_type']== task)

			def _transform_syn(sample):
				if is_llm: sample['query'] = add_prompts(sample['query'], sample['task_desc'], 'syn')
				sample['pos'] = [sample['pos']]
				sample['neg'] = [sample['neg']]
				sample['type'] = task_type
				return sample
			
			task_dataset = task_dataset.map(_transform_syn).remove_columns(['task_type', 'task_desc'])
			if len(task_dataset)>0: task_dataset.to_json(f'data/{group_size}_syn_{task}{task_suffix}.jsonl')

	if args.use_old_data:
		for data_name, flds in OLD_DATASETS.items():
			data_id, ratio, suffix, group_size, task_type = flds['id'], flds['ratio'], flds['suf'], flds['group_size'], flds['type']	
			data_dir = flds.get('config', 'data')
			dataset = load_dataset(data_id, data_dir=data_dir, split='train', token=args.token).shuffle()
			dataset = dataset.add_column('type',[task_type]*len(dataset))
			removed_columns = [c for c in dataset.column_names if c not in ['query', 'pos', 'neg', 'type']]
			dataset = dataset.remove_columns(removed_columns)
			if ratio < 1: dataset = dataset.select(range(int(len(dataset)*ratio)))
			if is_llm: 
				tasked_prompt = partial(_add_prompt, dataset_name=data_name)
				dataset = dataset.map(tasked_prompt)
			dataset.to_json(f'data/{group_size}_old_{data_name}{suffix}.jsonl')

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
	parser.add_argument('--filter_by_dpn', default=False)
	parser.add_argument('--dpn_thresh', default=0.98)
	args = parser.parse_args()
	main(args)