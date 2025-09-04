from datasets import load_dataset
import os
from argparse import ArgumentParser
from functools import partial


#SYN_TRAIN_FILE =  'syn_ret_nl.jsonl' #'syn_ret_nl.jsonl'
SYN_DATASETS = {'sl':{'id':'Ehsanl/SynEmbMinedkd', 'config':'sl', 'group_size':8, 'suf':'', 'type':'retrieval', 'ratio':1, 'order':7}, 
			 'ls': {'id':'Ehsanl/SynEmbMinedkd', 'config':'ls', 'group_size':2, 'suf':'-no_in_batch_neg', 'type':'classification', 'ratio':1, 'order':0}, 
			 'sts':{'id':'Ehsanl/SynEmbMinedkd', 'config':'sts', 'group_size':8, 'suf':'', 'type':'symmetric_sts', 'ratio':1, 'order':2}, 
			 'll':{'id':'Ehsanl/SynEmbMinedkd', 'config':'ll', 'group_size':8,'suf':'', 'type':'symmetric_clustering', 'ratio':1, 'order':4}, 
			 'ss':{'id':'Ehsanl/SynEmbMinedkd', 'config':'ss', 'group_size':8,'suf':'', 'type':'symmetric_clustering', 'ratio':1, 'order':5}
			}

OLD_DATASETS = {
	"HotpotQA-NL": {'id': "Ehsanl/RetNLMinedkd", 'config':'hpqa', 'ratio':1, 'suf':'', 'group_size':8, 'type':'retrieval', 'order':1},
	"FEVER-NL": {'id':"Ehsanl/RetNLMinedkd", 'config':'fevr', 'ratio':1, 'suf':'' , 'group_size':8, 'type':'retrieval', 'order':6},
	"MSMARCO-NL": {'id':"Ehsanl/RetNLMinedkd",'config':'mrco', 'ratio':.75, 'suf':'', 'group_size':8, 'type':'retrieval', 'order':3}
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
	print(is_llm, args.filter_by_dpn)

	def _add_prompt(sample, dataset_name):
		sample['query'] = add_prompts(sample['query'], None, data_type='old', dataset_name=dataset_name)
		return sample
	
	print('Building the datasets ...')

	if args.use_syn_data:
		for data_name, flds in SYN_DATASETS.items():
			data_id, ratio, suffix, order, group_size, task_type = flds['id'], flds['ratio'], flds['suf'], flds['order'], flds['group_size'], flds['type']	
			dataset = load_dataset(data_id, data_dir=data_name, split='train', token=args.token).shuffle()
			if args.filter_by_dpn:
				print('Filtering by dpn ...') 
				dataset = dataset.filter(lambda x:((x['task_type']=='ls') or (x['rr_pos_score'][0] >= .1) and (x['rr_pos_score'][0] - x['rr_neg_score'][0] <= args.dpn_thresh)))
			
			def _transform_syn(sample):
				if is_llm: sample['query'] = add_prompts(sample['query'], sample['task_desc'], 'syn')
				sample['pos'] = [sample['pos']]
				sample['type'] = task_type
				return sample
			
			dataset = dataset.map(_transform_syn)
			removed_columns = [c for c in dataset.column_names if c not in ['query', 'pos', 'neg', 'type', 'pos_scores', 'neg_scores']]
			dataset = dataset.remove_columns(removed_columns)
			if ratio < 1: dataset = dataset.select(range(int(len(dataset)*ratio)))
			if len(dataset)>0: dataset.to_json(f'data/{group_size}_{order}_syn_{data_name}{suffix}.jsonl')


	if args.use_old_data:
		for data_name, flds in OLD_DATASETS.items():
			data_id, ratio, suffix, order, group_size, task_type = flds['id'], flds['ratio'], flds['suf'], flds['order'], flds['group_size'], flds['type']	
			data_dir = flds.get('config', 'data')
			dataset = load_dataset(data_id, data_dir=data_dir, split='train', token=args.token).shuffle()
			dataset = dataset.add_column('type',[task_type]*len(dataset))
			removed_columns = [c for c in dataset.column_names if c not in ['query', 'pos', 'neg', 'type', 'pos_scores', 'neg_scores']]
			dataset = dataset.remove_columns(removed_columns)
			if ratio < 1: dataset = dataset.select(range(int(len(dataset)*ratio)))
			if is_llm: 
				tasked_prompt = partial(_add_prompt, dataset_name=data_name)
				dataset = dataset.map(tasked_prompt)
			dataset.to_json(f'data/{group_size}_{order}_old_{data_name}{suffix}.jsonl')

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
	parser.add_argument('--dpn_thresh', default=0.96)
	args = parser.parse_args()
	main(args)