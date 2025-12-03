"""Code for finetune_huatuo"""

import os
import copy
import json
import torch
import logging
import argparse
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
import sys
import ast

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import set_seed
import datasets
import shutil
import json
import random
from scorer import score_mix_llava
from cli import HuatuoChatbot


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelWithLMHead
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.dataset = {}
        with open(data_path) as f:
            self.dataset = json.load(f)
        self.datas = []

        test_id = 1
        for da in self.dataset:
            da['query'] = self.llava_prompt(da)
            da['test_id'] = test_id
            test_id += 1
            self.datas.append(da)

    def mmmu_prompt(self,da):
        self.query_prompt = "Please answer the multiple-choice questions below.\n{}\n{}"
        opt_title = ['A','B','C','D','E','F','G']
        if isinstance(da['options'],str):
            da['options'] = ast.literal_eval(da['options'])
        option_str = '\n'.join(f'{opt}. {ops}' for opt,ops in zip(opt_title,da['options']))
        return self.query_prompt.format(da['question'],option_str)
    
    def OmniMedVQA_prompt(self,entity):
        q_str = entity['question'] + f'Here are {len(entity["options"])} candidate answers:' + str(entity["options"])+' Only return what you think is the correct answer from the candidate answers, do not return any other irrelevant text!'
        return q_str

    def llava_prompt(self,da):
        self.query_prompt = "{}\n{}\nAnswer with the option's letter from the given choices directly."
        opt_title = ['A','B','C','D','E','F','G','H','I','J','K']
        if isinstance(da['options'],str):
            da['options'] = ast.literal_eval(da['options'])
        option_str = '\n'.join(f'{opt}. {ops}' for opt,ops in zip(opt_title,da['options']))
        return self.query_prompt.format(da['question'],option_str)

    def __getitem__(self, index):
        da = self.datas[index]
        return {
            'data': da,
            'query': da['query'],
            'image': da['image']
        }
    
    def __len__(self):
        return len(self.datas)

    def collate_fn(self, batch):
        out_batch = {}
        out_batch['query'] = [x['query'] for x in batch]
        out_batch['data'] = [x['data'] for x in batch]
        out_batch['image'] = [x['image'] for x in batch]
        return out_batch


def get_response(inputs,outputs,tokenizer,num_return):
    responses_list=[]
    batch_return=[]
    for i, output in enumerate(outputs):
        input_len = len(inputs[0])
        generated_output = output[input_len:]
        batch_return.append(tokenizer.decode(generated_output, skip_special_tokens=True))
        if i%num_return==num_return-1:
            responses_list.append(batch_return)
            batch_return=[]
    return responses_list

def table_to_csv_string(table):
    rows = [",".join(table.columns)]  
    for row in table.data:
        rows.append(",".join(map(str, row)))
    return "\n".join(rows)

def test(args):
    accelerator = Accelerator()
    torch.cuda.set_device(accelerator.process_index)
    accelerator.print(f'args:\n{args}')

    bot = HuatuoChatbot(args.model_path)
    accelerator.print(f'load_finish')

    bot.gen_kwargs['max_new_tokens'] = args.max_new_tokens

    dataset = TestDataset(args, args.data_path)

    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)
    
    val_dataloader = accelerator.prepare(val_dataloader)
    accelerator.wait_for_everyone()
    cache_data = []

    with torch.no_grad():
        ress = []

        dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader

        for batch in dataloader_iterator:
            for da,query,image in zip(batch["data"],batch['query'],batch['image']):
                response = bot.inference(query,[os.path.join(os.path.dirname(args.data_path),x) for x in image])
                for img in [os.path.join(os.path.dirname(args.data_path),x) for x in image]:
                    assert os.path.exists(img),f'{img} not exists'
                da['model_output'] = response[0]
                cache_data.append(da)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        all_data =  [None] * dist.get_world_size()
        all_response =  [None] * dist.get_world_size()

        dist.all_gather_object(all_data,cache_data)

        all_data = [item for sublist in all_data for item in sublist]

        for d in all_data:
            ress.append(d)

        if accelerator.is_main_process:
            task_name =  os.path.basename(args.model_path) + f'_{os.path.split(args.data_path)[-1].replace(".json","")}'
            out_file = f'./{task_name}.json'
            with open(out_file,'w') as fw:
                json.dump(ress,fw,ensure_ascii=False,indent=2)
            print(f'test results: {out_file}')
            print(f'question num: {len(ress)}')
            val_res = score_mix_llava(ress)
            outstr = json.dumps(val_res,ensure_ascii=False,indent = 2)
            accelerator.print(outstr)
            with open(f'{task_name}_result.json','w') as fw:
                json.dump(val_res,fw,ensure_ascii=False,indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    # Model Args
    parser.add_argument('--data_path', default='medical_multimodel_evaluation_data.json', type=str)
    parser.add_argument('--model_path', default='HuatuoGPT-Vision-7B', type=str)
    parser.add_argument('--max_new_tokens', default=256, type=int)
    parser.add_argument('--batch_size', default=2, type=int)

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    set_seed(args.seed)
    test(args)           
