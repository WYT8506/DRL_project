import gc
import os
import numpy as np
import torch
from utils.opt_utils import token_gradients, sample_control, get_logits, get_logits_kv_cache, target_loss, \
    load_model_and_tokenizer, get_nonascii_toks,get_filtered_cands,get_dataset, wait_for_available_gpu_memory
from utils.string_utils import SuffixManager,  remove_bos_token
import time
import argparse
import pandas as pd
import json
from tqdm import tqdm
import random
from utils.opt_utils import check_early_stopping
from utils.utils import generate
seed = 20
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def log_init():
    log_dict = {"loss": [], "suffix": [],
                "time": [], "respond": [], "success": []}
    return log_dict


def generate_pattern(length):
    length = length*4
    pattern = ["!" for i in range(length)]
    return ''.join(pattern)


def get_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("--momentum", type=float, default=1.0)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=50)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=20)    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--use_kv", type=bool,default = True)                  

    parser.add_argument("--dataset_path", type=str, default="./data/uni_train.csv")
    parser.add_argument("--target_dataset_path", type=str, default="./data/sentiment_analysis/data.csv")
    #parser.add_argument("--dataset_path", type=str, default="./data/duplicate_sentence_detection/data.csv")
    parser.add_argument("--save_suffix", type=str, default="normal")

    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--injection", type=str, default="dynamic")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    device = f'cuda:{args.device}'
    wait_for_available_gpu_memory(required_memory_gb=70, device=args.device, check_interval=500)
    model_path_dicts = {"mistral": "mistralai/Mistral-7B-Instruct-v0.2", "llama3": "meta-llama/Meta-Llama-3-8B-Instruct", "llama2": "meta-llama/Llama-2-7b-chat-hf", "vicuna": "./models/vicuna/vicuna-7b-v1.3",
                        "guanaco": "./models/guanaco/guanaco-7B-HF", "WizardLM": "./models/WizardLM/WizardLM-7B-V1.0",
                        "mpt-chat": "./models/mpt/mpt-7b-chat", "mpt-instruct": "./models/mpt/mpt-7b-instruct",
                        "falcon": "./models/falcon/falcon-7b-instruct", "llamag": "./models/llama2/llama-guard-7b"}
    model_path = model_path_dicts[args.model]
    template_name = args.model
    suffix_string_init = generate_pattern(args.tokens)

    num_steps = args.num_steps
    batch_size = args.batch_size
    topk = args.topk
    momentum = args.momentum
    allow_non_ascii = False
    model, tokenizer = load_model_and_tokenizer(model_path,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device=device)
    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)

    infos = {}
    suffix_tokens = remove_bos_token(tokenizer,tokenizer.encode(suffix_string_init))
    best_suffix_loss = 1e10
    suffix_search_space = torch.randint(0, 30000, (1000, 128))

    kv_caches = []
    dataset = get_dataset(split="train")
    #print(f"dataset: {dataset}")
    loss_history = []
    for j in tqdm(range(num_steps)):
        start_time = time.time()
        log = log_init()
        info = {"goal": "", "target": "", "final_suffix": "",
                "final_respond": "", "total_time": 0, "is_success": False, "log": log}
        dataset = get_dataset(split="train")

        new_suffix = get_filtered_cands(tokenizer,
                                        suffix_tokens,
                                        suffix_search_space,
                                        filter_cand=False,
                                        batch_size=args.batch_size)

        losses = 0
        correct_losses = 0
        start_logits_time = time.time()
        for i, (query,positive_response,negative_response) in enumerate(dataset):
  
            suffix_manager = SuffixManager(args,tokenizer=tokenizer,
                                           query = query,
                                           positive_response = positive_response,
                                           negative_response = negative_response,
                                           suffix_ids_list=new_suffix)
           
            input_ids_positive,input_ids_negative = suffix_manager.get_input_ids_train()
            input_ids_positive = torch.tensor(input_ids_positive).to(device)
            input_ids_negative = torch.tensor(input_ids_negative).to(device)
            with torch.no_grad():
                logits_positive= get_logits(model=model,
                                        tokenizer=tokenizer,
                                        input_ids_list=input_ids_positive)
                logits_negative= get_logits(model=model,
                        tokenizer=tokenizer,
                        input_ids_list=input_ids_negative)
                losses += target_loss(logits_positive, input_ids_positive,suffix_manager._positive_response_slice,logits_negative, input_ids_negative, suffix_manager._negative_response_slice)
                correct_losses =losses#target_loss(correct_logits, input_ids, suffix_manager._target_slice)
            # Clear GPU memory after each iteration
        #print(f"losses of candidate suffixes: {losses}")

        end_logits_time = time.time()
        #print(f"logits_time: {end_logits_time-start_logits_time}")
        
        completions = []
        if losses.min()/(len(dataset)) < best_suffix_loss:
            suffix_tokens= new_suffix[losses.argmin()]
            #print(f"best_adv_suffix: {best_adv_suffix}")
            best_suffix_loss = losses.min()/(len(dataset))
            coordinate_grad = 0
            #print("dataset: ", dataset)
            for i, (query,positive_response,negative_response) in enumerate(dataset):

                suffix_manager = SuffixManager(args,tokenizer=tokenizer,
                                            query=query,
                                            positive_response=positive_response,
                                            negative_response=negative_response,
                                            suffix_ids_list=[suffix_tokens])

                input_ids_positive,input_ids_negative = suffix_manager.get_input_ids_train()
                #print(f"input_ids: {input_ids.shape}")
                input_ids_positive = torch.tensor(input_ids_positive).to(device)
                input_ids_negative = torch.tensor(input_ids_negative).to(device)
                #print(f"input_ids_decoded: {tokenizer.decode(input_ids[0])}")
                if j%5==0:
                    completion = tokenizer.decode(model.generate(torch.tensor(suffix_manager.get_prompt_ids()).to(device), 
                                      max_new_tokens=10,  
                                      temperature= 0.0001
                                      )[0])
                    completion = completion.split("[/INST]")[-1]
                    #print(f"query {i}: {query}")
                    print(f"completion {i}: {completion}")
                    completions.append(completion)

                next_grad_positive = token_gradients(model,
                                            input_ids_positive[0],
                                            suffix_manager._control_slice,
                                            suffix_manager._positive_response_slice,
                                            suffix_manager._log_p_positive_slice)
                next_grad_negative = token_gradients(model,
                            input_ids_negative[0],
                            suffix_manager._control_slice,
                            suffix_manager._negative_response_slice,
                            suffix_manager._log_p_negative_slice)
                next_grad = next_grad_positive - next_grad_negative


                coordinate_grad += next_grad
                torch.cuda.empty_cache()

            final_coordinate_grad = coordinate_grad
            print(f"final_coordinate_grad: {final_coordinate_grad.shape}")

            suffix_search_space = sample_control(suffix_tokens,
                                    final_coordinate_grad,
                                    batch_size,
                                    topk=topk,
                                    temp=1,
                                    not_allowed_tokens=not_allowed_tokens)
            dataset = get_dataset(split="train")
            best_suffix = tokenizer.decode(suffix_tokens)
            current_loss = losses.min()/(len(dataset))
        correct_loss = correct_losses.min()/(len(dataset))
        print(
            f"Current Epoch: {j}/{num_steps}, Loss [best]:{best_suffix_loss.item()}, Loss [current]:{current_loss.item()}, Loss [correct]:{correct_loss.item()}, \nCurrent Best Suffix:\n{best_suffix}\n")

        info["log"]["loss"].append(best_suffix_loss.item())
        info["log"]["suffix"].append(best_suffix)
        gc.collect()
        torch.cuda.empty_cache()
        end_time = time.time()
        cost_time = round(end_time - start_time, 2)
        info["total_time"] = cost_time
        info["final_suffix"] = best_suffix
        info["final_suffix_ids"] = suffix_tokens
        info["completions"] = completions
        infos[j] = info

        if not os.path.exists(
                f"./results/{args.model}"):
            os.makedirs(
                f"./results/{args.model}")
        with open(
                f'./results/{args.model}/{args.model}.json',
                'w') as json_file:
            json.dump(infos, json_file)
        loss_history.append(best_suffix_loss.item())
        #if len(loss_history) > 200 and loss_history[-100]-loss_history[-1]<0.001:
           # break
        if len(completions) > 0:
            print(f"completions: {completions}")
            if check_early_stopping(completions,loss_history):
                break

