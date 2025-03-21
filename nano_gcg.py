import gc
import os
import numpy as np
import torch
from utils.opt_utils import token_gradients, sample_control_kv_cache, get_logits, get_logits_kv_cache, target_loss, get_filtered_cands_kv_cache, \
    load_model_and_tokenizer, get_nonascii_toks,get_filtered_cands_kv_cache,get_filtered_cands_random_search,get_dataset, initialize_kv_cache,wait_for_available_gpu_memory
from utils.string_utils import SuffixManager, load_conversation_template, query_target, modify_sys_prompts, remove_bos_token
import time
import argparse
import pandas as pd
import json
from tqdm import tqdm
import random
from utils.opt_utils import check_early_stopping

seed = 20
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 64
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    print(f"input_ids_decoded: {tokenizer.decode(input_ids[0])}")
    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks,
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id,
                                top_p=0.9,
                                do_sample=True,
                                temperature=0.7
                                )[0]
    return output_ids[assistant_role_slice.stop:]


def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
    gen_str = tokenizer.decode(generate(model,
                                        tokenizer,
                                        input_ids,
                                        assistant_role_slice,
                                        gen_config=gen_config)).strip()
    uppercased_test_prefixes = [word.upper() for word in test_prefixes]
    if len(gen_str) == 0:
        jailbroken = False
    elif gen_str.isupper():
        jailbroken = False
    else:
        jailbroken = not any([prefix in gen_str for prefix in test_prefixes + uppercased_test_prefixes])
    return jailbroken, gen_str


def log_init():
    log_dict = {"loss": [], "suffix": [],
                "time": [], "respond": [], "success": []}
    return log_dict


def generate_pattern(length):
    length = length*8
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
    parser.add_argument("--num_steps", type=int, default=30000)
    parser.add_argument("--use_kv", type=bool,default = True)

    parser.add_argument("--dataset_path", type=str, default="./data/uni_train.csv")
    parser.add_argument("--target_dataset_path", type=str, default="./data/sentiment_analysis/data.csv")
    #parser.add_argument("--dataset_path", type=str, default="./data/duplicate_sentence_detection/data.csv")
    parser.add_argument("--save_suffix", type=str, default="normal")

    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--injection", type=str, default="dynamic")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    device = f'cuda:{args.device}'
    wait_for_available_gpu_memory(required_memory_gb=70, device=args.device, check_interval=500)
    model_path_dicts = {"llama2": "./models/llama2/llama-2-7b-chat-hf", "vicuna": "./models/vicuna/vicuna-7b-v1.3",
                        "guanaco": "./models/guanaco/guanaco-7B-HF", "WizardLM": "./models/WizardLM/WizardLM-7B-V1.0",
                        "mpt-chat": "./models/mpt/mpt-7b-chat", "mpt-instruct": "./models/mpt/mpt-7b-instruct",
                        "falcon": "./models/falcon/falcon-7b-instruct","llama3": "meta-llama/Meta-Llama-3-8B-Instruct"}
    model_path = model_path_dicts[args.model]
    template_name = args.model
    adv_string_init = generate_pattern(args.tokens)

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

    harmful_data = pd.read_csv(args.dataset_path)
    target_data = pd.read_csv(args.target_dataset_path)
    infos = {}
    adv_suffix_tokens = remove_bos_token(tokenizer,tokenizer.encode(adv_string_init))
    best_adv_suffix_loss = 1e10
    adv_suffix_search_space = torch.randint(0, 30000, (1000, 128))

    kv_caches = []
    dataset = get_dataset(harmful_data, target_data, args)
    if args.use_kv:
        kv_caches,kv_cache_ids = initialize_kv_cache(args,dataset,model,tokenizer,[adv_suffix_tokens],device)
    # for kv_cache in kv_caches:
    #    print(f"kv_cache: {kv_cache[0][0].shape}")
    loss_history = []
    for j in tqdm(range(num_steps)):
        start_time = time.time()
        log = log_init()
        info = {"goal": "", "target": "", "final_suffix": "",
                "final_respond": "", "total_time": 0, "is_success": False, "log": log}
        dataset = get_dataset(harmful_data,target_data, args)
        if args.use_kv:
            new_adv_suffix = get_filtered_cands_kv_cache(tokenizer,
                                            adv_suffix_tokens,
                                            adv_suffix_search_space,
                                            filter_cand=False,
                                           batch_size=args.batch_size)
        else:
            new_adv_suffix = get_filtered_cands_random_search(tokenizer,
                                            adv_suffix_tokens,
                                            filter_cand=False,
                                           batch_size=1)

        dataset = get_dataset(harmful_data,target_data, args)
        losses = 0
        correct_losses = 0
        start_logits_time = time.time()
        for i, (instruction, input, output, task, dataset_name,target_instruction, target_input, target_output, target_task, target_dataset_name) in enumerate(dataset):
            target = query_target(args, instruction, output,target_output)
            suffix_manager = SuffixManager(tokenizer=tokenizer,
                                           instruction=instruction,
                                           query = input,
                                           target=target,
                                           target_instruction=target_instruction,
                                           target_query = target_input,
                                           adv_ids_list=new_adv_suffix,attack_type=args.injection)
            #print(f"suffix_manager.check_consistency(): {suffix_manager.check_consistency()}")
            #print(f"full prompt: {suffix_manager.get_prompt()}")
            input_ids = suffix_manager.get_input_ids_train()
            input_ids = torch.tensor(input_ids).to(device)
            with torch.no_grad():
                if args.use_kv:
                    logits= get_logits_kv_cache(model=model,
                                            tokenizer=tokenizer,
                                            input_ids_list=input_ids,
                                            kv_cache=kv_caches[i],
                                            kv_cache_ids=kv_cache_ids[i])
                else:
                    logits= get_logits(model=model,
                                    tokenizer=tokenizer,
                                   input_ids=input_ids)
                #correct_logits = get_logits(model=model,
                #                    tokenizer=tokenizer,
                #                    input_ids=input_ids)
                losses += target_loss(logits, input_ids, suffix_manager._target_slice)
                correct_losses =losses#target_loss(correct_logits, input_ids, suffix_manager._target_slice)
            # Clear GPU memory after each iteration
            

        end_logits_time = time.time()
        print(f"logits_time: {end_logits_time-start_logits_time}")
        
        completions = []
        targets = []
        if losses.min()/(len(harmful_data.instruction[args.start:args.end])) < best_adv_suffix_loss:
            adv_suffix_tokens= new_adv_suffix[losses.argmin()]
            #print(f"best_adv_suffix: {best_adv_suffix}")
            best_adv_suffix_loss = losses.min()/(len(harmful_data.instruction[args.start:args.end]))
            coordinate_grad = 0
            dataset = get_dataset(harmful_data,target_data, args)
            #print("dataset: ", dataset)
            for i, (instruction, input, output, task, dataset_name,target_instruction, target_input, target_output, target_task, target_dataset_name) in enumerate(dataset):
                target = query_target(args, instruction, output,target_output)
                targets.append(target)
                suffix_manager = SuffixManager(tokenizer=tokenizer,
                                            instruction=instruction,
                                            query=input,
                                            target=target,
                                            target_instruction=target_instruction,
                                            target_query=target_input,
                                            adv_ids_list=[adv_suffix_tokens],attack_type=args.injection)
                #if i == 0:
                    #print(f"prompt: {suffix_manager.get_prompt()}")

                input_ids = suffix_manager.get_input_ids_train()
                #print(f"input_ids: {input_ids.shape}")
                input_ids = torch.tensor(input_ids).to(device)
                #print(f"input_ids_decoded: {tokenizer.decode(input_ids[0])}")
                if best_adv_suffix_loss.item()<1:
                    completion = tokenizer.decode(model.generate(torch.tensor(suffix_manager.get_prompt_ids()).to(device), 
                                      max_new_tokens=10,  
                                      temperature= 0.0001
                                      )[0])
                    completion = completion.split("assistant")[-1]
                    print(f"completion {i}: {completion}")
                    completions.append(completion)
                #print(f"suffix_manager._control_slice: {suffix_manager._control_slice}")
                #print(f"input_ids: {input_ids.shape}")
                if args.use_kv:
                    next_grad = token_gradients(model,
                                                input_ids[0],
                                                suffix_manager._control_slice,
                                                suffix_manager._target_slice,
                                                suffix_manager._loss_slice)
                    #print(f"control_slice: {suffix_manager._control_slice}")
                    #print(f"target_slice: {suffix_manager._target_slice}")
                    #print(f"loss_slice: {suffix_manager._loss_slice}")
                    #print(f"next_grad: {next_grad[:,0:5]}")
                    coordinate_grad += next_grad
                    #print(f"coordinate_grad: {coordinate_grad[:,0:5]}")
                    # Clear GPU memory after each iteration
                torch.cuda.empty_cache()

            final_coordinate_grad = coordinate_grad
            #print(f"final_coordinate_grad: {final_coordinate_grad[:,0:10]}")
            if args.use_kv:
                adv_suffix_search_space = sample_control_kv_cache(adv_suffix_tokens,
                                        final_coordinate_grad,
                                        batch_size,
                                        topk=topk,
                                        temp=1,
                                        not_allowed_tokens=not_allowed_tokens)
                dataset = get_dataset(harmful_data, target_data, args)
                kv_caches,kv_cache_ids = initialize_kv_cache(args,dataset,model,tokenizer,[adv_suffix_tokens],device)
            best_adv_suffix = tokenizer.decode(adv_suffix_tokens)
            current_loss = losses.min()/(len(harmful_data.instruction[args.start:args.end]))
        correct_loss = correct_losses.min()/(len(harmful_data.instruction[args.start:args.end]))
        print(
            f"Current Epoch: {j}/{num_steps}, Loss [best]:{best_adv_suffix_loss.item()}, Loss [current]:{current_loss.item()}, Loss [correct]:{correct_loss.item()}, Current Target: {target}\nCurrent Best Suffix:\n{best_adv_suffix}\n")

        info["log"]["loss"].append(best_adv_suffix_loss.item())
        info["log"]["suffix"].append(best_adv_suffix)
        gc.collect()
        torch.cuda.empty_cache()
        end_time = time.time()
        cost_time = round(end_time - start_time, 2)
        info["total_time"] = cost_time
        info["final_suffix"] = best_adv_suffix
        info["final_suffix_ids"] = adv_suffix_tokens
        info["target"] = target
        info["completions"] = completions
        infos[j] = info

        if not os.path.exists(
                f"./results/eval_kv_cache/{args.model}/{args.injection}/momentum_{args.momentum}/token_length_{args.tokens}/target_{args.target}"):
            os.makedirs(
                f"./results/eval_kv_cache/{args.model}/{args.injection}/momentum_{args.momentum}/token_length_{args.tokens}/target_{args.target}")
        with open(
                f'./results/eval_kv_cache/{args.model}/{args.injection}/momentum_{args.momentum}/token_length_{args.tokens}/target_{args.target}/{args.start}_{args.end}_{seed}_{args.save_suffix}_{args.use_kv}.json',
                'w') as json_file:
            json.dump(infos, json_file)
        loss_history.append(best_adv_suffix_loss.item())
        #if len(loss_history) > 200 and loss_history[-100]-loss_history[-1]<0.001:
           # break
        if len(completions) > 0:
            print(f"completions: {completions}")
            if check_early_stopping(completions,targets,loss_history):
                break

