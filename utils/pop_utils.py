from utils.string_utils import SuffixManager,query_target
import torch
from utils.opt_utils import get_logits, target_loss, get_dataset
import random
import gc
def get_loo_adv_suffixs(adv_suffix_tokens, partition_size):
    loo_adv_suffixs = []
    loo_slices = []
    for i in range(0, len(adv_suffix_tokens), partition_size):
        loo_adv_suffixs.append(adv_suffix_tokens[:i] + adv_suffix_tokens[i+partition_size:])
        loo_slices.append(slice(i, i+partition_size))
    return loo_adv_suffixs[0:len(loo_adv_suffixs)-1], loo_slices[0:len(loo_slices)-1]
def get_filtered_cands_kv_cache_pop(tokenizer: object, adv_suffix_tokens: object, control_cand: object, filter_cand: object = True,token_change_number=1,batch_size = 16, partition_size=10) -> object:
    adv_suffix_list = []
    n_tokens_change = token_change_number
    #weights = [i for i in range(1, len(adv_suffix_tokens)-token_change_number+2)]
    #substitute_pos_start = random.choices(range(0, len(adv_suffix_tokens)-token_change_number+1), weights=weights, k=1)[0]
    substitute_pos_start = random.choice(range(len(adv_suffix_tokens)-1,len(adv_suffix_tokens)-token_change_number+1))
    for i in range(batch_size):
        while True:
            substitution_tokens = [random.choice(control_cand[pos,:].tolist()) for pos in range(substitute_pos_start, min(substitute_pos_start + n_tokens_change, len(adv_suffix_tokens)))]
            if not any(token in adv_suffix_tokens[substitute_pos_start:min(substitute_pos_start + n_tokens_change, len(adv_suffix_tokens))] for token in substitution_tokens):
                break
                                            
        #print(f"substitution_tokens: {substitution_tokens}")
        #print(f"adv_suffix_tokens: {adv_suffix_tokens[substitute_pos_start:min(substitute_pos_start + n_tokens_change, len(adv_suffix_tokens))]}")
        new_adv_suffix_tokens = adv_suffix_tokens[:substitute_pos_start]+substitution_tokens+adv_suffix_tokens[substitute_pos_start+n_tokens_change:]
        #print(f"adv_substitution_tokens: {substitution_tokens}")
        adv_suffix_list.append(new_adv_suffix_tokens)

    return adv_suffix_list
def get_logits_kv_cache_pop(*, model, tokenizer, input_ids_list, kv_cache, kv_cache_ids, batch_size=512):
    def slice_kv_cache(cache, k1, k2):
        # Directly return sliced key-value pairs without creating new list
        return [(key[:, :, k1:k2, :], value[:, :, k1:k2, :]) for key, value in cache]
    new_kv_caches = []
    new_input_ids_list = []
    start_all = 0
    for j,input_ids in enumerate(input_ids_list):
        start_idx = 0
        for i in range(len(input_ids)):
            if not torch.equal(input_ids[i], kv_cache_ids[i]):  # Compare tensors properly
                start_idx = i
                break
        
        if j == 0:
            start_all = start_idx-5
        #print(f"start_idx: {start_all}")
        new_kv_caches.append(slice_kv_cache(kv_cache, 0, start_all))

        new_input_ids = torch.tensor(input_ids[start_all:]).to(model.device)
        new_input_ids_list.append(new_input_ids)

    # Convert the first dimension of the list of list of tuples to the first dimension of K and V (batch)
    if new_kv_caches:
        batch_size = len(new_kv_caches)
        num_layers = len(new_kv_caches[0])
        new_kv_caches = [
                (
                    torch.stack([new_kv_caches[batch_idx][layer_idx][0] for batch_idx in range(batch_size)], dim=0).squeeze(1),
                    torch.stack([new_kv_caches[batch_idx][layer_idx][1] for batch_idx in range(batch_size)], dim=0).squeeze(1)
                )
                for layer_idx in range(num_layers)
            ]


    #print(f"new_kv_caches: {new_kv_caches[0][0].shape}")
    
    #print(f"new_input_ids_list: {tokenizer.decode(new_input_ids_list[0])}")
    with torch.no_grad():
        outputs = model(input_ids=torch.stack(new_input_ids_list).to(model.device),past_key_values=new_kv_caches, use_cache=True)
    logits = outputs.logits
    logits = torch.cat((torch.zeros(logits.size(0), start_all, logits.size(2), device=logits.device), logits), dim=1)
    #print(f"logits: {logits.shape}")
    gc.collect()
    return logits

def get_pop_token(args, adv_suffix_tokens,harmful_data,model, tokenizer, kv_caches, kv_cache_ids,partition_size=10):
    loo_adv_suffixs, loo_slices = get_loo_adv_suffixs(adv_suffix_tokens,partition_size)
    print(f"len(loo_adv_suffixs): {[len(loo_adv_suffix) for loo_adv_suffix in loo_adv_suffixs]}")
    batch_size = 16
    loo_adv_suffixs_batches = [loo_adv_suffixs[i:i+batch_size] for i in range(0, len(loo_adv_suffixs), batch_size)]
    losses_all = []
    for loo_adv_suffixs_batch in loo_adv_suffixs_batches:
        losses = 0
        dataset = get_dataset(harmful_data, args)
        for i, (instruction, input, output, task, dataset_name) in enumerate(dataset):
            
            target = query_target(args, instruction, output)
            suffix_manager = SuffixManager(tokenizer=tokenizer,
                                        instruction=instruction,
                                        query = input,
                                        target=target,
                                        adv_ids_list=loo_adv_suffixs_batch)
            #print(f"suffix_manager.check_consistency(): {suffix_manager.check_consistency()}")
            #print(f"full prompt: {suffix_manager.get_prompt()}")
            input_ids = suffix_manager.get_input_ids_train()
            input_ids = torch.tensor(input_ids).to(model.device)
            with torch.no_grad():
                logits= get_logits_kv_cache_pop(model=model,
                                        tokenizer=tokenizer,
                                        input_ids_list=input_ids,
                                        kv_cache=kv_caches[i],
                                        kv_cache_ids=kv_cache_ids[i])

                losses += target_loss(logits, input_ids, suffix_manager._target_slice)
        losses =losses/(len(harmful_data.instruction[args.start:args.end]))
        losses_all.extend(losses.flatten().tolist())
    losses_all = torch.tensor(losses_all)

    print(f"losses_all: {losses_all}")
    removal_id = losses_all[0:losses_all.shape[0]-10].argmin()
    #print('removed loo suffixs: ', tokenizer.convert_ids_to_tokens(loo_adv_suffixs[removal_id][0]))
    return loo_adv_suffixs[removal_id] + [tokenizer.convert_tokens_to_ids("!!!!!!!!") for _ in range(partition_size)],losses_all[removal_id]