import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)
import random
from pynvml import *
import time
from utils.string_utils import SuffixManager
def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        return model.model.embed_tokens.weight

def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        return model.model.embed_tokens(input_ids)

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice, imitate_target=None):
    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits

    targets = input_ids[target_slice]
    compare = logits[0, loss_slice, :]
    loss = nn.CrossEntropyLoss()(compare, targets)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad

def get_imitate(model, input_ids, input_slice, loss_slice):
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    compare = logits[0, loss_slice, :]
    return compare

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    return top_indices


def get_filtered_cands_random_search(tokenizer: object, adv_suffix_tokens: object, filter_cand: object = True,token_change_number=1,batch_size = 16) -> object:

    adv_suffix_list = []
    n_tokens_change = token_change_number
    substitute_pos_start = random.choice(range(len(adv_suffix_tokens)-n_tokens_change))
    for i in range(batch_size):
        substitution_tokens = [random.choice(range(tokenizer.vocab_size)) for pos in range(substitute_pos_start, min(substitute_pos_start + n_tokens_change, len(adv_suffix_tokens)))]
        new_adv_suffix_tokens = adv_suffix_tokens[:substitute_pos_start]+substitution_tokens+adv_suffix_tokens[substitute_pos_start+n_tokens_change:]
        adv_suffix_list.append(new_adv_suffix_tokens)
    return adv_suffix_list
def get_filtered_cands(tokenizer: object, adv_suffix_tokens: object, control_cand: object, filter_cand: object = True,token_change_number=1,batch_size = 16) -> object:

    adv_suffix_list = []
    n_tokens_change = token_change_number
    substitute_pos_start = random.choice(range(len(adv_suffix_tokens)-n_tokens_change))
    for i in range(batch_size):
        #print(f"control_cand_shape: {control_cand.shape}")
    #substitution_tokens = np.random.randint(0, max_token_value, n_tokens_change).tolist()
    #print(f"len adv_suffix_tokens: {len(adv_suffix_tokens)}")
    #print(f"substitute start: {substitute_pos_start}")
    #print(f"substitute end: {min(substitute_pos_start + n_tokens_change, len(adv_suffix_tokens))}")
        while True:
            substitution_tokens = [random.choice(control_cand[pos,:].tolist()) for pos in range(substitute_pos_start, min(substitute_pos_start + n_tokens_change, len(adv_suffix_tokens)))]
            if not any(token in adv_suffix_tokens[substitute_pos_start:min(substitute_pos_start + n_tokens_change, len(adv_suffix_tokens))] for token in substitution_tokens):
                break
                                            
        new_adv_suffix_tokens = adv_suffix_tokens[:substitute_pos_start]+substitution_tokens+adv_suffix_tokens[substitute_pos_start+n_tokens_change:]

        adv_suffix_list.append(new_adv_suffix_tokens)

    return adv_suffix_list


def get_logits(*, model, tokenizer, input_ids_list, batch_size=512):
    input_ids = torch.tensor(input_ids_list).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs[0][:, :, :]
    #print(f"correct logits: {logits.shape}")
    gc.collect()
    return logits

def get_logits_kv_cache(*, model, tokenizer, input_ids_list, kv_cache, kv_cache_ids, batch_size=512):
    def slice_kv_cache(cache, k1, k2):
        # Directly return sliced key-value pairs without creating new list
        return [(key[:, :, k1:k2, :], value[:, :, k1:k2, :]) for key, value in cache]
    new_kv_caches = []
    new_input_ids_list = []
    start= 0
    for input_ids in input_ids_list:
        start_idx = 0
        for i in range(len(input_ids)):
            if not torch.equal(input_ids[i], kv_cache_ids[i]):  # Compare tensors properly
                start_idx = i - 10
                break
        start = start_idx
        #print(f"start_idx: {start_idx}")
        new_kv_caches.append(slice_kv_cache(kv_cache, 0, start_idx))

        new_input_ids = torch.tensor(input_ids[start_idx:]).to(model.device)
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
    logits = torch.cat((torch.zeros(logits.size(0), start, logits.size(2), device=logits.device), logits), dim=1)
    #print(f"logits: {logits.shape}")
    gc.collect()
    return logits

def target_loss(logits_positive, ids_positive, response_slice_positive, logits_negative, ids_negative, response_slice_negative):

    crit = nn.CrossEntropyLoss(reduction='none')
    positve_loss_slice = slice(response_slice_positive.start - 1, response_slice_positive.stop - 1)
    negative_loss_slice = slice(response_slice_negative.start - 1, response_slice_negative.stop - 1)
    compare_positive = logits_positive[:, positve_loss_slice, :].transpose(1, 2)
    response_ids_postive = ids_positive[:,response_slice_positive]
    compare_negative = logits_negative[:, negative_loss_slice, :].transpose(1, 2)
    response_ids_negative = ids_negative[:,response_slice_negative]
    #print(crit(compare_positive, response_ids_postive).shape)
    #print(crit(compare_negative, response_ids_negative).shape)
    loss = crit(compare_positive, response_ids_postive).mean(dim=-1) - crit(compare_negative, response_ids_negative).mean(dim=-1)
    return loss

def get_nonascii_toks(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)

def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                trust_remote_code=True).to(device).eval()
    #model = AutoModelForCausalLM.from_pretrained(
    #    model_path,
    #    low_cpu_mem_usage=True,
    #    torch_dtype=torch.float16,
    #    trust_remote_code=True,
    #    **kwargs
    #).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_dataset(split="train"):
    import json
    if split == "train":
        dataset = []
        with open('./data/harmful_queries.json', 'r') as file:
            harmful_queries = json.load(file)
            for e in list(harmful_queries.values())[0:10]:
                dataset.append((e[0], e[1], e[2]))

        with open('./data/benign_queries.json', 'r') as file:
            benign_queries = json.load(file)
            for e in list(benign_queries.keys())[0:5]:
                dataset.append((e,benign_queries[e][0],benign_queries[e][1]))

    elif split == "test_harmful":
        with open('./data/harmful_queries.json', 'r') as file:
            harmful_queries = json.load(file)
            dataset = [query[0] for query in harmful_queries.values()][10:30]
    elif split == "test_benign":
        with open('./data/benign_queries.json', 'r') as file:
            benign_queries = json.load(file)
            dataset = list(benign_queries.keys())[10:30]
    return dataset



def check_early_stopping(completions,loss_history):
    count = 0
    for i in range(len(completions)):
        if "sorry" in completions[i].lower():
            count += 1
    if count / len(completions) >= 0.3 and (len(loss_history) > 600 and loss_history[-200] - loss_history[-1] < 0.01):
        print("success")
        return True
    if (len(loss_history) > 600 and loss_history[-500] - loss_history[-1] < 0.01):
        print("success")
        return True
    return False

def wait_for_available_gpu_memory(required_memory_gb=70, device=0, check_interval=5):
    """
    Waits until the required amount of GPU memory is available.
    Args:
    required_memory_gb (float): Required GPU memory in gigabytes.
    device (int): GPU device index (default is 0)
    check_interval (int): Time interval in seconds between memory checks.
    Returns:
    None
    """
    required_memory_bytes = required_memory_gb * 1e9  # Convert GB to bytes
    while True:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device)
            info = nvmlDeviceGetMemoryInfo(handle)
            available_memory = info.free
            if available_memory >= required_memory_bytes:
                print(f"Sufficient GPU memory available: {available_memory / 1e9:.2f} GB")
                nvmlShutdown()
                return
            else:
                print(f"Waiting for GPU memory. Available: {available_memory / 1e9:.2f} GB, Required: {required_memory_gb:.2f} GB")
            nvmlShutdown()
        except NVMLError as error:
            print(f"Error getting GPU memory: {error}")
            # Fallback to PyTorch method
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                available_memory = total_memory - allocated_memory
                if available_memory >= required_memory_bytes:
                    print(f"Sufficient GPU memory available (PyTorch): {available_memory / 1e9:.2f} GB")
                    return 1
                else:
                    print(f"Waiting for GPU memory (PyTorch). Available: {available_memory / 1e9:.2f} GB, Required: {required_memory_gb:.2f} GB")
            else:
                print("CUDA is not available")
        time.sleep(check_interval)