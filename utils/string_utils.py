import torch
from fastchat import model
def remove_bos_token( tokenizer,token_ids):
    if token_ids[0] == tokenizer.bos_token_id:
        token_ids = token_ids[1:]
    return token_ids
def load_conversation_template(template_name):
    if template_name == 'llama3':
        template_name = 'llama-3'
    conv_template = model.get_conversation_template(template_name)
    #print(f"conv_template: {conv_template}")
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-3':
        conv_template.sep2 = conv_template.sep2.strip()
    return conv_template


def modify_sys_prompts(conv_template, instruction, template_name):
    if template_name == "llama3":
        conv_template.system = f"[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\n"
    return conv_template

class SuffixManager:
    def __init__(self, args, tokenizer,query, positive_response, negative_response,  suffix_ids_list):

        self.tokenizer = tokenizer
        self.positive_response = positive_response
        self.negative_response = negative_response
        self.suffix_ids_list = suffix_ids_list
        self.query= query
        if "llama3" in args.model:
            self.msg_start = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""
            self.msg_end = """<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        elif "mistral" in args.model:
            self.msg_start = """[INST] """
            self.msg_end = """ [/INST]"""
        else:
            raise ValueError(f"Model {args.model} not supported")

        self.positive_response_ids = remove_bos_token(self.tokenizer,self.tokenizer.encode(self.positive_response))
        self.negative_response_ids = remove_bos_token(self.tokenizer,self.tokenizer.encode(self.negative_response))
        self.msg_end_ids = remove_bos_token(self.tokenizer,self.tokenizer.encode(self.msg_end))
        self.prompt_clean_ids = self.tokenizer.encode(self.msg_start+ self.query)
    
        self.prompt_ids_list = [self.prompt_clean_ids+suffix_ids+self.msg_end_ids for suffix_ids in self.suffix_ids_list]
        self._negative_response_slice = slice(len(self.prompt_ids_list[0]), len(self.prompt_ids_list[0]) + len(self.negative_response_ids))
        self._positive_response_slice = slice(len(self.prompt_ids_list[0]), len(self.prompt_ids_list[0]) + len(self.positive_response_ids))
        self._log_p_positive_slice = slice(len(self.prompt_ids_list[0])-1, len(self.prompt_ids_list[0]) + len(self.positive_response_ids)-1)
        self._log_p_negative_slice = slice(len(self.prompt_ids_list[0])-1, len(self.prompt_ids_list[0]) + len(self.negative_response_ids)-1)
        self._control_slice = slice(
            len(self.prompt_clean_ids), 
            len(self.prompt_clean_ids)+len(self.suffix_ids_list[0])
        )
        #print("prompt: ",self.get_prompt())
    def get_prompt_ids(self):
        return self.prompt_ids_list
    def get_prompt(self):
        return self.msg_start+self.query+self.tokenizer.decode(self.suffix_ids_list[0])+self.msg_end

    def get_input_ids_train(self):
        assert all(len(suffix_ids) == len(self.suffix_ids_list[0]) for suffix_ids in self.suffix_ids_list)
        positive_input_ids = [self.prompt_clean_ids + suffix_ids +self.msg_end_ids+ self.positive_response_ids for suffix_ids in self.suffix_ids_list]
        negative_input_ids = [self.prompt_clean_ids + suffix_ids +self.msg_end_ids+ self.negative_response_ids for suffix_ids in self.suffix_ids_list]
        return positive_input_ids, negative_input_ids
    def check_consistency(self):
        input_ids_list = self.get_input_ids_train()[0]
        for i in range(0,len(input_ids_list)):
            input_ids = input_ids_list[i]
            start_idx = 0
            for j in range(len(input_ids)):
                if not input_ids[j]==input_ids_list[0][j]:  # Compare tensors properly
                    start_idx = j
                    break

            #print(f"start_idx: {start_idx}")
        return True
            



