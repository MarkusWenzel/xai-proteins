from captum.attr import LayerIntegratedGradients
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.common import _reshape_and_sum
import torch.nn as nn
import torch
from torch.functional import Tensor
import transformers
from prot_ec_mod import ProteinClassifier
from pathlib import Path
import pandas as pd
import logging, glob, json, sys, csv, os, re

def run_embedding_all_proteins(prot_df, model, output_dir):
    ''' store embedding attribution for every sample in embedding_attribution.csv '''
    
    with open(f'{output_dir}embedding_attribution.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["protein_name","protein_label","attribution"])
        for row in tqdm(prot_df.itertuples()):
            prot_seq, prot_name ,prot_label = row[1],row[2],row[3]
            input_ids, ref_input_ids, target = get_model_inputs(prot_seq, prot_label, model)
            attr = ig_embedding(model, input_ids, ref_input_ids, target, module_name="embeddings",n_steps=500) # return embedding tensor
            attr_across_tokens = torch.sum(attr.squeeze(0),dim=1)
            spamwriter.writerow([prot_name, prot_label,attr_across_tokens.tolist()])

            del attr,input_ids,ref_input_ids,target
            torch.cuda.empty_cache()
            csvfile.flush()

def run_attention_attribution_all_proteins(n_steps:int, hooks:list):
    """
    Run integrated gradient for each sample and retrieve the attribution at 
    every attention mechanism.

    Args:
        n_steps (int): How many steps Integrated Gradient should run for.
        hooks (list): A list of the hook indexes to run Integrated Gradient with.
    """
    model, output_dir, prot_df = ig_setup(level=1,ec_class=50)
    if not os.path.isdir(output_dir):
        raise(f"{output_dir} does not exist.")
    else:
        print("output_dir:",output_dir)
        print("Executing Hook:", str(hooks))
    
    # iterate over the protein data
    for row in tqdm(prot_df.itertuples()):
        # get a list of files created
        prot_seq, prot_name ,prot_label = row[1],row[2],row[3]
        if os.path.isfile(output_dir+prot_name+"_"+str(hooks)+".csv"):
            print(output_dir+prot_name+"_"+str(hooks)+".csv already exists.")
            continue
        else:
            print(output_dir+prot_name+"_"+str(hooks)+".csv does not exist.")

        # get the correct input tensors
        input_ids, ref_input_ids, target = get_model_inputs(prot_seq, prot_label, model)
        assert input_ids.shape[-1]<=1000, "input sequence is above 1000"
        attr = ig_embedding(model, input_ids, ref_input_ids, target, module_name="all", n_steps=n_steps,hooks=hooks) # returns a list of tensors
        attribution_to_csv(torch.cat(attr,dim=0), output_dir=output_dir, protein_name=prot_name, part=str(hooks))      
        del attr,input_ids,ref_input_ids,target
        torch.cuda.empty_cache()

            
def ig_setup(level, ec_class, data_path: str = "data/ec50_level1/test.json"):
    '''
    Get model and protein dataframe for the datapath
    Return model, output_dir, prot_df
    '''
    print("data_path:",data_path)

    assert os.path.isfile(data_path), f"{data_path} does not exist."
    prot_df = pd.read_json(data_path)
    
    prot_df['sequence'] = prot_df.sequence.apply(lambda x: re.sub(r"[UZOB]", "X", x)) # replace rare aminoacids
    prot_df['sequence'] = prot_df.sequence.apply(lambda x: [" ".join(list(x))]) # spacing the sequences
    prot_df['label'] = prot_df['label'].apply(lambda x: [str(float(x))]) # label style for prepare_sample function

    model = get_model(level=level, ec_class=ec_class)
    output_dir = os.path.dirname(data_path)+'/ig_outputs/'
    try:
        os.mkdir(output_dir)
        logging.info(f"{output_dir} Folder created.")
    except FileExistsError as e:
        logging.info(f"{output_dir} Folder already exists.")
        pass

    print("model_hparams",model.hparams)
    print("output dir:",output_dir)

    return model, output_dir, prot_df


def get_model(level, ec_class):
    '''
    Gets the fine tuned model from the directory specified by level and ec_class
    Parameters: level: 0,1,2; ec_class: 40,50
    Return: model (cuda zero_grad eval)
    '''

    current_dir = os.path.dirname(__file__)    
    path = Path(current_dir).resolve()
    pretrained_model_path = f"{path}/models/ec{ec_class}_level{level}/checkpoints/"
    assert os.path.isdir(pretrained_model_path) , f"Following folder does not exist: {pretrained_model_path}"
    best_path = glob.glob(pretrained_model_path + "*")[0]
    model_og = ProteinClassifier.load_from_checkpoint(best_path)
    model_og.hparams.gradient_checkpointing = False
    model_og.cuda()
    model_og.eval()
    model_og.zero_grad()

    return model_og


def get_model_inputs(prot_seq, prot_label, model):
    '''
    Get the vars from the .json protein file and construct the model input and baseline with the given tokenizer.
    Returns:
    \t Tensors input_ids, ref_input_ids
    \t target, prot_seq
    '''

    inputs_dict, label_dict = model.prepare_sample({'seq':prot_seq,'label':prot_label})
    input_ids = inputs_dict['input_ids'] # with special tokens
    input_ids = torch.tensor(input_ids, device='cuda')

    cls_token_id = model.tokenizer.cls_token_id
    pad_token_id = model.tokenizer.pad_token_id
    sep_token_id = model.tokenizer.sep_token_id

    ref_input_ids = [cls_token_id] + [pad_token_id]*(input_ids.shape[-1]-2) + [sep_token_id]
    ref_input_ids = torch.tensor([ref_input_ids], device='cuda')

    target = label_dict['labels']

    return input_ids, ref_input_ids, target


def attribution_to_csv(attribution: Tensor, output_dir: str, protein_name: str, part:str) -> None:
    '''
    Split attribution of size 30xNx1024 in individual heads and write in csv 480 rows
    of length N.

    Parameter:
        attribution: 30x(N+2)x1024 Tensor where 30 = Layer amount of model
        output_path: path for the csv
        protein_name: the protein name for labeling the file
        target: the true label of the sample
    '''
    tok_seq_len = attribution.shape[1]
    protein_name = protein_name
    if not output_dir.endswith("/"):
        output_dir = output_dir+"/"
    output_path = f"{output_dir}{protein_name}_{part}.csv"

    # output_file appending
    fo = open(output_path, "w")
    fo.close()
    with open(output_path, mode="a") as file:

        # write csv header "0,1,2,3,4..N+2"
        file_writer = csv.writer(file, delimiter=",")
        header = np.arange(tok_seq_len).tolist()
        file_writer.writerow(header)

        # splitting attention and writing rows
        for layer_tensor in attribution:
            for head in range(16):
                start = head*64
                end = head*64+64
                head_string_rel = layer_tensor[:, start:end]  # [N,64]
                att_sum_string = torch.sum(head_string_rel, axis=1)  # [N]
                line = att_sum_string.tolist()
                file_writer.writerow(line)


def ig_embedding(model, input_ids, ref_input_ids, target, n_steps: int = 50, attribute_to_layer_input: bool = False, module_name: str = "embeddings", method = "gausslegendre", hooks:list = [1], **kwargs):

    # Parameter: module_name: for embedding 'embedding' or nothing, for layer type i.e. 'layer23', for all layers pass any other string
    # Return: a) attribution; b) case:all: list of 30 tensors of size 1xNx1024

    def forward_func(inputs, token_type_ids=None, attention_mask=None):
        pred_logit = model(inputs,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        pred = pred_logit['logits']
        return pred

    # set module/layer hooks, get module inp,base
    if module_name.startswith("layer"):
        layer_idx = int(module_name[5:])
        module = model.ProtTrans.encoder.layer[layer_idx].attention.self
        back_hook = Hook(module, backward=True, hook_to_module_input=False)
        fwd_hook = Hook(module, backward=False, hook_to_module_input=False)
    elif module_name == "embeddings" :
        pass
    elif module_name == "all":
        fwd_hooks, back_hooks = get_all_layer_hooks(model,hooks)

    # run integrated gradients
    attr = run_ig(forward_func, model.ProtTrans.embeddings, input_ids, ref_input_ids, target, n_steps, method, False, attribute_to_layer_input)

    # scaling hook gradients
    if module_name=="embeddings":
        return attr
    elif module_name.startswith("layer"):
        module_forward_out = fwd_hook.out[2:] # due to inp, baseline for embedding fwd run.
        hook_attr = scaling_backward_grads(back_hook.out, module_forward_out, n_steps)
        hook_attr_sum = round(torch.sum(hook_attr).item(),4)
        return hook_attr
    elif module_name == "all":
        logging.info("Scaling all layer attributions.")
        hook_attr_list = []
        for i in range(len(hooks)):
            module_forward_out = fwd_hooks[i].out[2:] # due to inp, baseline for embedding fwd run.
            hook_attr = scaling_backward_grads(back_hooks[i].out, module_forward_out, n_steps)
            hook_attr_sum = torch.sum(hook_attr,dim=0).unsqueeze(0) # n_stepsxNx1024 -> 1xNx1024
            hook_attr_list.append(hook_attr_sum)
        for b,f in zip(back_hooks,fwd_hooks):
            b.close()
            f.close()
            del b,f
        del attr, hook_attr_sum, hook_attr, module_forward_out
        return hook_attr_list


def get_all_layer_hooks(model,hooks):
    '''
    Creates 30 forward and back hooks
    could probably get the layer number from module hparams instead of hard coding
    '''
    encoder_module = model.ProtTrans.encoder
    if hooks != []:
        hook_list=hooks
    else:
        hook_list = np.arange(30).tolist()
    back_hooks_module_list = [encoder_module.layer[i].attention.self for i in hook_list]
    fwd_hooks_module_list = [encoder_module.layer[i].attention.self for i in hook_list]  
    back_hooks = [Hook(module, backward=True, hook_to_module_input=False) for module in back_hooks_module_list]
    fwd_hooks = [Hook(module, backward=False, hook_to_module_input=False) for module in fwd_hooks_module_list]    
    return fwd_hooks, back_hooks


class Hook():
    '''
    create hooks on modules i.e.
    back_hook = Hook(layer[1],backward=True)
    fwd_hook = Hook(layer[1])
    '''
    def __init__(self, module, backward=False, hook_to_module_input=False):
        self.lin = []
        self.out = []
        self.backward = backward

        if backward == True:
            if hook_to_module_input == True:
                self.hook = module.register_full_backward_hook(self.hook_fn_module_input)
            else:
                self.hook = module.register_full_backward_hook(self.hook_fn_module_backward_output)

        elif backward == False:
            if hook_to_module_input == True:
                self.hook = module.register_forward_hook(self.hook_fn_module_input)
            else:
                self.hook = module.register_forward_hook(self.hook_fn_module_output)

    def hook_fn_module_input(self, module, input, output):
        f = input[0].detach().cpu()
        self.lin.append(f)
    def hook_fn_module_output(self, module, input, output):
        f = output[0].detach().cpu()
        self.out.append(f)
    def hook_fn_module_backward_output(self, module, input, output):
        f = output[0].detach().cpu()
        self.out.append(f)

    def close(self):
        self.hook.remove()
        del self.out, self.lin


def run_ig(forward_func, module, input_ids, ref_input_ids, target, n_steps, method, return_convergence_delta, attribute_to_layer_input):
    ''' Runs integrated gradient.  Return: attribution or attribution,delta '''
    lig = LayerIntegratedGradients(forward_func, module)
    if return_convergence_delta==True:
        attr, delta = lig.attribute(input_ids,
                                baselines=ref_input_ids,
                                target=target,
                                n_steps=n_steps,
                                method = method,
                                internal_batch_size=1,
                                return_convergence_delta=return_convergence_delta,
                                attribute_to_layer_input=attribute_to_layer_input)
        return attr,delta
    else:
        attr = lig.attribute(input_ids,
                        baselines=ref_input_ids,
                        target=target,
                        n_steps=n_steps,
                        method = method,
                        internal_batch_size=1,
                        return_convergence_delta=return_convergence_delta,
                        attribute_to_layer_input=attribute_to_layer_input)
        return attr


def scaling_backward_grads(module_back_grads_out:list, module_forward_out:list, n_steps:int):
    '''
    Scaling the backward gradient using finite difference coefficient, with accuracy 2, of the forward pass tensors,
    because when extracting the gradient at a layer different from what is passed to captum the gradients
    do not lie on a straight path.
    More on fdc: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    More on numpy.gradient: https://numpy.org/doc/stable/reference/generated/numpy.gradient.html

    Parameter:
      module_back_grads_out: list of tensors from the backward hook from each step
      module_forward_out: list of tensors from the forward hook
      list of alpha values

    Return: 
      scaled_grad: scaled gradients
    '''
    assert len(module_back_grads_out)==(len(module_forward_out)), f"backward_gradient_list length is unequal forward_tensor_list.{len(module_back_grads_out)}/{len(module_forward_out)}"
    step_func, alphas_func = approximation_parameters("gausslegendre")
    step_sizes, alpha_list = step_func(n_steps),alphas_func(n_steps) # alpha list is without 0,1

    grads = torch.cat(module_back_grads_out)
    module_forward_out_tensor = torch.cat(module_forward_out)
    alpha_list = (torch.tensor(alpha_list).to(module_forward_out_tensor.device),)

    # IG formula implementation
    scaled_tensors = torch.gradient(module_forward_out_tensor, spacing=alpha_list, edge_order=1,dim=0)[0]
    scaled_tensors =  torch.cat([x.unsqueeze(0).detach() for x in scaled_tensors])
    
    scaled_grads = grads.contiguous().view(n_steps, -1)* torch.tensor(step_sizes).view(n_steps, 1).to(grads.device)
    scaled_grads = scaled_grads.reshape((n_steps,) + scaled_tensors.shape[1:])
   
    total_grads = scaled_grads * scaled_tensors
    return total_grads

