import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import logging


def pruning_model(model, px):

    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Linear):
            parameters_to_prune.append((m, 'weight'))

    logging.info('Global pruning module number = {}'.format(len(parameters_to_prune)))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def prune_model_custom(model, mask_dict):

    for name,m in model.named_modules():
        if isinstance(m, nn.Linear):
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def remove_prune(model):
    # See https://pytorch.org/docs/stable/_modules/torch/nn/utils/prune.html#L1Unstructured
    for m in model.modules():
        if isinstance(m, nn.Linear):
            prune.remove(m, 'weight')

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict

def set_original_weights(model, original_state_dict):
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight_orig' in name:
            module_name = '.'.join(name.split('.')[:-1])
            original_weight_tensor = original_state_dict[module_name + '.weight'].data.cpu().numpy()
            weight_dev = param.device
            param.data = torch.from_numpy(original_weight_tensor).to(weight_dev)
    return model

def check_sparsity(model):
    
    zero_count = 0
    all_count = 0

    for m in model.modules():
        if isinstance(m, nn.Linear):
            zero_count += float(torch.sum(m.weight == 0))
            all_count += float(m.weight.nelement())

    remain_weight = 100*(1-zero_count/all_count)
    logging.info('* remain weight = {:.4f}%'.format(remain_weight))
    print('* remain weight = {:.4f}%'.format(remain_weight))

    return remain_weight

def print_dict(model_dict):
    for key in model_dict.keys():
        print(key, model_dict[key].shape)

def count_pruning_weight_rate(model):

    all_count = 0
    overall_number = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            all_count += float(m.weight.nelement())

    for key in model.state_dict().keys():
        overall_number += model.state_dict()[key].nelement()

    logging.info('parameters of pruning scope = {}'.format(all_count))
    logging.info('parameters overall = {}'.format(overall_number))
    logging.info('rate = {:.6f}'.format(all_count/overall_number))

def check_mask(mask_dict):
    zero_count = 0
    all_count = 0 
    for key in mask_dict.keys():
        zero_count += float(torch.sum(mask_dict[key] == 0))
        all_count += float(mask_dict[key].nelement())
    logging.info(100*(1-zero_count/all_count))

def zero_rate(tensor):
    zero = float(torch.sum(tensor == 0))
    all_count = float(tensor.nelement())
    return 100*zero/all_count

def check_different(model, mask_dict):

    for name,m in model.named_modules():
        if isinstance(m, nn.Linear):
            logging.info(name+'.weight', 'mask rate = {:.2f}'.format(zero_rate(mask_dict[name+'.weight_mask'])), 'module rate = {:.2f}'.format(zero_rate(m.weight)))

def prune_main(model, prune_percentage, original_state_dict):
    # global magnitude pruning
    pruning_model(model, prune_percentage)

    return rewind(model, original_state_dict)


def rewind(model, original_state_dict):
    # extract mask weight in model.state_dict()
    mask_dict = extract_mask(model.state_dict())

    # remove pruning for the purpose of loading original weight
    remove_prune(model)

    # rewind to original weight
    # be carefull:  don't let original_weight and model.state_dict() share the same memory
    model.load_state_dict(original_state_dict)

    # pruning with custom mask
    prune_model_custom(model, mask_dict)

    # check current sparsity
    check_sparsity(model)

    return mask_dict

