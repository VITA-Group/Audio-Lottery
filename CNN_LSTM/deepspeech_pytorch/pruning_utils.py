import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import logging


def pruning_model(model, px, rnn_type=nn.LSTM, bidirectional=True, Linear=True):

    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, 'weight'))
        elif isinstance(m, rnn_type):
            parameters_to_prune.append((m, 'weight_ih_l0'))
            parameters_to_prune.append((m, 'weight_hh_l0'))
            if bidirectional:
                parameters_to_prune.append((m, 'weight_ih_l0_reverse'))
                parameters_to_prune.append((m, 'weight_hh_l0_reverse'))
        elif isinstance(m, nn.Linear):
            if Linear:
                parameters_to_prune.append((m, 'weight'))

    logging.info('Global pruning module number = {}'.format(len(parameters_to_prune)))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_model_random(model, px, rnn_type=nn.LSTM, bidirectional=True, Linear=True):

    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.random_unstructured(m, 'weight', amount=px)
        elif isinstance(m, rnn_type):
            prune.random_unstructured(m, 'weight_ih_l0', amount=px)
            prune.random_unstructured(m, 'weight_hh_l0', amount=px)
            if bidirectional:
                prune.random_unstructured(m, 'weight_ih_l0_reverse', amount=px)
                prune.random_unstructured(m, 'weight_hh_l0_reverse', amount=px)
        elif isinstance(m, nn.Linear):
            if Linear:
                prune.random_unstructured(m, 'weight', amount=px)


def prune_model_custom(model, mask_dict, rnn_type=nn.LSTM, bidirectional=True, Linear=True):

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
        elif isinstance(m, rnn_type):
            prune.CustomFromMask.apply(m, 'weight_ih_l0', mask=mask_dict[name+'.weight_ih_l0_mask'])
            prune.CustomFromMask.apply(m, 'weight_hh_l0', mask=mask_dict[name+'.weight_hh_l0_mask'])
            if bidirectional:
                prune.CustomFromMask.apply(m, 'weight_ih_l0_reverse', mask=mask_dict[name+'.weight_ih_l0_reverse_mask'])
                prune.CustomFromMask.apply(m, 'weight_hh_l0_reverse', mask=mask_dict[name+'.weight_hh_l0_reverse_mask'])
        elif isinstance(m, nn.Linear):
            if Linear:
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def remove_prune(model, rnn_type=nn.LSTM, bidirectional=True, Linear=True):
    # See https://pytorch.org/docs/stable/_modules/torch/nn/utils/prune.html#L1Unstructured
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m, 'weight')
        elif isinstance(m, rnn_type):
            prune.remove(m, 'weight_ih_l0')
            prune.remove(m, 'weight_hh_l0')
            if bidirectional:
                prune.remove(m, 'weight_ih_l0_reverse')
                prune.remove(m, 'weight_hh_l0_reverse')
        elif isinstance(m, nn.Linear):
            if Linear:
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

def randomize_weights(model, rnn_type, bidirectional=True, Linear=True):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, rnn_type):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.xavier_uniform_(m.weight_hh_l0)
            if bidirectional:
                nn.init.xavier_uniform_(m.weight_ih_l0_reverse)
                nn.init.xavier_uniform_(m.weight_hh_l0_reverse)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)


def check_sparsity(model, rnn_type=nn.LSTM, bidirectional=True, Linear=True):
    
    zero_count = 0
    all_count = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            zero_count += float(torch.sum(m.weight == 0))
            all_count += float(m.weight.nelement())
        elif isinstance(m, rnn_type):

            zero_count += float(torch.sum(m.weight_ih_l0 == 0))
            all_count += float(m.weight_ih_l0.nelement())

            zero_count += float(torch.sum(m.weight_hh_l0 == 0))
            all_count += float(m.weight_hh_l0.nelement())

            if bidirectional:
                zero_count += float(torch.sum(m.weight_ih_l0_reverse == 0))
                all_count += float(m.weight_ih_l0_reverse.nelement())

                zero_count += float(torch.sum(m.weight_hh_l0_reverse == 0))
                all_count += float(m.weight_hh_l0_reverse.nelement())

        elif isinstance(m, nn.Linear):
            if Linear:
                zero_count += float(torch.sum(m.weight == 0))
                all_count += float(m.weight.nelement())

    remain_weight = 100*(1-zero_count/all_count)
    logging.info('* remain weight = {:.4f}%'.format(remain_weight))

    return remain_weight

def print_dict(model_dict):
    for key in model_dict.keys():
        print(key, model_dict[key].shape)

def count_pruning_weight_rate(model, rnn_type=nn.LSTM, bidirectional=True, Linear=True):

    all_count = 0
    overall_number = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            all_count += float(m.weight.nelement())
        elif isinstance(m, rnn_type):
            all_count += float(m.weight_ih_l0.nelement())
            all_count += float(m.weight_hh_l0.nelement())
            if bidirectional:
                all_count += float(m.weight_ih_l0_reverse.nelement())
                all_count += float(m.weight_hh_l0_reverse.nelement())
        elif isinstance(m, nn.Linear):
            if Linear:
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

def check_different(model, mask_dict, rnn_type=nn.LSTM, bidirectional=True, Linear=True):

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            logging.info(name+'.weight', 'mask rate = {:.2f}'.format(zero_rate(mask_dict[name+'.weight_mask'])), 'module rate = {:.2f}'.format(zero_rate(m.weight)))
        elif isinstance(m, rnn_type):
            logging.info(name+'.weight_ih_l0', 'mask rate = {:.2f}'.format(zero_rate(mask_dict[name+'.weight_ih_l0_mask'])), 'module rate = {:.2f}'.format(zero_rate(m.weight_ih_l0)))
            logging.info(name+'.weight_hh_l0', 'mask rate = {:.2f}'.format(zero_rate(mask_dict[name+'.weight_hh_l0_mask'])), 'module rate = {:.2f}'.format(zero_rate(m.weight_hh_l0)))

            if bidirectional:
                logging.info(name+'.weight_ih_l0_reverse', 'mask rate = {:.2f}'.format(zero_rate(mask_dict[name+'.weight_ih_l0_reverse_mask'])), 'module rate = {:.2f}'.format(zero_rate(m.weight_ih_l0_reverse)))
                logging.info(name+'.weight_hh_l0_reverse', 'mask rate = {:.2f}'.format(zero_rate(mask_dict[name+'.weight_hh_l0_reverse_mask'])), 'module rate = {:.2f}'.format(zero_rate(m.weight_hh_l0_reverse)))

        elif isinstance(m, nn.Linear):
            if Linear:
                logging.info(name+'.weight', 'mask rate = {:.2f}'.format(zero_rate(mask_dict[name+'.weight_mask'])), 'module rate = {:.2f}'.format(zero_rate(m.weight)))


def prune_main(model, prune_percentage, rnn_type, original_state_dict, bidirectional=True, Linear=True, random_prune=False):
    # random prune for baseline
    if random_prune:
        pruning_model_random(model, prune_percentage, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)
    else:
        # global magnitude pruning
        pruning_model(model, prune_percentage, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)

    return rewind(model, rnn_type, original_state_dict, bidirectional, Linear)


def rewind(model, rnn_type, original_state_dict, bidirectional=True, Linear=True):
    # extract mask weight in model.state_dict()
    mask_dict = extract_mask(model.state_dict())

    # remove pruning for the purpose of loading original weight
    remove_prune(model, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)

    # rewind to original weight
    # be carefull:  don't let original_weight and model.state_dict() share the same memory
    model.load_state_dict(original_state_dict)

    # pruning with custom mask
    prune_model_custom(model, mask_dict, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)

    # check current sparsity
    check_sparsity(model, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)
    
    return mask_dict

def load_winning_ticket(model, mask_dict, rnn_type, original_state_dict, bidirectional=True, Linear=True, remove_p=True):
    # remove pruning for the purpose of loading original weight
    if remove_p:
        remove_prune(model, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)
    
    model.load_state_dict(original_state_dict)
    # load init checkpoint
    # model = set_original_weights(model, original_state_dict)

    # pruning with custom mask
    prune_model_custom(model, mask_dict, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)

    # check current sparsity
    check_sparsity(model, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)

def load_winning_ticket_with_random_init(model, mask_dict, rnn_type, bidirectional=True, Linear=True):
    randomize_weights(model, rnn_type)
    # load init checkpoint
    # model = set_original_weights(model, original_state_dict)
    # pruning with custom mask
    prune_model_custom(model, mask_dict, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)
    # check current sparsity
    check_sparsity(model, rnn_type=rnn_type, bidirectional=bidirectional, Linear=Linear)
