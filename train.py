import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from lib.dataset import AVDataset, ImageAVDataset
from lib.eval import WordErrorRate, RecoveryRate
from lib.reranker.rerankpipeline import RerankPipeline, RerankLayer

import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.wav2vec2.bundles import WAV2VEC2_ASR_BASE_960H
import torch.nn.functional as F
from lib.reranker.loader import RerankDataset
from lib.models import TransformerPipeline, MultimodalTransformerPipeline
from main import init_word_to_token
from transformers import GPT2Tokenizer, GPT2Model, AutoModelWithLMHead, DataCollatorForLanguageModeling
import csv
import sys

from PIL import Image



def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    # Data Extraction Parameters
    parser.add_argument('--run', type=str,
                        help='ID of the run to load weights from.')
    parser.add_argument('--id_noise', type=str, nargs="*",
                        help='Noise id. Choices are: ["clean", "mask_{rho}", "swap_{rho}"].')
    parser.add_argument('--eval_set', type=str, default='valid',
                        help='Evaluation set to use.')
    parser.add_argument('--train_set', type=str, default='train',
                        help='Evaluation set to use.')
    
    parser.add_argument('--top_k', type=int, default=5,
                        help='Beam size for beam search decoder.')
    parser.add_argument('--out', type=str, default='models/results.csv',
                        help='Output file path.')
    parser.add_argument('--dir_source', type=str, default="data",
                        help='Name of directory to load the data from.')

    # Pipeline Parameters
    parser.add_argument('--d_audio', type=int, default=[312, 768], nargs=2,
                        help='Dimension of the audio embedding.')
    parser.add_argument('--d_vision', type=int, default=512,
                        help='Dimension of the vision embedding.')
    parser.add_argument('--d_obj_count', type=int, default=59,
                        help='Dimension of the object counts (the number of objects possible).')
    parser.add_argument('--max_target_len', type=int, default=25,
                        help='Maximum sequence length of a target transcript.')
    parser.add_argument('--depth', type=int, default=4,
                        help='Depth of the TransformerDecoder')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout of the TransformerDecoderLayer')

    # Torch Parameters
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help='Device to use for torch.')

    args = parser.parse_args()

    # Assertions, to make sure params are valid.
    assert len(args.d_audio) == 2, "d_audio must have length 2"

    assert args.run == 'baseline' or os.path.exists(f"models/{args.run}.pt"), f"run '{args.run}' not found"
    assert len(args.id_noise) > 0, "id_noise must have at least one value"
    for id_noise in args.id_noise:
        assert os.path.isdir(f"{args.dir_source}/train/audio/{id_noise}"), f"id_noise '{id_noise}' not found"

    return args

def init_pipeline(device: str):
    bundle = WAV2VEC2_ASR_BASE_960H
    if args.run[:8] == 'unimodal':
        pipeline = 'unimodal'
    else:
        pipeline = 'multi[clip]'
    if pipeline == 'unimodal':
        pipeline = nn.DataParallel(TransformerPipeline(
            args.d_audio, n_tokens, args.depth, args.max_target_len, args.dropout))
    elif pipeline in ['multi[resnet]', 'multi[clip]']:
        pipeline = nn.DataParallel(MultimodalTransformerPipeline(
            args.d_audio, args.d_vision, n_tokens, args.depth, args.max_target_len, args.dropout))
    elif args.pipeline == 'multi[obj_count]':
        pipeline = nn.DataParallel(MultimodalTransformerPipeline(
            args.d_audio, args.d_obj_count, n_tokens, args.depth, args.max_target_len, args.dropout))
    else:
        assert False, f"Pipeline {args.pipeline} not implemented."

    pipeline.module.load_state_dict(torch.load(f'models/{args.run}.pt', map_location=torch.device('cpu')))
    pipeline.to(device)
    return pipeline


def init_dataloader(dataset):
    aux_modality = "clip"
    if args.pipeline == 'multi[clip]':
        aux_modality = "clip"
    elif args.pipeline == "multi[obj_count]":
        aux_modality = "obj_count"
    dataset = AVDataset(f'{args.dir_source}/{dataset}',
                        list(args.id_noise),
                        aux_modality=aux_modality,
                        pad_token=PAD_token,
                        max_target_length=args.max_target_len,
                        load_noise=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset, loader

def init_dataloader_image(dataset):
    aux_modality = "clip"
    if args.pipeline == 'multi[clip]':
        aux_modality = "clip"
    elif args.pipeline == "multi[obj_count]":
        aux_modality = "obj_count"
    dataset = ImageAVDataset(f'{args.dir_source}/{dataset}',
                        list(args.id_noise),
                        aux_modality=aux_modality,
                        pad_token=PAD_token,
                        max_target_length=args.max_target_len,
                        load_noise=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset, loader


def train_reranker(train_dataset, test_dataset, pipeline, scorer, rerankmodel, pipeline_type, args, num_epochs):
    count = 0
    celoss = torch.nn.CrossEntropyLoss()
    running_loss = 0
    
    train_loader = DataLoader(train_dataset, batch_size = 10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 10, shuffle=True)
    print("Number of Batches:", len(test_dataset)/10)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rerankmodel.parameters())
    best_wer = 1.0
    for i in range(num_epochs):
        print("Starting epoch {}...".format(i))
        loader = 0
        for load in [train_loader, test_loader]:
            wer = 0
            epoch_loss = 0
            batch_loss = 0
            batch_count = 0
            epoch_wer = 0
            base_wer = 0
            for vals, labs, wers, top_1_wer, min_idx in iter(load):
                
                optimizer.zero_grad()
                output = rerankmodel(vals)
                loss = loss_fn(output, min_idx)
                if loader == 0:   
                    loss.backward()
                    optimizer.step()
                batch_loss += loss.item() * vals.size(0)
                pred_wer_indices = torch.argmax(output, dim=1)
                #print(wers)
                #print(pred_wer_indices)
                
                
                for j in range(10):
                    base_wer += wers[min_idx[j]][j]
                    epoch_wer += wers[pred_wer_indices[j]][j]
                
                print("epoch {}, batch {} avg loss {}".format(i, batch_count, batch_loss/3))
                sys.stdout.flush()
                epoch_loss += batch_loss
                batch_loss = 0
                batch_count+=1
                if batch_count%50 == 0 and batch_count!=0:
                  
                    if loader == 0:
                        num = batch_count/50
                        print("TRAIN batch {} wer =".format(num), epoch_wer/(50*10))
                        print("BASE WER:", base_wer/(50*10))
                        base_wer = 0
                        epoch_wer = 0
            else:
                print("VALID epoch {} loss =".format(i), epoch_loss/(len(test_loader)*3), "wer =", epoch_wer/(len(test_loader)*3))
                print("BASE WER:", base_wer/(len(test_loader)*3))
                """
                if epoch_wer/len(train_loader) < best_wer:
                    best_wer = epoch_wer/len(train_loader)
                    best_wts = copy.deepcopy(rerankmodel.state_dict())
                    print("BASE WER:", base_wer/batch_count)
                """
            loader += 1
    rerankmodel.load_state_dict(best_wts)
    return rerankmodel
            

if __name__ == "__main__":
    args = parse_args()
    # Determine unimodal/multimodal
    if args.run == 'baseline' or args.run[:8] == 'unimodal':
        args.pipeline = 'unimodal'
        pipeline_type = 'unimodal'
    else:
        pipeline_type = 'multimodal'
        args.pipeline = args.run[:args.run.find('_[')]

        if args.pipeline not in ['multi[resnet]', 'multi[clip]', 'multi[obj_count]']:
            assert False, f"Pipeline {args.pipeline} not found."

    # Print device
    print(f"> Using device: {bcolors.OKGREEN}{args.device}{bcolors.ENDC}")
    print(f"> Testing run {bcolors.OKCYAN}{args.run}{bcolors.ENDC}")

    # Initialize pipeline and logger
    word_to_token, BOS_token, EOS_token, PAD_token, n_tokens = init_word_to_token(args.dir_source)
    pipeline = init_pipeline(args.device)
    asrtraindataset, _ = init_dataloader_image(args.train_set)
    asrtestdataset, asrtestloader = init_dataloader(args.eval_set)
    
    scorer = RerankPipeline(5) 
    traindataset = RerankDataset(asrtraindataset, pipeline, scorer, args, pipeline_type)
    testdataset = RerankDataset(asrtraindataset, pipeline, scorer, args, pipeline_type)
    model = RerankLayer(5)
    sd = {}
    lin_weight = torch.FloatTensor([0.2, 0.2, 0.3, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03])
    lin_weight_1 = torch.FloatTensor([0.02, 0.02, 0.03, 0.2, 0.2, 0.3, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03])
    lin_weight_2 = torch.FloatTensor([0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.2, 0.2, 0.3, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03])
    lin_weight_3 = torch.FloatTensor([0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.2, 0.2, 0.3, 0.02, 0.02, 0.03])
    lin_weight_4 = torch.FloatTensor([0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.2, 0.2, 0.3])
    lin_weight = torch.stack((lin_weight, lin_weight_1, lin_weight_2, lin_weight_3, lin_weight_4), dim=0)
    sd['lin.weight'] = lin_weight
    sd['lin.bias'] = torch.FloatTensor([-0.0447, -0.0447, -0.0447, -0.0447, -0.0447])
    
    model.load_state_dict(sd)
    model = train_reranker(traindataset, testdataset, pipeline, scorer, model, pipeline_type, args, 1)
    model.save(torch.path(os.path.join(os.getcwd(), "reranking/unimodal_mask_all.pth")))

    
    
