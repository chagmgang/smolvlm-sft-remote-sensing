from PIL import Image
import random
import torchvision
import numpy as np
import os
from datasets import load_dataset
import torch
from PIL import Image
import random
from transformers import SmolVLMForConditionalGeneration, Idefics3Processor
from transformers import Trainer, TrainingArguments
import torch

import os

from datasets import load_from_disk, load_dataset
from PIL import Image

import argparse
import torch
import deepspeed

parser = argparse.ArgumentParser(
    description='Deepspeed gpu benchmark for MistralForVisionLLM')
parser.add_argument('--local-rank', '--local_rank', default=-1, type=int)
parser.add_argument('--world-size', default=-1, type=int)
parser.add_argument('--per-device-train-batch-size', default=4, type=int)
parser.add_argument('--weight-decay', default=0.0001, type=float)
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--gradient-checkpointing', action='store_true')
parser.add_argument('--model-path')
parser = deepspeed.add_config_arguments(parser)


def calc_num_options(class_name, candidates):
    assert class_name in candidates

    selected_idx = np.arange(len(candidates))
    np.random.shuffle(selected_idx)

    num_options = random.randint(3, len(candidates))
    selected_idx = selected_idx[:num_options]
    
    selected_candidates = list()
    for idx in selected_idx:
        selected_candidates.append(candidates[idx])

    selected_candidates.append(class_name)
    selected_candidates = list(set(selected_candidates))
    np.random.shuffle(selected_candidates)

    prob = random.random()
    if prob < float(1 / len(candidates)): # no option
        selected_candidates = [s for s in selected_candidates if s != class_name]
        class_name = 'no option'
        rand_index = random.randint(0, len(selected_candidates))
        selected_candidates.insert(rand_index, class_name)
    else:
        class_name = class_name
        selected_candidates = selected_candidates
    return class_name, selected_candidates


def make_template(candidates, class_name):

    templates = ''
    for idx, candidate in enumerate(candidates):
        template = f"<|reserved_special_token_{idx}|>"
        template += '\n'
        template += candidate
        if idx != len(candidates) - 1:
            template += '\n'
        templates += template

    index = candidates.index(class_name)
    label_token = f"<|reserved_special_token_{index}|>"
    return templates, label_token
    

class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        prefix,
        dataset,
        transforms,
    ):
        self.prefix = prefix
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        filename = os.path.join(self.prefix, data['images'])
        image = Image.open(filename)
        image = self.transforms(image)

        class_name, candidates = calc_num_options(data['class'], data['candidates'])
        human, assistant = make_template(candidates, class_name)

        chats = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': 'You are an expert whose job is to analysis remote sensing imagery. Based on the images, choose the most appropriate category from the following.'},
                    {'type': 'text', 'text': human},
                ],
            },
            {
                'role': 'Assistant',
                'content': [
                    {'type': 'text', 'text': assistant},
                ],
            },
        ]

        return image, chats


class CollateFn(object):

    def __init__(
        self,
        processor,
        max_tokens: int = -1,
    ):
        self.processor = processor
        self.max_tokens = max_tokens

    def __call__(self, batch):
        batch = list(zip(*batch))
        image_files, chats = batch

        class_names = list()
        for chat in chats:
            class_names.append(chat[-1]['content'][0]['text'])

        class_ids = [self.processor.tokenizer.convert_tokens_to_ids(c) for c in class_names]

        texts = list()        
        for chat in chats:
            text = self.processor.apply_chat_template(chat, tokenize=False)
            texts.append(text)

        images = list()
        for image_file in image_files:
            images.append(image_file)

        batch = self.processor(text=texts, images=images, return_tensors='pt', padding=True)
        labels = batch['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            str(self.processor.image_token))
        labels[labels == image_token_id] = -100

        batch['labels'] = labels

        return batch

def main():

    args = parser.parse_args()

    model = SmolVLMForConditionalGeneration.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')
    processor = Idefics3Processor.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ])

    dataset_loaded = load_from_disk('/nas/k8s/dev/mlops/chagmgang/FGSC')
    dataset_config = {
        'FGSCM_52': {
            'prefix': '/nas/k8s/dev/mlops/vl-data',
            'multi': 10,
        },
        'FGSCR_42': {
            'prefix': '/nas/k8s/dev/mlops/vl-data',
            'multi': 1,
        },
        'AiRound': {
            'prefix': '/nas/k8s/dev/mlops/vl-data/airound/aerial',
            'multi': 10,
        },
        'CV_BrCT': {
            'prefix': '/nas/k8s/dev/mlops/vl-data/cvbrct/aerial',
            'multi': 10,
        },
        'EuroSAT': {
            'prefix': '/nas/k8s/dev/mlops/vl-data/eurosat/2750',
            'multi': 4,
        },
        'OPTINAL_31': {
            'prefix': '/nas/k8s/dev/mlops/vl-data/optimal-31/OPTIMAL-31/Images',
            'multi': 50,
        },
        'RESISC': {
            'prefix': '/nas/Dataset/NWPU-RESISC45/nwpu-rsisc45/NWPU-RESISC45',
            'multi': 3,
        },
        'RSI_CB128': {
            'prefix': '/nas/Dataset/rsi-cb128/RSI-CB128',
            'multi': 3,
        },
        'UC_Merced': {
            'prefix': '/nas/k8s/dev/mlops/vl-data/UCMerced-Landuse/UCMerced_LandUse/Images',
            'multi': 50,
        },
        'WHU_RS19': {
            'prefix': '/nas/Dataset/WHU-RS19/WHU-RS19',
            'multi': 100,
        },
        'mlrsnet': {
            'prefix': '/nas/k8s/dev/mlops/vl-data/mlrsnet/Images',
            'multi': 1,
        },
    }
    
    dataset_list = list()
    for key in dataset_config.keys():
        for _ in range(dataset_config[key]['multi']):
            dataset = ClassificationDataset(
                prefix=dataset_config[key]['prefix'],
                dataset=dataset_loaded[key],
                transforms=transforms,
            )
            dataset_list.append(dataset)
    
    dataset = torch.utils.data.ConcatDataset(dataset_list)

    collate_fn = CollateFn(processor=processor)

    training_args = TrainingArguments(
        output_dir='smolvlm-256m-tuning/model',
        logging_dir='smolvlm-256m-tuning/logs',
        logging_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        weight_decay=0.05,
        learning_rate=1e-4,
        save_strategy='steps',
        save_steps=500,
        warmup_steps=250,
        max_steps=15000,
        deepspeed=args.deepspeed_config,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            'use_reentrant': False,
        },
        report_to='tensorboard',
        do_train=True,
        bf16=True,
        dataloader_num_workers=4,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    trainer.train()

if __name__ == '__main__':
    main()
