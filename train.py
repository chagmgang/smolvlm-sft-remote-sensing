from transformers import SmolVLMForConditionalGeneration, Idefics3Processor
from transformers import Trainer, TrainingArguments
import torch

import os

from datasets import load_dataset
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



class DatasetFromHuggingfaceHub(torch.utils.data.Dataset):

    def __init__(
        self,
        prefix,
        dataset,
    ):

        self.prefix = prefix
        self.dataset = dataset

    def __getitem__(self, idx):
        data = self.dataset[idx]
        filename = data['filename']
        chats = data['chats']
        full_filename = os.path.join(self.prefix, filename)

        return full_filename, chats


    def __len__(self):
        return len(self.dataset)


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
        filenames, chats = batch
        assert len(filenames) == 1
        assert len(chats) == 1

        image = Image.open(filenames[0])
        prompt = self.processor.apply_chat_template(chats[0])
        encode = self.processor(
            text=[prompt],
            images=[[image]],
            return_tensors='pt',
        )

        encode['labels'] = encode['input_ids']
        return encode



def main():

    args = parser.parse_args()

    model = SmolVLMForConditionalGeneration.from_pretrained(args.model_path)

    processor = Idefics3Processor.from_pretrained(args.model_path)

    dataset_list = list()
    dataset_load = load_dataset('KevinCha/smolvlm-format-earthgpt')['train']
    dataset = DatasetFromHuggingfaceHub(
        prefix='/nas/k8s/dev/mlops/vl-data/MMRS-1M',
        dataset=dataset_load,
    )
    dataset_list.append(dataset)

    dataset_load = load_dataset('KevinCha/smolvlm-format-rsgpt-eval')['train']
    dataset = DatasetFromHuggingfaceHub(
        prefix='/nas/k8s/dev/mlops/vl-data/rsgpt/rsgpt_dataset/RSIEval/images/',
        dataset=dataset_load,
    )
    dataset_list.append(dataset)

    dataset_load = load_dataset('KevinCha/smolvlm-format-rsgpt-caption')['train']
    dataset = DatasetFromHuggingfaceHub(
        prefix='/nas/k8s/dev/mlops/vl-data/rsgpt/rsgpt_dataset/RSICap/images/',
        dataset=dataset_load,
    )

    collate_fn = CollateFn(processor=processor)

    training_args = TrainingArguments(
        output_dir='smolvlm-500m-tuning/model',
        logging_dir='smolvlm-500m-tuning/logs',
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        weight_decay=0.05,
        learning_rate=5e-5,
        save_strategy='steps',
        save_steps=500,
        warmup_steps=1000,
        max_steps=33000,
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
