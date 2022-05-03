"""Main run function.

Copyright: Stanford CS224 2019 class.
"""
import argparse
import random

import torch
from tqdm import tqdm
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import dataset
from mingptdemo.mingpt import model, utils
import trainer


def evaluate_places(filepath, predicted_places):
    """ Computes percent of correctly predicted birth places.

  Arguments:
    filepath: path to a file with our name, birth place data.
    predicted_places: a list of strings representing the 
        predicted birth place of each person.

  Returns: 
    (total, correct), floats
  """
    with open(filepath) as fin:
        lines = [x.strip().split('\t') for x in fin]
        if len(lines[0]) == 1:
            print('No gold birth places provided; returning (0,0)')
            return (0, 0)
        true_places = [x[1] for x in lines]
        total = len(true_places)
        assert total == len(predicted_places)
        correct = len(list(filter(lambda x: x[0] == x[1],
                                  zip(true_places, predicted_places))))
        return (float(total), float(correct))


argp = argparse.ArgumentParser()
argp.add_argument('function',
                  help="Whether to pretrain, finetune or evaluate a model",
                  choices=["pretrain", "finetune", "evaluate"])
argp.add_argument('variant',
                  help="Which variant of the model to run ('vanilla' or 'synthesizer')",
                  choices=["vanilla", "synthesizer"])
argp.add_argument('pretrain_corpus_path',
                  help="Path of the corpus to pretrain on", default=None)
argp.add_argument('--reading_params_path',
                  help="If specified, path of the model to load before finetuning/evaluation",
                  default=None)
argp.add_argument('--writing_params_path',
                  help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--finetune_corpus_path',
                  help="Path of the corpus to finetune on", default=None)
argp.add_argument('--eval_corpus_path',
                  help="Path of the corpus to evaluate on", default=None)
argp.add_argument('--outputs_path', default=None)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the 
# vocab from the pretraining corpus.)
block_size = 128
text = open(args.pretrain_corpus_path).read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the synthesizer models
mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
                        n_layer=4, n_head=8, n_embd=256)

"""
Don't change above here; write your code below
"""
# TODO [part c]: Define some model here

if args.variant == 'vanilla':
    gpt_model = model.GPT(mconf)
elif args.variant == 'synthesizer':
    gpt_model = model.GPT(mconf)
else:
    raise ValueError('Invalid variant: {}'.format(args.variant))

# From here on, your code should be identical independent of which
# variant (vanilla or synthesizer) has been chosen.

if args.function == 'pretrain':
    assert args.pretrain_corpus_path is not None
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters for pretraining:
    #     max_epochs=650
    #     batch_size=128
    #     learning_rate=6e-3
    #     lr_decay=True
    #     warmup_tokens=512*20
    #     final_tokens=200*len(pretrain_dataset)*block_size
    #     num_workers=4
    tconf = trainer.TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=6e-3,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=200 * len(pretrain_dataset) * block_size,
        num_workers=4
    )
    gpt_trainer = trainer.Trainer(gpt_model, pretrain_dataset, None, tconf)
    gpt_trainer.train()
    torch.save(gpt_model.state_dict(), args.writing_params_path)

elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part c] [part f]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #     Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    if args.reading_params_path is None:
        print('No pretrained model specified, finetuning without a pretrained model')
        tconf = trainer.TrainerConfig(
            max_epochs=75,
            batch_size=256,
            learning_rate=6e-4,
            lr_decay=True,
            warmup_tokens=512 * 20,
            final_tokens=200 * len(pretrain_dataset) * block_size,
            num_workers=4)
    else:
        print('Pretrained model specified, finetuning with a pretrained model')
        gpt_model.load_state_dict(torch.load(args.reading_params_path))
        gpt_model = gpt_model.to(device)
        tconf = trainer.TrainerConfig(
            max_epochs=10,
            batch_size=256,
            learning_rate=6e-4,
            lr_decay=True,
            warmup_tokens=512 * 20,
            final_tokens=200 * len(pretrain_dataset) * block_size,
            num_workers=4)

    finetune_text = open(args.finetune_corpus_path).read()
    finetune_dataset = dataset.NameDataset(pretrain_dataset, finetune_text)
    gpt_trainer = trainer.Trainer(gpt_model, finetune_dataset, None, tconf)
    gpt_trainer.train()
    torch.save(gpt_model.state_dict(), args.writing_params_path)

elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    # gpt_model.load_state_dict(torch.load(args.reading_params_path, map_location=torch.device('cpu')))
    gpt_model.load_state_dict(torch.load(args.reading_params_path))
    gpt_model = gpt_model.to(device)
    # gpt_model.eval()
    correct = 0
    total = 0
    with open(args.outputs_path, 'w') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path)):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
            pred = utils.sample(gpt_model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
    else:
        print('Predictions written to {}; no targets provided'
              .format(args.outputs_path))
