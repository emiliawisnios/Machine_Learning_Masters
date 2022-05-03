# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

from run import evaluate_places
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
args = argp.parse_args()

preds = []
for line in open(args.eval_copus_path, 'r'):
    preds.append('London')
total, correct = evaluate_places(args.eval_copus_path, preds)

print('London baseline accuracy: %.2f' % (correct / total))
