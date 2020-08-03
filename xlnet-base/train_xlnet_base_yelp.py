"""Modified finetuning script for XLNet base model on the IMDB dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW, XLNetForSequenceClassification, XLNetTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample():
    """
    A single training/test example for simple sequence classification.

    Attributes
    ----------
    guid: str
        Unique id for this specific example
    text_a: str
        Untokenized text of the string sequence.  In this case, text_a represents
        the movie review from the IMDB dataset.
    label: int
        Determines whether a review was classified as "good" or "bad".
        Should be specified for training and dev examples but not for testing examples.
    """

    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    """
    Contains the features of a single example's data.

    Attributes
    ----------
    input_ids: torch.tensor(max_seq_length), dtype=torch.int64
        Padded input tensor to pass to the model.
        Ids corresponding to the tokenized input sentence followed by padding.
    input_mask: torch.tensor(max_seq_length), dtype=torch.int64
        Attention mask for input ids. 1's for the length of the input_ids,
        and 0's for all of the padding inputs.
    segment_ids: torch.tensor(max_seq_length), dtype=torch.int64
        Token_type_ids to pass to the model. For IMDB, torch.zeros(max_seq_length)
    label_id: int
        Ground truth label for this example.  For IMDB, 0 for negative review and
        1 for positive review.
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor():
    """
    Base class data converters for sequence classification data sets.
    """

    def get_train_examples(self, data_dir):
        """
        Get a collection of `InputExample`s for the train set.
        
        Parameters
        ----------
        data_dir: str
            Path to the directory containing sequence classification dataset.
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Get a collection of `InputExample`s for the dev set.

        Parameters
        ----------
        data_dir: str
            Path to the directory containing sequence classification dataset.
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Get a list of possible output labels for the dataset.
        """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a tab separated value file.  Doesn't appear to be needed by IMDB dataset.

        Parameters
        ----------
        input_file: str
            Path to tab separated file to read.
        quotechar: str
            Character used to quote fields containing special characters.
        """
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class YELPProcessor(DataProcessor):
    """Processor for the YELP data set. Performs no truncation of sequence strings."""

    def get_train_examples(self, data_dir, data_num=None):
        """Read training CSV and create InputExamples for those values."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), delimiter=",", names=["sentiment", "sequence"])
        train_data["sentiment"] = train_data["sentiment"] - 1 # change sentiments to 0 or 1
        return self._create_examples(train_data, "train", data_num=data_num)

    def get_dev_examples(self, data_dir, data_num=None):
        """Read testing CSV and create InputExamples for those values."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), delimiter=",", names=["sentiment", "sequence"])
        dev_data["sentiment"] = dev_data["sentiment"] - 1 # change sentiments to 0 or 1
        return self._create_examples(dev_data, "dev", data_num=data_num)

    def get_labels(self):
        """Get list of [0, 1] because those are the possible output labels for this dataset"""
        return ["0","1"]

    def _create_examples(self, lines, set_type, data_num=None):
        """
        Create InputExamples from the given dataset.

        Parameters
        ----------
        lines: pd.DataFrame
            The read lines from the training or testing CSV.
        set_type: str
            Identifies if this is a train or test dataset for the unique id.
            Usually one of either "train" or "dev".
            Defines whether this is a training or testing dataset.
        data_num: int
            Defines maximum number of data examples to use from both training and testing.
            Does not need to be defined.
        """
        examples = []
        for (i, line) in enumerate(lines.values):
            guid = "%s-%s" % (set_type, i)
            text_a = str(line[1])
            label = str(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def _clip_grad_norm(params, max_grad_norm):
    """
    Clip the gradient function if it gets too large.  Replaces a deprecated clip_grad_norm_function.

    Parameters
    ----------
    params: list
        List of parameters to clip the gradient.
    max_grad_norm:
        The maximum gradient to not clip. Typically 1.0
    """
    clip_fn = torch.nn.utils.clip_grad_norm_
    for p in params:
        if isinstance(p, dict):
            clip_fn(p['params'], max_grad_norm)
        else:
            clip_fn(p, max_grad_norm)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """
    Tokenize and group InputExamples into a list of `InputBatch`s with attention masks and token_type_ids.

    Parameters
    ----------
    examples: list(InputExample)
        List of all of the InputExamples for the training or testing set.
    label_list: list(int)
        List of the possible label types for examples. [0, 1]
    max_seq_length: int
        Maximum sequence length for the created inputs. Typically 512.
    tokenizer: tokenizer
        Instance of the XLNetTokenizer.

    Returns
    -------
    features: dict
        Keys
        ----
        input_ids: torch.tensor(num_examples, max_seq_length), dtype=torch.int64
            Tokenized sequence text.
        token_type_ids: torch.tensor(num_examples, max_seq_length), dtype=torch.int64
            Token type ids for the inputs.
        attention_mask: torch.tensor(num_examples, max_seq_length), dtype=torch.int64
            Masking tensor for the inputs.
        labels: torch.tensor(num_examples), dtype=torch.int64
            Tensor of labels for each of the examples.
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    all_labels = [label_map[example.label] for example in examples]
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    all_example_text = [example.text_a for example in examples]

    # even though XLNet has no max length, the paper suggests using 512
    features = tokenizer(all_example_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    features['labels'] = labels_tensor
    return features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)


def get_analytics_pos_sent(outputs, labels):
    """
    Calculate precision, F1, and recall assuming positive sentiment with a 1 output is a TP.

    Parameters
    ----------
    outputs: torch.tensor([num_examples, 2]), dtype=torch.float32
        The logits output from the sentiment analysis model.
    labels: torch.tensor(num_examples)
        The true labels according to the dataset.

    Returns
    -------
    precision: torch.tensor(1), dtype=torch.float32
        The precision score assuming a positive sentiment is a true positive.
    recall: torch.tensor(1), dtype=torch.float32
        The recall score assuming a positive sentiment is a true positive.
    f1: torch.tensor(1), dtype=torch.float32
        The f1 score assuming a positive sentiment is a true positive.
    """
    tp = np.sum(outputs*labels==1)
    tn = np.sum((1-outputs)*(1-labels))
    fp = np.sum(outputs*(1-labels))
    fn = np.sum((1-outputs)*labels)
    precision = np.nan_to_num(tp/(tp+fp).astype(np.float32), nan=0.0)
    recall = np.nan_to_num(tp/(tp+fn).astype(np.float32), nan=0.0)
    f1 = np.nan_to_num(2*(recall*precision)/(recall+precision), nan=0.0)
    return precision, recall, f1


def get_analytics_neg_sent(outputs, labels):
    """
    Calculate precision, F1, and recall assuming negative sentiment with a 0 output is a TP.
    
    Parameters
    ----------
    outputs: torch.tensor([num_examples, 2]), dtype=torch.float32
        The logits output from the sentiment analysis model.
    labels: torch.tensor(num_examples)
        The true labels according to the dataset.

    Returns
    -------
    precision: torch.tensor(1), dtype=torch.float32
        The precision score assuming a negative sentiment is a true positive.
    recall: torch.tensor(1), dtype=torch.float32
        The recall score assuming a negative sentiment is a true positive.
    f1: torch.tensor(1), dtype=torch.float32
        The f1 score assuming a negative sentiment is a true positive.
    """
    tp = np.sum((1-outputs)*(1-labels)==1)
    tn = np.sum(outputs*labels==1)
    fp = np.sum((1-outputs)*labels)
    fn = np.sum((outputs)*(1-labels))
    precision = np.nan_to_num(tp/(tp+fp).astype(np.float32), nan=0.0)
    recall = np.nan_to_num(tp/(tp+fn).astype(np.float32), nan=0.0)
    f1 = np.nan_to_num(2*(recall*precision)/(recall+precision), nan=0.0)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--discr",
                        default=False,
                        action='store_true',
                        help="Whether to do discriminative fine-tuning.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--layers',
                        type=int,
                        nargs='+',
                        default=[-2],
                        help="choose the layers that used for downstream tasks, "
                             "-2 means use pooled output, -1 means all layer,"
                             "else means the detail layers. default is -2")
    parser.add_argument('--num_datas',
                        default=None,
                        type=int,
                        help="the number of data examples")
    parser.add_argument('--num_test_datas',
                        default=None,
                        type=int,
                        help="the number of data examples"
                        )
    parser.add_argument('--pooling_type',
                        default=None,
                        type=str,
                        choices=[None, 'mean', 'max'])
    args = parser.parse_args()

    processors = {
        "yelp": YELPProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    summary_writer = SummaryWriter(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, data_num=args.num_datas)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)

    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")

    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay= ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_parameters,
                         lr=args.learning_rate,
                         correct_bias=False)


    global_step = 0
    global_train_step = 0

    eval_examples = processor.get_dev_examples(args.data_dir, data_num=args.num_test_datas)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = eval_features['input_ids']
    all_input_mask = eval_features['attention_mask']
    all_segment_ids = eval_features['token_type_ids']
    all_label_ids = eval_features['labels']

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = train_features['input_ids']
        all_input_mask = train_features['attention_mask']
        all_segment_ids = train_features['token_type_ids']
        all_label_ids = train_features['labels']

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        print("TOTAL STEPS: ", (len(train_dataloader)*int(args.num_train_epochs)))

        epoch=0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch+=1
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, token_type_ids, label_ids = batch
                # print("Input ids shape:", input_ids.shape)
                # print("Input mask shape:", input_mask.shape)
                # print("Tok type Ids shape:", segment_ids.shape)
                # print("Labels shape:", label_ids.shape)
                loss, _ = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    # scheduler.step()

                    summary_writer.add_scalar('Loss/train', loss.item(), global_step)

                    # possibly comment this out
                    max_grad_norm = 1.0
                    _clip_grad_norm(optimizer_parameters, max_grad_norm)
                    model.zero_grad()
                    global_step += 1

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            pos_eval_prec, pos_eval_recall, pos_eval_f1 = 0, 0, 0
            neg_eval_prec, neg_eval_recall, neg_eval_f1 = 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_ep"+str(epoch)+".txt"),"w") as f:
                for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluate"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.detach().to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output)+"\n")
                    tmp_eval_accuracy=np.sum(outputs == label_ids)
                    tmp_eval_prec, tmp_eval_recall, tmp_eval_f1 = get_analytics_neg_sent(outputs, label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy
                    neg_eval_prec += tmp_eval_prec
                    neg_eval_recall += tmp_eval_recall
                    neg_eval_f1 += tmp_eval_f1

                    tmp_eval_prec, tmp_eval_recall, tmp_eval_f1 = get_analytics_pos_sent(outputs, label_ids)
                    pos_eval_prec += tmp_eval_prec
                    pos_eval_recall += tmp_eval_recall
                    pos_eval_f1 += tmp_eval_f1

                    global_train_step += 1

                    summary_writer.add_scalar("Loss/test", tmp_eval_loss.mean().item(), global_train_step)
                    summary_writer.add_scalar("Accuracy/test", tmp_eval_accuracy, global_train_step)

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            pos_eval_prec = pos_eval_prec / nb_eval_steps
            pos_eval_recall = pos_eval_recall / nb_eval_steps
            pos_eval_f1 = pos_eval_f1 / nb_eval_steps
            
            neg_eval_prec = neg_eval_prec / nb_eval_steps
            neg_eval_recall = neg_eval_recall / nb_eval_steps
            neg_eval_f1 = neg_eval_f1 / nb_eval_steps

            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': tr_loss/nb_tr_steps,
                      'pos_eval_precision': pos_eval_prec,
                      'neg_eval_precision': neg_eval_prec,
                      'pos_eval_recall': pos_eval_recall,
                      'neg_eval_recall': neg_eval_recall,
                      'pos_eval_f1': pos_eval_f1,
                      'neg_eval_f1': neg_eval_f1}

            summary_writer.add_scalar("Epoch_loss/train", tr_loss, epoch)
            summary_writer.add_scalar("Epoch_loss/test", eval_loss, epoch)
            summary_writer.add_scalar("Epoch_accuracy/test", eval_accuracy, epoch)

            summary_writer.add_scalar("Epoch_positive_precision/test", pos_eval_prec, epoch)
            summary_writer.add_scalar("Epoch_negative_precision/test", neg_eval_prec, epoch)

            summary_writer.add_scalar("Epoch_positive_recall/test", pos_eval_recall, epoch)
            summary_writer.add_scalar("Epoch_negative_recall/test", neg_eval_recall, epoch)

            summary_writer.add_scalar("Epoch_positive_f1/test", pos_eval_f1, epoch)
            summary_writer.add_scalar("Epoch_negative_f1/test", neg_eval_f1, epoch)

            output_eval_file = os.path.join(args.output_dir, "eval_results_ep"+str(epoch)+".txt")
            print("output_eval_file=",output_eval_file)
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            print("Saving model")
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, "imdb-finetuned-xlnet-model_"+str(epoch)+".pth"))


if __name__ == "__main__":
    main()
