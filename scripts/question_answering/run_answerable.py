"""
Question Answering with Pretrained Language Model
"""
# pylint:disable=redefined-outer-name,logging-format-interpolation

import os
import json
import time
import logging
import argparse
import functools
import collections
from multiprocessing import Pool, cpu_count

import mxnet as mx
import numpy as np
from mxnet.lr_scheduler import PolyScheduler

import gluonnlp.data.batchify as bf
from models import ModelForAnswerable
from eval_utils import squad_eval, answerable_eval
from squad_utils import SquadFeature, get_squad_examples, convert_squad_example_to_feature
from gluonnlp.models import get_backbone
from gluonnlp.utils.misc import grouper, repeat, set_seed, parse_ctx, logging_config, count_parameters
from gluonnlp.initializer import TruncNorm
from gluonnlp.utils.parameter import clip_grad_global_norm

mx.npx.set_np()

CACHE_PATH = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', 'cached'))
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Question Answering example. '
                    'We fine-tune the pretrained model on SQuAD dataset.')
    parser.add_argument('--model_name', type=str, default='google_albert_base_v2',
                        help='Name of the pretrained model.')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model')
    parser.add_argument('--data_dir', type=str, default='squad')
    parser.add_argument('--version', default='2.0', choices=['1.1', '2.0'],
                        help='Version of the SQuAD dataset.')
    parser.add_argument('--output_dir', type=str, default='squad_out',
                        help='The output directory where the model params will be written.'
                             ' default is squad_out')
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100, help='The logging interval.')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='the number of steps to save model parameters.'
                             'default is every epoch')
    parser.add_argument('--epochs', type=float, default=3.0,
                        help='Number of epochs, default is 3')
    parser.add_argument('--num_train_steps', type=int, default=None,
                        help='The number of training steps. Note that epochs will be ignored '
                             'if training steps are set')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size. Number of examples per gpu in a minibatch. default is 32')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluate batch size. Number of examples per gpu in a minibatch for '
                             'evaluation.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm.')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimization algorithm. default is adamw')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon of AdamW optimizer')
    parser.add_argument('--adam_betas', default='(0.9, 0.999)', metavar='B',
                        help='betas for Adam optimizer')
    parser.add_argument('--num_accumulated', type=int, default=1,
                        help='The number of batches for gradients accumulation to '
                             'simulate large batch size.')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Initial learning rate. default is 2e-5')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Ratio of warmup steps in the learning rate scheduler.')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='warmup steps. Note that either warmup_steps or warmup_ratio is set.')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--layerwise_decay', type=float, default=-1, help='Layer-wise lr decay')
    parser.add_argument('--untunable_depth', type=float, default=-1,
                        help='Depth of untunable parameters')
    parser.add_argument('--classifier_dropout', type=float, default=0.1,
                        help='dropout of classifier')
    # Data pre/post processing
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='The maximum total input sequence length after tokenization.'
                             'Sequences longer than this will be truncated, and sequences shorter '
                             'than this will be padded. default is 512')
    parser.add_argument('--doc_stride', type=int, default=128,
                        help='When splitting up a long document into chunks, how much stride to '
                             'take between chunks. default is 128')
    parser.add_argument('--max_query_length', type=int, default=64,
                        help='The maximum number of tokens for the query. Questions longer than '
                             'this will be truncated to this length. default is 64')
    parser.add_argument('--round_to', type=int, default=None,
                        help='The length of padded sequences will be rounded up to be multiple'
                             ' of this argument. When round to is set to 8, training throughput '
                             'may increase for mixed precision training on GPUs with TensorCores.')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Whether to overwrite the feature cache.')
    # Evaluation hyperparameters
    parser.add_argument('--start_top_n', type=int, default=5,
                        help='Number of start-position candidates')
    parser.add_argument('--end_top_n', type=int, default=5,
                        help='Number of end-position candidates corresponding '
                             'to a start position')
    parser.add_argument('--n_best_size', type=int, default=20, help='Top N results written to file')
    parser.add_argument('--max_answer_length', type=int, default=30,
                        help='The maximum length of an answer that can be generated. This is '
                             'needed because the start and end predictions are not conditioned '
                             'on one another. default is 30')
    parser.add_argument('--param_checkpoint', type=str, default=None,
                        help='The parameter checkpoint for evaluating the model')
    parser.add_argument('--backbone_path', type=str, default=None,
                        help='The parameter checkpoint of backbone model')
    parser.add_argument('--all_evaluate', action='store_true',
                        help='Whether to evaluate all intermediate checkpoints '
                             'instead of only last one')
    parser.add_argument('--max_saved_ckpt', type=int, default=10,
                        help='The maximum number of saved checkpoints')
    args = parser.parse_args()
    return args


class SquadDatasetProcessor:

    def __init__(self, tokenizer, doc_stride, max_seq_length, max_query_length):
        """

        Parameters
        ----------
        tokenizer
            The tokenizer
        doc_stride
            The stride to chunk the document
        max_seq_length
            Maximum length of the merged data
        max_query_length
            Maximum query length
        """
        self._tokenizer = tokenizer
        self._doc_stride = doc_stride
        self._max_seq_length = max_seq_length
        self._max_query_length = max_query_length

        vocab = tokenizer.vocab
        self.pad_id = vocab.pad_id
        # For roberta model, taking sepecial token <s> as [CLS] and </s> as [SEP]
        self.cls_id = vocab.bos_id if 'cls_token' not in vocab.special_token_keys else vocab.cls_id
        self.sep_id = vocab.eos_id if 'sep_token' not in vocab.special_token_keys else vocab.sep_id

        # TODO(sxjscience) Consider to combine the NamedTuple and batchify functionality.
        self.ChunkFeature = collections.namedtuple('ChunkFeature',
                                                   ['qas_id',
                                                    'data',
                                                    'valid_length',
                                                    'segment_ids',
                                                    'masks',
                                                    'answer_masks',
                                                    'is_impossible',
                                                    'gt_start',
                                                    'gt_end',
                                                    'context_offset',
                                                    'chunk_start',
                                                    'chunk_length'])
        self.BatchifyFunction = bf.NamedTuple(self.ChunkFeature,
                                              {'qas_id': bf.List(),
                                               'data': bf.Pad(val=self.pad_id),
                                               'valid_length': bf.Stack(),
                                               'segment_ids': bf.Pad(),
                                               'masks': bf.Pad(val=1),
                                               'answer_masks': bf.Pad(val=0),
                                               'is_impossible': bf.Stack(),
                                               'gt_start': bf.Stack(),
                                               'gt_end': bf.Stack(),
                                               'context_offset': bf.Stack(),
                                               'chunk_start': bf.Stack(),
                                               'chunk_length': bf.Stack()})

    def process_sample(self, feature: SquadFeature):
        """Process the data to the following format.

        Note that we mask all the special tokens except the CLS token. The reason for not masking
        the CLS token is that if the question is not answerable, we will set the start and end to
        be 0.


        Merged:      <CLS> Question <SEP> Context <SEP>
        Segment IDs:  0       0       0      1      1
        Mask:         0       1       1      0      1

        Here, we need to emphasize that when mask = 1, the data are actually not masked!

        Parameters
        ----------
        feature
            Tokenized SQuAD feature

        Returns
        -------
        ret
            Divide the feature into multiple chunks and extract the feature which contains
            the following:
            - data
                The data that concatenates the query and the context + special tokens
            - valid_length
                The valid_length of the data
            - segment_ids
                We assign the query part as segment 0 and the context part as segment 1.
            - masks
                We mask all the special tokens. 1 --> not masked, 0 --> masked.
            - is_impossible
                Whether the provided context is impossible to answer or not.
            - gt_start
                The ground-truth start location of the span
            - gt_end
                The ground-truth end location of the span
            - chunk_start
                The start of the chunk
            - chunk_length
                The length of the chunk
        """
        ret = []
        truncated_query_ids = feature.query_token_ids[:self._max_query_length]
        chunks = feature.get_chunks(
            doc_stride=self._doc_stride,
            max_chunk_length=self._max_seq_length - len(truncated_query_ids) - 3)
        for chunk in chunks:
            data = np.array([self._tokenizer.vocab.cls_id] + truncated_query_ids +
                            [self._tokenizer.vocab.sep_id] +
                            feature.context_token_ids[chunk.start:(chunk.start + chunk.length)] +
                            [self._tokenizer.vocab.sep_id], dtype=np.int32)
            valid_length = len(data)
            segment_ids = np.array([0] + [0] * len(truncated_query_ids) +
                                   [0] + [1] * chunk.length + [1], dtype=np.int32)
            chunk_masks = [0] * chunk.length
            masks = np.array([0] + [1] * len(truncated_query_ids) + [1] + chunk_masks + [1],
                             dtype=np.int32)
            context_offset = len(truncated_query_ids) + 2
            plau_chunk_masks = chunk_masks.copy()

            if chunk.start_pos is not None and chunk.end_pos is not None:
                # Here, we increase the start and end because we put query before context
                start_pos = chunk.start_pos + context_offset
                end_pos = chunk.end_pos + context_offset
                # for answer_masks 1-> not mask, 0 -> mask
                plau_chunk_masks[chunk.start_pos:chunk.end_pos + 1] = [1] * (end_pos - start_pos + 1)
                answer_masks = np.array([1] + [1] * len(truncated_query_ids) + [0] + plau_chunk_masks + [0],
                                 dtype=np.int32)
            else:
                start_pos = 0
                end_pos = 0
                answer_masks = np.array([1] + [1] * len(truncated_query_ids) + [0] + chunk_masks + [0],
                                 dtype=np.int32)
            is_impossible = feature.is_impossible or chunk.is_impossible
            if is_impossible:
                start_pos = 0
                end_pos = 0

            chunk_feature = self.ChunkFeature(qas_id=feature.qas_id,
                                              data=data,
                                              valid_length=valid_length,
                                              segment_ids=segment_ids,
                                              masks=masks,
                                              answer_masks=answer_masks,
                                              is_impossible=is_impossible,
                                              gt_start=start_pos,
                                              gt_end=end_pos,
                                              context_offset=context_offset,
                                              chunk_start=chunk.start,
                                              chunk_length=chunk.length)
            ret.append(chunk_feature)
        return ret

    def get_train(self, features, skip_unreliable=True):
        """Get the training dataset

        Parameters
        ----------
        features
        skip_unreliable
            Whether to skip the unreliable spans in the training set

        Returns
        -------
        train_dataset
        num_token_answer_mismatch
        num_unreliable
        """
        train_dataset = []
        num_token_answer_mismatch = 0
        num_unreliable = 0
        for feature in features:
            if feature.token_answer_mismatch:
                num_token_answer_mismatch += 1
            if feature.unreliable_span:
                num_unreliable += 1
            if skip_unreliable and feature.unreliable_span:
                # Skip when not reliable
                continue
            # Process the feature
            chunk_features = self.process_sample(feature)
            train_dataset.extend(chunk_features)
        return train_dataset, num_token_answer_mismatch, num_unreliable


def get_network(model_name,
                ctx_l,
                dropout=0.1,
                checkpoint_path=None,
                backbone_path=None):
    """
    Get the network that fine-tune the Question Answering Task

    Parameters
    ----------
    model_name : str
        The model name of the backbone model
    ctx_l :
        Context list of training device like [mx.gpu(0), mx.gpu(1)]
    dropout : float
        Dropout probability of the task specified layer
    checkpoint_path: str
        Path to a Fine-tuned checkpoint
    backbone_path: str
        Path to the backbone model to be loaded in qa_net

    Returns
    -------
    cfg
    tokenizer
    qa_net
    use_segmentation
    """
    # Create the network
    use_segmentation = 'roberta' not in model_name and 'xlmr' not in model_name
    Model, cfg, tokenizer, download_params_path, _ = \
        get_backbone(model_name, load_backbone=not backbone_path)
    backbone = Model.from_cfg(cfg, use_pooler=False)
    # Load local backbone parameters if backbone_path provided.
    # Otherwise, download backbone parameters from gluon zoo.

    backbone_params_path = backbone_path if backbone_path else download_params_path
    if checkpoint_path is None:
        # TODO(zheyuye), be careful of allow_missing that used to pass the mlm parameters in roberta
        backbone.load_parameters(
            backbone_params_path,
            ignore_extra=True,
            allow_missing=True,
            ctx=ctx_l)
        num_params, num_fixed_params = count_parameters(backbone.collect_params())
        logging.info(
            'Loading Backbone Model from {}, with total/fixd parameters={}/{}'.format(
                backbone_params_path, num_params, num_fixed_params))
    qa_net = ModelForAnswerable(backbone=backbone,
                                dropout_prob=dropout,
                                weight_initializer=TruncNorm(stdev=0.02),
                                use_segmentation=use_segmentation,
                                prefix='qa_net_')
    if checkpoint_path is None:
        # Ignore the UserWarning during initialization,
        # There is no need to re-initialize the parameters of backbone
        qa_net.initialize(ctx=ctx_l)
    else:
        qa_net.load_parameters(checkpoint_path, ctx=ctx_l, cast_dtype=True)
    qa_net.hybridize()

    return cfg, tokenizer, qa_net, use_segmentation


def untune_params(model, untunable_depth, not_included=[]):
    """Froze part of parameters according to layer depth.

    That is, make all layer that shallower than `untunable_depth` untunable
    to stop the gradient backward computation and accelerate the training.

    Parameters:
    ----------
    model
        qa_net
    untunable_depth: int
        the depth of the neural network starting from 1 to number of layers
    not_included: list of str
        A list or parameter names that not included in the untunable parameters
    """
    all_layers = model.backbone.encoder.all_encoder_layers
    for _, v in model.collect_params('.*embed*').items():
        model.grad_req = 'null'

    for layer in all_layers[:untunable_depth]:
        for key, value in layer.collect_params().items():
            for pn in not_included:
                if pn in key:
                    continue
            value.grad_req = 'null'

def train(args):
    ctx_l = parse_ctx(args.gpus)
    cfg, tokenizer, qa_net, use_segmentation = \
        get_network(args.model_name, ctx_l,
                    args.classifier_dropout,
                    args.param_checkpoint,
                    args.backbone_path)
    # Load the data
    train_examples = get_squad_examples(args.data_dir, segment='train', version=args.version)
    logging.info('Load data from {}, Version={}'.format(args.data_dir, args.version))
    num_process = min(cpu_count(), 8)
    train_cache_path = os.path.join(
        CACHE_PATH, 'train_{}_squad_{}.ndjson'.format(
            args.model_name, args.version))
    if os.path.exists(train_cache_path) and not args.overwrite_cache:
        train_features = []
        with open(train_cache_path, 'r') as f:
            for line in f:
                train_features.append(SquadFeature.from_json(line))
        logging.info('Found cached training features, load from {}'.format(train_cache_path))

    else:
        start = time.time()
        logging.info('Tokenize Training Data:')
        with Pool(num_process) as pool:
            train_features = pool.map(
                functools.partial(
                    convert_squad_example_to_feature,
                    tokenizer=tokenizer,
                    is_training=True),
                train_examples)
        logging.info('Done! Time spent:{:.2f} seconds'.format(time.time() - start))
        with open(train_cache_path, 'w') as f:
            for feature in train_features:
                f.write(feature.to_json() + '\n')

    dataset_processor = SquadDatasetProcessor(tokenizer=tokenizer,
                                              doc_stride=args.doc_stride,
                                              max_seq_length=args.max_seq_length,
                                              max_query_length=args.max_query_length)
    logging.info('Processing the Training data:')
    train_dataset, num_answer_mismatch, num_unreliable \
        = dataset_processor.get_train(train_features, skip_unreliable=True)
    logging.info('Done! #Unreliable Span={} / #Mismatched Answer={} / #Total={}'
                 .format(num_unreliable, num_answer_mismatch, len(train_features)))

    # Get dataset statistics
    num_impossible = 0
    for sample in train_dataset:
        num_impossible += sample.is_impossible
    logging.info('Before Chunking, #Train/Is Impossible = {}/{}'
                 .format(len(train_features),
                         sum([ele.is_impossible for ele in train_features])))
    logging.info('After Chunking, #Train Sample/Is Impossible = {}/{}'
                 .format(len(train_dataset), num_impossible))
    train_dataloader = mx.gluon.data.DataLoader(
        train_dataset,
        batchify_fn=dataset_processor.BatchifyFunction,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True)
    # Froze parameters
    if 'electra' in args.model_name:
        # does not work for albert model since parameters in all layers are shared
        if args.untunable_depth > 0:
            untune_params(qa_net, args.untunable_depth)
        if args.layerwise_decay > 0:
            qa_net.backbone.apply_layerwise_decay(args.layerwise_decay)

    # Do not apply weight decay to all the LayerNorm and bias
    for _, v in qa_net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in qa_net.collect_params().values() if p.grad_req != 'null']
    # Set grad_req if gradient accumulation is required
    if args.num_accumulated > 1:
        logging.info('Using gradient accumulation. Effective global batch size = {}'
                     .format(args.num_accumulated * args.batch_size * len(ctx_l)))
        for p in params:
            p.grad_req = 'add'
    epoch_size = (len(train_dataloader) + len(ctx_l) - 1) // len(ctx_l)
    if args.num_train_steps is not None:
        num_train_steps = args.num_train_steps
    else:
        num_train_steps = int(args.epochs * epoch_size / args.num_accumulated)
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(num_train_steps * args.warmup_ratio)
    assert warmup_steps is not None, 'Must specify either warmup_steps or warmup_ratio'
    log_interval = args.log_interval
    save_interval = args.save_interval if args.save_interval is not None\
        else epoch_size // args.num_accumulated
    logging.info('#Total Training Steps={}, Warmup={}, Save Interval={}'
                 .format(num_train_steps, warmup_steps, save_interval))

    # set up optimization
    lr_scheduler = PolyScheduler(max_update=num_train_steps,
                                 base_lr=args.lr,
                                 warmup_begin_lr=0,
                                 pwr=1,
                                 final_lr=0,
                                 warmup_steps=warmup_steps,
                                 warmup_mode='linear')
    optimizer_params = {'learning_rate': args.lr,
                        'wd': args.wd,
                        'lr_scheduler': lr_scheduler,
                        }
    adam_betas = eval(args.adam_betas)
    if args.optimizer == 'adamw':
        optimizer_params.update({'beta1': adam_betas[0],
                                 'beta2': adam_betas[1],
                                 'epsilon': args.adam_epsilon,
                                 'correct_bias': False,
                                 })
    elif args.optimizer == 'adam':
        optimizer_params.update({'beta1': adam_betas[0],
                                 'beta2': adam_betas[1],
                                 'epsilon': args.adam_epsilon,
                                 })
    trainer = mx.gluon.Trainer(qa_net.collect_params(),
                               args.optimizer, optimizer_params,
                               update_on_kvstore=False)
    num_samples_per_update = 0
    loss_denom = float(len(ctx_l) * args.num_accumulated)

    log_total_loss = 0
    log_sample_num = 0
    if args.num_accumulated != 1:
        # set grad to zero for gradient accumulation
        qa_net.collect_params().zero_grad()

    # start training
    global_tic = time.time()
    tic = time.time()
    for step_num, batch_data in enumerate(
            grouper(repeat(train_dataloader), len(ctx_l) * args.num_accumulated)):
        for sample_l in grouper(batch_data, len(ctx_l)):
            loss_l = []
            for sample, ctx in zip(sample_l, ctx_l):
                if sample is None:
                    continue
                # Copy the data to device
                tokens = sample.data.as_in_ctx(ctx)
                log_sample_num += len(tokens)
                num_samples_per_update += len(tokens)
                segment_ids = sample.segment_ids.as_in_ctx(ctx) if use_segmentation else None
                valid_length = sample.valid_length.as_in_ctx(ctx)
                p_mask = sample.masks.as_in_ctx(ctx)
                a_mask = sample.answer_masks.as_in_ctx(ctx)
                is_impossible = sample.is_impossible.as_in_ctx(ctx).astype(np.int32)
                batch_idx = mx.np.arange(tokens.shape[0], dtype=np.int32, ctx=ctx)
                p_mask = 1 - p_mask  # In the network, we use 1 --> no_mask, 0 --> mask
                with mx.autograd.record():
                    answerable_logits = qa_net(tokens, segment_ids, valid_length, p_mask, a_mask)
                    sel_answerable_logits = answerable_logits[batch_idx, is_impossible]
                    answerable_loss = - 0.5 * sel_answerable_logits.sum()
                    loss = answerable_loss / loss_denom
                    loss_l.append(loss)

            for loss in loss_l:
                loss.backward()
            # All Reduce the Step Loss
            log_total_loss += sum([ele.as_in_ctx(ctx_l[0])
                                   for ele in loss_l]).asnumpy() * loss_denom
        # update
        trainer.allreduce_grads()
        # Here, the accumulated gradients are
        # \sum_{n=1}^N g_n / loss_denom
        # Thus, in order to clip the average gradient
        #   \frac{1}{N} \sum_{n=1}^N      -->  clip to args.max_grad_norm
        # We need to change the ratio to be
        #  \sum_{n=1}^N g_n / loss_denom  -->  clip to args.max_grad_norm  * N / loss_denom
        total_norm, ratio, is_finite = clip_grad_global_norm(
            params, args.max_grad_norm * num_samples_per_update / loss_denom)
        total_norm = total_norm / (num_samples_per_update / loss_denom)

        trainer.update(num_samples_per_update / loss_denom, ignore_stale_grad=True)
        if args.num_accumulated != 1:
            # set grad to zero for gradient accumulation
            qa_net.collect_params().zero_grad()

        # saving
        if (step_num + 1) % save_interval == 0 or (step_num + 1) >= num_train_steps:
            version_prefix = 'squad' + args.version
            ckpt_name = '{}_{}_{}.params'.format(args.model_name,
                                                 version_prefix,
                                                 (step_num + 1))
            params_saved = os.path.join(args.output_dir, ckpt_name)
            qa_net.save_parameters(params_saved)
            ckpt_candidates = [
                f for f in os.listdir(
                    args.output_dir) if f.endswith('.params')]
            # keep last 10 checkpoints
            if len(ckpt_candidates) > args.max_saved_ckpt:
                ckpt_candidates.sort(key=lambda ele: (len(ele), ele))
                os.remove(os.path.join(args.output_dir, ckpt_candidates[0]))
            logging.info('Params saved in: {}'.format(params_saved))

        # logging
        if (step_num + 1) % log_interval == 0:
            log_total_loss /= log_sample_num
            toc = time.time()
            logging.info(
                'Step: {}/{}, Loss total={:.4f},'
                ' LR={:.8f}, grad_norm={:.4f}. Time cost={:.2f}, Throughput={:.2f} samples/s'
                ' ETA={:.2f}h'.format((step_num + 1), num_train_steps, log_total_loss, trainer.learning_rate, total_norm,
                                      toc - tic, log_sample_num / (toc - tic),
                                      (num_train_steps - (step_num + 1)) / ((step_num + 1) / (toc - global_tic)) / 3600))
            tic = time.time()
            log_total_loss = 0
            log_sample_num = 0
            num_samples_per_update = 0

        if (step_num + 1) >= num_train_steps:
            logging.info('Finish training step: %d', (step_num + 1))
            break

    return params_saved


RawResultExtended = collections.namedtuple(
    'RawResultExtended',
    ['qas_id',
     'answerable_logits'])


def predict_extended(original_feature,
                     chunked_features,
                     results,
                     n_best_size,
                     max_answer_length=64,
                     start_top_n=5,
                     end_top_n=5):
    """Get prediction results for SQuAD.

    Start Logits: (B, N_start)
    End Logits: (B, N_start, N_end)

    Parameters
    ----------
    original_feature:
        The original SquadFeature before chunked
    chunked_features
        List of ChunkFeatures
    results
        List of model predictions for span start and span end.
    n_best_size
        Best N results written to file
    max_answer_length
        Maximum length of the answer tokens.
    start_top_n
        Number of start-position candidates
    end_top_n
        Number of end-position candidates
    Returns
    -------
    not_answerable_score
        Model's estimate that the question is not answerable.
    prediction
        The final prediction.
    nbest_json
        n-best predictions with their probabilities.
    """
    not_answerable_score = 1000000  # Score for not-answerable. We set it to be a large and positive
    # If one chunk votes for answerable, we will treat the context as answerable,
    # Thus, the overall not_answerable_score = min(chunk_not_answerable_score)
    for chunk_id, (result, chunk_feature) in enumerate(zip(results, chunked_features)):
        # We use the log-likelihood as the not answerable score.
        # Thus, a high score indicates that the answer is not answerable
        cur_not_answerable_score = float(result.answerable_logits[1])
        not_answerable_score = min(not_answerable_score, cur_not_answerable_score)
    return not_answerable_score


def evaluate(args, last=True):
    ctx_l = parse_ctx(args.gpus)
    cfg, tokenizer, qa_net, use_segmentation = get_network(
        args.model_name, ctx_l, args.classifier_dropout)
    # Prepare dev set
    dev_cache_path = os.path.join(CACHE_PATH,
                                  'dev_{}_squad_{}.ndjson'.format(args.model_name,
                                                                  args.version))
    if os.path.exists(dev_cache_path) and not args.overwrite_cache:
        dev_features = []
        with open(dev_cache_path, 'r') as f:
            for line in f:
                dev_features.append(SquadFeature.from_json(line))
        logging.info('Found cached dev features, load from {}'.format(dev_cache_path))
    else:
        dev_examples = get_squad_examples(args.data_dir, segment='dev', version=args.version)
        start = time.time()
        num_process = min(cpu_count(), 8)
        logging.info('Tokenize Dev Data:')
        with Pool(num_process) as pool:
            dev_features = pool.map(functools.partial(convert_squad_example_to_feature,
                                                      tokenizer=tokenizer,
                                                      is_training=False), dev_examples)
        logging.info('Done! Time spent:{:.2f} seconds'.format(time.time() - start))
        with open(dev_cache_path, 'w') as f:
            for feature in dev_features:
                f.write(feature.to_json() + '\n')
    dev_data_path = os.path.join(args.data_dir, 'dev-v{}.json'.format(args.version))
    dataset_processor = SquadDatasetProcessor(tokenizer=tokenizer,
                                              doc_stride=args.doc_stride,
                                              max_seq_length=args.max_seq_length,
                                              max_query_length=args.max_query_length)
    dev_all_chunk_features = []
    dev_chunk_feature_ptr = [0]
    for feature in dev_features:
        chunk_features = dataset_processor.process_sample(feature)
        dev_all_chunk_features.extend(chunk_features)
        dev_chunk_feature_ptr.append(dev_chunk_feature_ptr[-1] + len(chunk_features))

    def eval_validation(ckpt_name, best_eval):
        """
        Model inference during validation or final evaluation.
        """
        ctx_l = parse_ctx(args.gpus)
        # We process all the chunk features and also
        dev_dataloader = mx.gluon.data.DataLoader(
            dev_all_chunk_features,
            batchify_fn=dataset_processor.BatchifyFunction,
            batch_size=args.eval_batch_size,
            num_workers=0,
            shuffle=False)

        log_interval = args.log_interval
        all_results = []
        epoch_tic = time.time()
        tic = time.time()
        epoch_size = len(dev_features)
        total_num = 0
        log_num = 0
        for batch_idx, dev_batch in enumerate(grouper(dev_dataloader, len(ctx_l))):
            # Predict for each chunk
            for sample, ctx in zip(dev_batch, ctx_l):
                if sample is None:
                    continue
                # Copy the data to device
                tokens = sample.data.as_in_ctx(ctx)
                total_num += len(tokens)
                log_num += len(tokens)
                segment_ids = sample.segment_ids.as_in_ctx(ctx) if use_segmentation else None
                valid_length = sample.valid_length.as_in_ctx(ctx)
                p_mask = sample.masks.as_in_ctx(ctx)
                a_mask = sample.answer_masks.as_in_ctx(ctx)
                p_mask = 1 - p_mask  # In the network, we use 1 --> no_mask, 0 --> mask
                answerable_logits = qa_net(tokens, segment_ids, valid_length, p_mask, a_mask)
                for i, qas_id in enumerate(sample.qas_id):
                    result = RawResultExtended(qas_id=qas_id,
                                               answerable_logits=answerable_logits[i].asnumpy())

                    all_results.append(result)

            # logging
            if (batch_idx + 1) % log_interval == 0:
                # Output the loss of per step
                toc = time.time()
                logging.info(
                    '[batch {}], Time cost={:.2f},'
                    ' Throughput={:.2f} samples/s, ETA={:.2f}h'.format(
                        batch_idx + 1, toc - tic, log_num / (toc - tic),
                        (epoch_size - total_num) / (total_num / (toc - epoch_tic)) / 3600))
                tic = time.time()
                log_num = 0

        epoch_toc = time.time()
        logging.info('Time cost=%2f s, Thoughput=%.2f samples/s', epoch_toc - epoch_tic,
                     total_num / (epoch_toc - epoch_tic))

        no_answer_score_json = collections.OrderedDict()
        for index, (left_index, right_index) in enumerate(zip(dev_chunk_feature_ptr[:-1],
                                                              dev_chunk_feature_ptr[1:])):
            chunked_features = dev_all_chunk_features[left_index:right_index]
            results = all_results[left_index:right_index]
            original_feature = dev_features[index]
            qas_ids = set([result.qas_id for result in results] +
                          [feature.qas_id for feature in chunked_features])
            assert len(qas_ids) == 1, 'Mismatch Occured between features and results'
            example_qas_id = list(qas_ids)[0]
            assert example_qas_id == original_feature.qas_id, \
                'Mismatch Occured between original feature and chunked features'
            not_answerable_score = predict_extended(
                original_feature=original_feature,
                chunked_features=chunked_features,
                results=results,
                n_best_size=args.n_best_size,
                max_answer_length=args.max_answer_length,
                start_top_n=args.start_top_n,
                end_top_n=args.end_top_n)
            no_answer_score_json[example_qas_id] = not_answerable_score

        cur_eval = answerable_eval(dev_data_path, no_answer_score_json)
        logging.info('The evaluated results are {}'.format(json.dumps(cur_eval)))

        cur_metrics = cur_eval['accuracy']
        if best_eval:
            best_metrics = best_eval['accuracy']
        else:
            best_metrics = 0.

        if cur_metrics > best_metrics:
            logging.info('The evaluated files are saved in {}'.format(args.output_dir))
            na_prob_file = os.path.join(args.output_dir, 'na_prob.json')

            with open(na_prob_file, 'w') as of:
                of.write(json.dumps(no_answer_score_json, indent=4) + '\n')

            best_eval = cur_eval
            best_eval.update({'best_ckpt': ckpt_name})
        return best_eval

    if args.param_checkpoint and args.param_checkpoint.endswith('.params'):
        ckpt_candidates = [args.param_checkpoint]
    else:
        ckpt_candidates = [f for f in os.listdir(args.output_dir) if f.endswith('.params')]
        ckpt_candidates.sort(key=lambda ele: (len(ele), ele))
    if last:
        ckpt_candidates = ckpt_candidates[-1:]

    best_eval = {}
    for ckpt_name in ckpt_candidates:
        logging.info('Starting evaluate the checkpoint {}'.format(ckpt_name))
        ckpt_path = os.path.join(args.output_dir, ckpt_name)
        qa_net.load_parameters(ckpt_path, ctx=ctx_l, cast_dtype=True)
        best_eval = eval_validation(ckpt_name, best_eval)

    logging.info('The best evaluated results are {}'.format(json.dumps(best_eval)))
    output_eval_results_file = os.path.join(args.output_dir, 'best_results.json')
    with open(output_eval_results_file, 'w') as of:
        of.write(json.dumps(best_eval, indent=4) + '\n')
    return best_eval


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    os.environ['MXNET_USE_FUSION'] = '0'  # Manually disable pointwise fusion
    args = parse_args()
    logging_config(args.output_dir, name='finetune_squad{}'.format(args.version))
    set_seed(args.seed)
    if args.do_train:
        train(args)
    if args.do_eval:
        evaluate(args, last=not args.all_evaluate)
