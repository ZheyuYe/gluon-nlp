"""
Question Answering with Pretrained Language Model using ModelForQABasic
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
from eval_utils import squad_eval
from squad_utils import SquadFeature, get_squad_examples, convert_squad_example_to_feature
from gluonnlp.models import get_backbone
from gluonnlp.utils.misc import grouper, repeat, set_seed, parse_ctx, logging_config, count_parameters
from gluonnlp.initializer import TruncNorm
from gluonnlp.utils.parameter import clip_grad_global_norm
from run_squad import SquadDatasetProcessor, parse_args, get_network

mx.npx.set_np()

CACHE_PATH = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', 'cached'))
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH, exist_ok=True)


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
                    args.backbone_path,
                    qa_model_type='basic')
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
                gt_start = sample.gt_start.as_in_ctx(ctx)
                gt_end = sample.gt_end.as_in_ctx(ctx)
                is_impossible = sample.is_impossible.as_in_ctx(ctx).astype(np.int32)
                batch_idx = mx.np.arange(tokens.shape[0], dtype=np.int32, ctx=ctx)
                p_mask = 1 - p_mask  # In the network, we use 1 --> no_mask, 0 --> mask
                with mx.autograd.record():
                    start_logits, end_logits \
                        = qa_net(tokens, segment_ids, valid_length, p_mask)
                    sel_start_logits = start_logits[batch_idx, gt_start]
                    sel_end_logits = end_logits[batch_idx, gt_end]
                    span_loss = - 0.5 * (sel_start_logits + sel_end_logits).sum()
                    loss = span_loss / loss_denom
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
                'Step: {}/{}, Loss span={:.4f},'
                ' LR={:.8f}, grad_norm={:.4f}. Time cost={:.2f}, Throughput={:.2f} samples/s'
                ' ETA={:.2f}h'.format((step_num + 1), num_train_steps, log_total_loss,
                                      trainer.learning_rate, total_norm,
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
     'start_top_logits',
     'start_top_index',
     'end_top_logits',
     'end_top_index'])


def predict_extended(original_feature,
                     chunked_features,
                     results,
                     n_best_size=20,
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
    score_null = 1000000  # Score for not-answerable. We set it to be a large and positive
    # If one chunk votes for answerable, we will treat the context as answerable,
    # Thus, the overall not_answerable_score = min(chunk_not_answerable_score)
    all_start_idx = []
    all_end_idx = []
    all_pred_score = []
    context_length = len(original_feature.context_token_ids)
    token_max_context_score = np.full((len(chunked_features), context_length),
                                      -np.inf,
                                      dtype=np.float32)
    for i, chunked_feature in enumerate(chunked_features):
        chunk_start = chunked_feature.chunk_start
        chunk_length = chunked_feature.chunk_length
        for j in range(chunk_start, chunk_start + chunk_length):
            # This is a heuristic score
            # TODO investigate the impact
            token_max_context_score[i, j] = min(j - chunk_start,
                                                chunk_start + chunk_length - 1 - j) \
                + 0.01 * chunk_length
    token_max_chunk_id = token_max_context_score.argmax(axis=0)

    for chunk_id, (result, chunk_feature) in enumerate(zip(results, chunked_features)):
        # We use the log-likelihood as the not answerable score.
        # Thus, a high score indicates that the answer is not answerable

        # Calculate the start_logits + end_logits as the overall score
        context_offset = chunk_feature.context_offset
        chunk_start = chunk_feature.chunk_start
        chunk_length = chunk_feature.chunk_length
        for i in range(start_top_n):
            for j in range(end_top_n):
                pred_score = float(result.start_top_logits[i] + result.end_top_logits[j])
                score_null = min(pred_score, score_null)

                start_index = result.start_top_index[i]
                end_index = result.end_top_index[j]
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the answer span is in the query tokens or out of
                # the chunk. We throw out all invalid predictions.
                if not (context_offset <= start_index < context_offset + chunk_length) or \
                   not (context_offset <= end_index < context_offset + chunk_length) or \
                   end_index < start_index:
                    continue
                pred_answer_length = end_index - start_index + 1
                if pred_answer_length > max_answer_length:
                    continue
                start_idx = int(start_index - context_offset + chunk_start)
                end_idx = int(end_index - context_offset + chunk_start)
                if token_max_chunk_id[start_idx] != chunk_id:
                    continue
                all_start_idx.append(start_idx)
                all_end_idx.append(end_idx)
                all_pred_score.append(pred_score)

    all_start_idx.append(None)
    all_end_idx.append(None)
    all_pred_score.append(score_null)

    sorted_start_end_score = sorted(zip(all_start_idx, all_end_idx, all_pred_score),
                                    key=lambda args: args[-1], reverse=True)
    nbest = []
    context_text = original_feature.context_text
    context_token_offsets = original_feature.context_token_offsets
    seen_predictions = set()
    for start_idx, end_idx, pred_score in sorted_start_end_score:
        if len(seen_predictions) >= n_best_size:
            break
        if start_idx is not None and end_idx is not None:
            pred_answer = context_text[context_token_offsets[start_idx][0]:
                                       context_token_offsets[end_idx][1]]
        else:
            pred_answer = ""
        seen_predictions.add(pred_answer)
        nbest.append((pred_answer, pred_score))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if len(nbest) == 0:
        nbest.append(('', float('-inf')))
    all_scores = np.array([ele[1] for ele in nbest], dtype=np.float32)
    probs = np.exp(all_scores) / np.sum(np.exp(all_scores))
    nbest_json = []
    for i, (entry, prob) in enumerate(zip(nbest, probs)):
        output = collections.OrderedDict()
        output['text'] = entry[0]
        output['probability'] = float(prob)
        nbest_json.append(output)

    assert len(nbest_json) >= 1
    not_answerable_score = score_null - float(nbest[0][1])
    return not_answerable_score, nbest[0][0], nbest_json


def evaluate(args, last=True):
    ctx_l = parse_ctx(args.gpus)
    cfg, tokenizer, qa_net, use_segmentation = get_network(
        args.model_name, ctx_l, args.classifier_dropout, qa_model_type='basic')
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
                p_mask = 1 - p_mask  # In the network, we use 1 --> no_mask, 0 --> mask
                start_top_logits, start_top_index, end_top_logits, end_top_index \
                    = qa_net.inference(tokens, segment_ids, valid_length, p_mask,
                                       args.start_top_n, args.end_top_n)
                for i, qas_id in enumerate(sample.qas_id):
                    result = RawResultExtended(qas_id=qas_id,
                                               start_top_logits=start_top_logits[i].asnumpy(),
                                               start_top_index=start_top_index[i].asnumpy(),
                                               end_top_logits=end_top_logits[i].asnumpy(),
                                               end_top_index=end_top_index[i].asnumpy())

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

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
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
            not_answerable_score, best_pred, nbest_json = predict_extended(
                original_feature=original_feature,
                chunked_features=chunked_features,
                results=results,
                n_best_size=args.n_best_size,
                max_answer_length=args.max_answer_length,
                start_top_n=args.start_top_n,
                end_top_n=args.end_top_n)
            no_answer_score_json[example_qas_id] = not_answerable_score
            all_predictions[example_qas_id] = best_pred
            all_nbest_json[example_qas_id] = nbest_json

        if args.version == '2.0':
            exact = 'best_exact'
            f1 = 'best_f1'
            na_prob = no_answer_score_json
        else:
            exact = 'exact'
            f1 = 'f1'
            na_prob = None

        cur_eval, revised_predictions = squad_eval(
            dev_data_path, all_predictions, na_prob, revise=na_prob is not None)
        logging.info('The evaluated results are {}'.format(json.dumps(cur_eval)))

        cur_metrics = 0.5 * (cur_eval[exact] + cur_eval[f1])
        if best_eval:
            best_metrics = 0.5 * (best_eval[exact] + best_eval[f1])
        else:
            best_metrics = 0.

        if cur_metrics > best_metrics:
            logging.info('The evaluated files are saved in {}'.format(args.output_dir))
            output_prediction_file = os.path.join(args.output_dir, 'predictions.json')
            output_nbest_file = os.path.join(args.output_dir, 'nbest_predictions.json')
            na_prob_file = os.path.join(args.output_dir, 'na_prob.json')
            revised_prediction_file = os.path.join(args.output_dir, 'revised_predictions.json')

            with open(output_prediction_file, 'w') as of:
                of.write(json.dumps(all_predictions, indent=4) + '\n')
            with open(output_nbest_file, 'w') as of:
                of.write(json.dumps(all_nbest_json, indent=4) + '\n')
            with open(na_prob_file, 'w') as of:
                of.write(json.dumps(no_answer_score_json, indent=4) + '\n')
            with open(revised_prediction_file, 'w') as of:
                of.write(json.dumps(revised_predictions, indent=4) + '\n')

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
