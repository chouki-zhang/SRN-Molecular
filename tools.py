#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:46:32 2019

@author: qizhang
"""


class transformer_parameters:
    def __init__(self,
                 alpha=0.0,
                 attn_debug=False,
                 batch_size=30,
                 beam_size=5,
                 beta=-0.0,
                 block_ngram_repeat=0,
                 coverage_penalty='none',
                 data_type='text',
                 dump_beam='',
                 dynamic_dict=False,
                 fast=True,
                 gpu=-1,
                 ignore_when_blocking=[],
                 image_channel_size=3,
                 length_penalty='none',
                 log_file='',
                 log_probs=False,
                 mask_from='',
                 max_input_len=4,
                 max_length=100,
                 max_sent_length=None,
                 min_length=0,
                 models=[''],
                 n_best=1,
                 output="",
                 replace_unk=True,
                 report_bleu=False,
                 report_rouge=False,
                 sample_rate=16000,
                 share_vocab=False,
                 src='',
                 src_dir='',
                 stepwise_penalty=False,
                 tgt=None,
                 verbose=False,
                 window='hamming',
                 window_size=0.02,
                 window_stride=0.01):
        self.alpha = alpha
        self.attn_debug = attn_debug
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.beta = beta
        self.block_ngram_repeat = block_ngram_repeat
        self.coverage_penalty = coverage_penalty
        self.data_type = data_type
        self.dump_beam = dump_beam
        self.dynamic_dict = dynamic_dict
        self.fast = fast
        self.gpu = gpu
        self.ignore_when_blocking = ignore_when_blocking
        self.image_channel_size = image_channel_size
        self.length_penalty = length_penalty
        self.log_file = log_file
        self.log_probs = log_probs
        self.mask_from = mask_from
        self.max_input_len = max_input_len
        self.max_length = max_length
        self.max_sent_length = max_sent_length
        self.min_length = min_length
        self.models = models
        self.n_best = n_best
        self.output = output
        self.replace_unk = replace_unk
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.sample_rate = sample_rate
        self.share_vocab = share_vocab
        self.src = src
        self.src_dir = src_dir
        self.stepwise_penalty = stepwise_penalty
        self.tgt = tgt
        self.verbose = verbose
        self.window = window
        self.window_size = window_size
        self.window_stride = window_stride


class Reactor():
    def __init__(self, batch_size=128):
        self.batch_size = batch_size

    def BuildReactor(self, model_list=[], max_length=100, n_best=1, gpu=-1):
        opt = transformer_parameters(batch_size=self.batch_size, max_length=max_length,
                                     n_best=n_best, gpu=gpu, models=model_list, src=None)
        from onmt.translate.translator import build_translator
        self.model = build_translator(opt, report_score=False)

    def react(self, reactants):
        [all_scores, all_predictions] = self.model.translate(
            src_data_iter=reactants, tgt_path=None, src_dir='', batch_size=self.batch_size, attn_debug=False)
        return all_scores, all_predictions
