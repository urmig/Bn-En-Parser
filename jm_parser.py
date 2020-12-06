#)!/usr/bin/python -*- coding: utf-8 -*-
import io
import re
import os
import sys
import copy
import random
import pickle
import utils
import argparse

import dynet as dy
import numpy as np
from collections import Counter, namedtuple, defaultdict

from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from lib.arc_eager import ArcEager
from lib.pseudoProjectivity import *

import ParserClass

def Train(sentence, epoch):
    parser.eval = False
    configuration = utils.Configuration(sentence)
    pr_bi_exps, pos_errs = parser.feature_extraction(sentence[1:-1])
    while not parser.isFinalState(configuration):
        rfeatures = parser.basefeaturesEager(configuration.nodes, configuration.stack, configuration.b0)
        #MLP layer
        xi = dy.concatenate([pr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
        xh = parser.pr_W1 * xi
        xh = parser.meta.activation(xh) + parser.pr_b1
        xo = parser.pr_W2*xh + parser.pr_b2
        output_probs = dy.softmax(xo).npvalue()

        ranked_actions = sorted(zip(output_probs, range(len(output_probs))), reverse=True)
        pscore, paction = ranked_actions[0]

        validTransitions, allmoves = parser.get_valid_transitions(configuration)
        while parser.action_cost(configuration, parser.meta.i2td[paction], parser.meta.transitions, validTransitions) > 500:
           ranked_actions = ranked_actions[1:]
           pscore, paction = ranked_actions[0]

        #find gold action : cost = 0
        gaction = None
        for i,(score, ltrans) in enumerate(ranked_actions):
           cost = parser.action_cost(configuration, parser.meta.i2td[ltrans], parser.meta.transitions, validTransitions)
           if cost == 0:
              gaction = ltrans
              break

        gtransitionstr, goldLabel = parser.meta.i2td[gaction]
        goldTransitionFunc = allmoves[parser.meta.transitions[gtransitionstr]]
        goldTransitionFunc(configuration, goldLabel)
        parser.loss.append(dy.pickneglogsoftmax(xo, parser.meta.td2i[(gtransitionstr, goldLabel)]))
    parser.loss.extend(pos_errs)


def Test(test_file, lang=None):
    with io.open(test_file, encoding='utf-8') as fp:
        inputGenTest = re.finditer("(.*?)\n\n", fp.read(), re.S)

    parser.eval = True
    scores = defaultdict(int)
    good, bad = 0.0, 0.0
    for idx, sentence in enumerate(inputGenTest):
        graph = list(utils.dependencyGraph(sentence.group(1), lang))
        pr_bi_exps, pos_errs = parser.feature_extraction(graph[1:-1])
        for xo, node in zip(pos_errs, graph[1:-1]):
            if node.tag == parser.meta.i2p[np.argmax(xo)]:
                good += 1
            else:
                bad += 1

        configuration = utils.Configuration(graph)
        while not parser.isFinalState(configuration):
            rfeatures = parser.basefeaturesEager(configuration.nodes, configuration.stack, configuration.b0)
            xi = dy.concatenate([pr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
            xh = parser.pr_W1 * xi
            xh = parser.meta.activation(xh) + parser.pr_b1
            xo = parser.pr_W2*xh + parser.pr_b2
            output_probs = dy.softmax(xo).npvalue()
            validTransitions, _ = parser.get_valid_transitions(configuration)
            sortedPredictions = sorted(zip(output_probs, range(len(output_probs))), reverse=True)
            for score, action in sortedPredictions:
    	        transition, predictedLabel = parser.meta.i2td[action]
    	        if parser.meta.transitions[transition] in validTransitions:
    	            predictedTransitionFunc = validTransitions[parser.meta.transitions[transition]]
    	            predictedTransitionFunc(configuration, predictedLabel)
    	            break
        dgraph = deprojectivize(graph[1:-1])
        scores = utils.tree_eval(dgraph, scores)
    sys.stderr.write('\n')

    UAS = round(100. * scores['rightAttach']/(scores['rightAttach']+scores['wrongAttach']),2)
    LAS = round(100. * scores['rightLabeledAttach']/(scores['rightLabeledAttach']+scores['wrongLabeledAttach']),2)
    POS = good/(good+bad) * 100.
    return POS, UAS, LAS

def backpropagate(loss, cum_loss):
    batch_loss = dy.esum(loss)
    cum_loss += batch_loss.scalar_value()
    batch_loss.backward()
    trainer.update()
    return [], cum_loss

def train_parser(dataset, batchsize):
    n_samples = len(dataset)
    sys.stdout.write("Training Samples: %s Classes: %s\n\n" % (n_samples, parser.meta.n_outs))
    psc, num_tagged, cum_loss = 0., 0, 0.
    for epoch in range(args.iter):
        random.shuffle(dataset)
        parser.loss = []
        dy.renew_cg()
        for sid, sentence in enumerate(dataset, 1):
            if sid % 1000 == 0 or sid == n_samples:
                trainer.status() #print status
                print(cum_loss / num_tagged)
                cum_loss, num_tagged = 0, 0
                sys.stdout.flush()
            csentence = copy.deepcopy(sentence)
            Train(csentence, epoch+1)
            num_tagged += 2 * len(sentence[1:-1]) - 1
            if len(parser.loss) > batchsize: #batching
                parser.loss, cum_loss = backpropagate(parser.loss, cum_loss)
                dy.renew_cg()
                sys.stderr.flush()
        if parser.loss:
            parser.loss, cum_loss = backpropagate(parser.loss, cum_loss)
            dy.renew_cg()
            sys.stderr.flush()

        POS, UAS, LAS = Test(args.bcdev)
        sys.stderr.write("BN-EN CM POS ACCURACY: {}% UAS: {}%, and LAS: {}%\n".format(POS, UAS, LAS))
        sys.stderr.flush()
        if LAS > psc:
            sys.stderr.write('SAVE POINT %d\n' %epoch)
            psc = LAS
            if args.save_model:
                parser.model.save('%s.dy' %args.save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Bn-En Code-Mixed Parser.", description="Bi-LSTM Parser")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-seed', dest='seed', type=int, default='127')
    parser.add_argument('--etrain', help='English CONLL Train file')
    parser.add_argument('--edev', help='English CONLL Dev/Test file')
    parser.add_argument('--btrain', help='Bengali CONLL Train file')
    parser.add_argument('--htrain', help='Hindi CONLL Train file')
    parser.add_argument('--hdev', help='Hindi CONLL Dev/Test file')
    parser.add_argument('--bdev', help='Bengali CONLL Dev/Test file')
    parser.add_argument('--hcdev', help='Hindi-English CS CONLL Dev/Test file')
    parser.add_argument('--bcdev', help='Bengali-English CS CONLL Dev/Test file')
    parser.add_argument('--hi-embds', dest='hembd', help='Pretrained Hindi word2vec Embeddings')
    parser.add_argument('--hi-limit', dest='hlimit', type=int, default=None,
                        help='load only top-n pretrained Hindi word vectors (default=all vectors)')
    parser.add_argument('--bn-embds', dest='bembd', help='Pretrained Bengali word2vec Embeddings')
    parser.add_argument('--bn-limit', dest='blimit', type=int, default=None,
                        help='load only top-n pretrained Bengali word vectors (default=all vectors)')
    parser.add_argument('--en-embds', dest='eembd', help='Pretrained English word2vec Embeddings')
    parser.add_argument('--en-limit', dest='elimit', type=int, default=None,
                        help='load only top-n pretrained English word vectors (default=all vectors)')
    parser.add_argument('--trainer', default='momsgd', help='Trainer [momsgd|adam|adadelta|adagrad]')
    parser.add_argument('--activation-fn', dest='act_fn', default='tanh', help='Activation function [tanh|rectify|logistic]')
    parser.add_argument('--batch-size', dest='batch_size', default=64, help='Batch size for training')
    parser.add_argument('--ud', type=int, default=1, help='1 if UD treebank else 0')
    parser.add_argument('--iter', type=int, default=100, help='No. of Epochs')
    parser.add_argument('--bvec', type=int, help='1 if binary embedding file else 0')
    group.add_argument('--save-model', dest='save_model', help='Specify path to save model')
    group.add_argument('--load-model', dest='load_model', help='Load Pretrained Model')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    meta = ParserClass.Meta()
    extra_meta=ParserClass.Meta()

    if args.load_model:
            sys.stderr.write('Loading Models ...\n')
            parser = Parser.Parser(model=args.load_model)
            sys.stderr.write('Done!\n')
            POS, UAS, LAS = Test(args.bcdev)
            sys.stderr.write("BN-EN TEST-SET POS: {}%, UAS: {}%, and LAS: {}%\n".format(POS, UAS, LAS))

    else: #Training mode
        train_sents = []
        train_sents+=utils.read(args.btrain, 'bn')
        train_sents+=utils.read(args.htrain, 'hi')
        train_sents+=utils.read(args.etrain, 'en')

        (meta.bhc2i,meta.ec2i,plabels,tdlabels) = utils.set_class_map(train_sents)
        extra_meta.bwvm = KeyedVectors.load_word2vec_format(args.bembd, binary=args.bvec, limit=args.hlimit)
        meta.n_words_ben = extra_meta.bwvm.vectors.shape[0]+meta.add_words
        meta.bw2i = {}
        for w in extra_meta.bwvm.vocab:
            meta.bw2i[w] = extra_meta.bwvm.vocab[w].index + meta.add_words
        extra_meta.ewvm = KeyedVectors.load_word2vec_format(args.eembd, binary=args.bvec, limit=args.elimit)
        meta.n_words_eng = extra_meta.ewvm.vectors.shape[0]+meta.add_words
        meta.ew2i={}
        for w in extra_meta.ewvm.vocab:
            meta.ew2i[w] = extra_meta.ewvm.vocab[w].index + meta.add_words
        if(extra_meta.bwvm.vectors.shape[1]==extra_meta.ewvm.vectors.shape[1]):
            meta.w_dim=extra_meta.bwvm.vectors.shape[1]
        else:
            sys.stderr.write('Error Use vectors of same dimensions')
            exit()
        extra_meta.hwvm = KeyedVectors.load_word2vec_format(args.hembd, binary=args.bvec, limit=args.hlimit)
        if(extra_meta.hwvm.vectors.shape[1]!=meta.w_dim):
            sys.stderr.write('Error Use vectors of same dimensions')
            exit()
        meta.n_words_hin = extra_meta.hwvm.vectors.shape[0]+meta.add_words
        meta.hw2i={}
        for w in extra_meta.hwvm.vocab:
            meta.hw2i[w] = extra_meta.hwvm.vocab[w].index + meta.add_words

        meta.i2p = dict(enumerate(plabels))
        meta.i2td = dict(enumerate(tdlabels))
        meta.p2i = {v: k for k,v in meta.i2p.items()}
        meta.td2i = {v: k for k,v in meta.i2td.items()}
        meta.n_outs = len(meta.i2td)
        meta.n_tags = len(meta.i2p)
        meta.n_chars_bh = len(meta.bhc2i)
        meta.n_chars_eng = len(meta.ec2i)

        trainers, act_fns = utils.trainer_defn()
        meta.trainer = trainers[args.trainer]
        meta.activation = act_fns[args.act_fn]

        if args.save_model:
                pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))

        parser = ParserClass.Parser(meta=meta, new_meta=extra_meta, args=args)
        trainer = meta.trainer(parser.model)
        train_parser(train_sents, int(args.batch_size))
