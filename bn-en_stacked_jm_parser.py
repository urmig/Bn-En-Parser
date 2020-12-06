#)!/usr/bin/python -*- coding: utf-8 -*-

import io
import os
import re
import sys
import copy
import random
import pickle

import argparse
import numpy as np
from collections import Counter, namedtuple, defaultdict

import dynet as dy
from gensim.models.word2vec import Word2Vec

from lib.arc_eager import ArcEager
from lib.pseudoProjectivity import *

import StackedParserClass
import utils

def Train(sentence, epoch, dynamic=True):
    loss = []
    totalError = 0
    parser.eval = False
    configuration = utils.Configuration(sentence)
    pr_bi_exps, xpr_bi_exps, pos_errs = parser.feature_extraction(sentence[1:-1])
    while not parser.isFinalState(configuration):
        rfeatures = parser.basefeaturesEager(configuration.nodes, configuration.stack, configuration.b0)
        xi = dy.concatenate([pr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
        yi = dy.concatenate([xpr_bi_exps[id-1] if id > 0 else parser.xpad for id, rform in rfeatures])
        xh = parser.pr_W1 * xi
        xi = dy.concatenate([yi, xh])
        xh = parser.xpr_W1 * xi
        xh = parser.meta.activation(xh) + parser.xpr_b1
        xo = parser.xpr_W2*xh + parser.xpr_b2
        output_probs = dy.softmax(xo).npvalue()
        ranked_actions = sorted(zip(output_probs, range(len(output_probs))), reverse=True)
        pscore, paction = ranked_actions[0]

        validTransitions, allmoves = parser.get_valid_transitions(configuration)
        while parser.action_cost(configuration, parser.meta.i2td[paction], parser.meta.transitions, validTransitions) > 500:
           ranked_actions = ranked_actions[1:]
           pscore, paction = ranked_actions[0]

        gaction = None
        for i,(score, ltrans) in enumerate(ranked_actions):
           cost = parser.action_cost(configuration, parser.meta.i2td[ltrans], parser.meta.transitions, validTransitions)
           if cost == 0:
              gaction = ltrans
              break

        gtransitionstr, goldLabel = parser.meta.i2td[gaction]
        ptransitionstr, predictedLabel = parser.meta.i2td[paction]

        goldTransitionFunc = allmoves[parser.meta.transitions[gtransitionstr]]
        goldTransitionFunc(configuration, goldLabel)
        parser.loss.append(dy.pickneglogsoftmax(xo, parser.meta.td2i[(gtransitionstr, goldLabel)]))
    parser.loss.extend(pos_errs)

def Test(test_file, ofp=None, lang=None):
    with io.open(test_file, encoding='utf-8') as fp:
        inputGenTest = re.finditer("(.*?)\n\n", fp.read(), re.S)
    parser.eval = True
    scores = defaultdict(int)
    good, bad = 0.0, 0.0
    for idx, sentence in enumerate(inputGenTest):
        graph = list(utils.dependencyGraph(sentence.group(1), lang))
        pr_bi_exps, xpr_bi_exps, pos_errs = parser.feature_extraction(graph[1:-1])
        pred_pos = []
        for xo, node in zip(pos_errs, graph[1:-1]):
            p_tag = parser.meta.i2p[np.argmax(xo)]
            pred_pos.append(p_tag)
            if node.tag == p_tag:
                good += 1
            else:
                bad += 1

        configuration = utils.Configuration(graph)
        while not parser.isFinalState(configuration):
            rfeatures = parser.basefeaturesEager(configuration.nodes, configuration.stack, configuration.b0)
            xi = dy.concatenate([pr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
            yi = dy.concatenate([xpr_bi_exps[id-1] if id > 0 else parser.xpad for id, rform in rfeatures])
            xh = parser.pr_W1 * xi
            xi = dy.concatenate([yi, xh])
            xh = parser.xpr_W1 * xi
            xh = parser.meta.activation(xh) + parser.xpr_b1
            xo = parser.xpr_W2*xh + parser.xpr_b2
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
        #sys.stderr.write("Testing Instances:: %s\r"%idx)
    sys.stderr.write('\n')

    UAS = round(100. * scores['rightAttach']/(scores['rightAttach']+scores['wrongAttach']),2)
    LAS = round(100. * scores['rightLabeledAttach']/(scores['rightLabeledAttach']+scores['wrongLabeledAttach']),2)
    POS = good/(good+bad) * 100.
    return POS, UAS, LAS

def build_dependency_graph(parser, graph):
    pred_pos = []
    parser.eval = True
    pr_bi_exps, xpr_bi_exps, pos_errs = parser.feature_extraction(graph[1:-1])
    for xo, node in zip(pos_errs, graph[1:-1]):
        p_tag = parser.meta.i2p[np.argmax(xo)]
        pred_pos.append(p_tag)

    configuration = utils.Configuration(graph)
    while not parser.isFinalState(configuration):
            rfeatures = parser.basefeaturesEager(configuration.nodes, configuration.stack, configuration.b0)
            pr_bi_exps
            xi = dy.concatenate([pr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
            yi = dy.concatenate([xpr_bi_exps[id-1] if id > 0 else parser.xpad for id, rform in rfeatures])
            xh = parser.pr_W1 * xi
            xi = dy.concatenate([yi, xh])
            xh = parser.xpr_W1 * xi
            xh = parser.meta.activation(xh) + parser.xpr_b1
            xo = parser.xpr_W2*xh + parser.xpr_b2
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

    return dgraph, pred_pos

def backpropagate(loss, cum_loss):
    batch_loss = dy.esum(loss)
    cum_loss += batch_loss.scalar_value()
    batch_loss.backward()
    trainer.update()
    return [], cum_loss

def train_parser(dataset, batchsize):
    n_samples = len(dataset)
    sys.stdout.write("Started training ...\n")
    sys.stdout.write("Training Examples: %s Classes: %s Epochs: %d\n\n" % (n_samples, parser.meta.xn_outs, args.iter))
    psc, num_tagged, cum_loss = 0., 0, 0.
    for epoch in range(args.iter):
        random.shuffle(dataset)
        parser.loss = []
        dy.renew_cg()
        for sid, sentence in enumerate(dataset, 1):
            if sid % 1000 == 0 or sid == n_samples:   # print status
                trainer.status()
                print(cum_loss / num_tagged)
                cum_loss, num_tagged = 0, 0
                sys.stdout.flush()
            csentence = copy.deepcopy(sentence)
            Train(csentence, epoch+1)
            num_tagged += 2 * len(sentence[1:-1]) - 1
            if len(parser.loss) > batchsize:
                parser.loss, cum_loss = backpropagate(parser.loss, cum_loss)
                dy.renew_cg()
                sys.stderr.flush()
        if parser.loss:
            parser.loss, cum_loss = backpropagate(parser.loss, cum_loss)
            dy.renew_cg()

        POS, UAS, LAS = Test(args.bcdev)
        sys.stderr.write("\nBengali CM POS ACCURACY: {}% UAS: {}%, and LAS: {}%\n".format(POS, UAS, LAS))
        sys.stderr.flush()
        if LAS > psc:
            sys.stderr.write('SAVE POINT %d\n' %epoch)
            psc = LAS
            if args.save_model:
                parser.model.save('%s.dy' %args.save_model)

def parse_sent(parser, sentence):
    parser.eval = True
    leaf = namedtuple('leaf', ['id','form','lemma','tag','lang','features','parent','pparent', 'drel','pdrel','left','right', 'visit'])
    PAD = leaf._make([-1,'PAD','PAD','PAD','PAD','PAD',-1,-1,'PAD','PAD',[None],[None], False])
    graph = [leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', 'ROOT_T', -1, -1, 'ROOT', 'ROOT', PAD, [None], False])]
    for i,w in enumerate(sentence.split('\n'), 1):
        t=w.split()
        graph += [leaf._make([int(t[0]),t[1],t[1],'_',t[2], '_',-1,-1,'_','_',[None],[None], False])]
    graph += [leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', 'ROOT_T', -1, -1, 'ROOT', 'ROOT', [None], [None], False])]
    graph, ppos = build_dependency_graph(parser, graph)
    return '\n'.join(['\t'.join([str(node.id), node.form, node.lemma, pos, u'_', u'_', str(node.pparent),
                  node.pdrel.strip('%'), u'_', u'_']) for node,pos in zip(graph, ppos)])

def test_raw_sents(testfile, outfile):
    ofp = io.open(outfile, 'w+', encoding='utf-8')
    with io.open(testfile, 'r', encoding='utf-8') as ifp:
        inputGenTest = re.finditer("(.*?)\n\n", ifp.read(), re.S)
        for sen in inputGenTest:
            parsed_sent = parse_sent(parser, sen.group(1))
            ofp.write(parsed_sent+'\n\n')
    ofp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Bn-En Code-Mixed Stacked Neural Network Parser.", description="Bi-LSTM Parser")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-gpu')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-devices')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed', dest='seed', type=int, default='127')
    parser.add_argument('--bctrain', help='Hindi-English CS CONLL Train file')
    parser.add_argument('--hctrain', help='Hindi-English CS CONLL Train file')
    parser.add_argument('--bcdev', help='Hindi-English CS CONLL Dev/Test file')
    parser.add_argument('--trainer', default='momsgd', help='Trainer [momsgd|adam|adadelta|adagrad]')
    parser.add_argument('--activation-fn', dest='act_fn', default='tanh', help='Activation function [tanh|rectify|logistic]')
    parser.add_argument('--ud', type=int, default=1, help='1 if UD treebank else 0')
    parser.add_argument('--iter', type=int, default=100, help='No. of Epochs')
    parser.add_argument('--bvec', type=int, help='1 if binary embedding file else 0')
    group.add_argument('--save-model', dest='save_model', help='Specify path to save model')
    group.add_argument('--load-model', dest='load_model', help='Load Pretrained Model')
    parser.add_argument('--base-model', dest='base_model', help='build a stacking model on this pretrained model')
    parser.add_argument('--output-file', dest='outfile', default='/tmp/out.conll', help='Output File')
    parser.add_argument('--test', help='Test file. See sample file test.txt')
    parser.add_argument('--batch-size', dest='batch_size', default=64, help='Batch size for training')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.bcdev:
        bcdev = utils.read(args.bcdev)

    if not args.load_model:
        xmeta = StackedParserClass.Meta()
        meta = pickle.load(open('%s.meta' %args.base_model, 'rb'))


        train_sents = utils.read(args.hctrain)
        train_sents += utils.read(args.bctrain)
        plabels, tdlabels = utils.x_set_class_map(train_sents)

        meta.i2p = dict(enumerate(plabels))
        meta.i2td = dict(enumerate(tdlabels))
        meta.p2i = {v: k for k,v in meta.i2p.items()}
        meta.td2i = {v: k for k,v in meta.i2td.items()}
        meta.xn_outs = len(meta.i2td)
        meta.xn_tags = len(meta.p2i)
        meta.xc_dim = xmeta.xc_dim
        meta.xp_hidden = xmeta.xp_hidden
        meta.xn_hidden = xmeta.xn_hidden
        meta.xlstm_wc_dim = xmeta.xlstm_wc_dim
        meta.xlstm_char_dim = xmeta.xlstm_char_dim

        trainers, act_fns = utils.trainer_defn()
        meta.trainer = trainers[args.trainer]
        meta.activation = act_fns[args.act_fn]

    if args.save_model:
        pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))

    if args.load_model:
        sys.stderr.write('Loading Models ...\n')
        parser = StackedParserClass.Parser(model=args.load_model, test=True)
        sys.stderr.write('Done!\n')
        if args.bcdev:
            POS, UAS, LAS = Test(args.bcdev)
            sys.stderr.write("Bengali CM TEST-SET POS: {}%, UAS: {}%, and LAS: {}%\n".format(POS, UAS, LAS))
        if args.test:
            test_raw_sents(args.test, args.outfile)

    elif args.base_model:
        parser = StackedParserClass.Parser(model=args.base_model, new_meta=meta)
        trainer = meta.trainer(parser.model)
        train_parser(train_sents, int(args.batch_size))
