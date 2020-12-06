# Serve model as a flask application

import io
import re
import sys
import dynet as dy
import numpy as np
from flask import Flask, request, render_template
from collections import Counter, namedtuple
from lib.pseudoProjectivity import *
import StackedParserClass
import utils


parser = None
app = Flask(__name__)

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
            validTransitions, _ = parser.get_valid_transitions(configuration) #{0: <bound method arceager.SHIFT>}
            sortedPredictions = sorted(zip(output_probs, range(len(output_probs))), reverse=True)
            for score, action in sortedPredictions:
    	        transition, predictedLabel = parser.meta.i2td[action]
    	        if parser.meta.transitions[transition] in validTransitions:
    	            predictedTransitionFunc = validTransitions[parser.meta.transitions[transition]]
    	            predictedTransitionFunc(configuration, predictedLabel)
    	            break
    dgraph = deprojectivize(graph[1:-1])

    return dgraph, pred_pos

def parse_sent(parser, sentence):
    parser.eval = True
    leaf = namedtuple('leaf', ['id','form','lemma','tag','lang','features','parent','pparent', 'drel','pdrel','left','right', 'visit'])
    PAD = leaf._make([-1,'PAD','PAD','PAD','PAD','PAD',-1,-1,'PAD','PAD',[None],[None], False])
    graph = [leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', 'ROOT_T', -1, -1, 'ROOT', 'ROOT', PAD, [None], False])]
    for i, w in enumerate(sentence.split(), 1):
        sys.stdout.flush()
        app.logger.info(w)
        t=w.split("_")
        if(len(t)!=2) or t[1] not in ['en', 'bn']:
            return "Error in Input Format. Please refer to the example given above."
        graph += [leaf._make([int(i),t[0],t[0],'_',t[1], '_',-1,-1,'_','_',[None],[None], False])]
        sys.stdout.flush()

    graph += [leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', 'ROOT_T', -1, -1, 'ROOT', 'ROOT', [None], [None], False])]
    graph, ppos = build_dependency_graph(parser, graph)
    return '\n'.join(['\t'.join([str(node.id), node.form, node.lemma, pos, u'_', u'_', str(node.pparent),
                  node.pdrel.strip('%'), u'_', u'_', node.lang]) for node,pos in zip(graph, ppos)])

def load_model(parsermodel):
    global parser
    # model variable refers to the global variable
    sys.stderr.write('Loading Models ...\n')
    parser = StackedParserClass.Parser(model=parsermodel, test=True)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        d = request.form
    output = parse_sent(parser, d['sentence'])
    return render_template('index.html', parsed_text='\n' + output.strip() +'\n')

if __name__ == '__main__':

    load_model("bn-en-stacked.model")  # load model
    app.run(host='0.0.0.0', port=80)
