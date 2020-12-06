import io, re, sys
from collections import Counter, namedtuple, defaultdict
from lib.pseudoProjectivity import *
import dynet as dy

class Configuration(object):
    def __init__(self, nodes=[]):
        self.stack = list()
        self.b0 = 1
        self.nodes = nodes

def set_class_map(data):
    #character map
    plabels = set()
    tdlabels = set()
    tdlabels.add(('SHIFT', None))
    tdlabels.add(('REDUCE', None))
    bhc2i, ec2i = [{'bos':0, 'eos':1, 'unk':2}]*2
    bhcid, ecid = len(bhc2i), len(ec2i)
    for graph in data:
        for pnode in graph[1:-1]:
            for c in pnode.form:
                if pnode.lang in ['hi', 'bn']: #IMP: Convert Hindi and Bengali to WX notation!
                    if c not in bhc2i:
                        bhc2i[c] = bhcid
                        bhcid += 1
                else:
                    if c not in ec2i:
                        ec2i[c] = ecid
                        ecid += 1
            plabels.add(pnode.tag)

            if pnode.parent == 0:
                tdlabels.add(('LEFTARC', pnode.drel))
            elif pnode.id < pnode.parent:
                tdlabels.add(('LEFTARC', pnode.drel))
            else:
                tdlabels.add(('RIGHTARC', pnode.drel))
    return bhc2i, ec2i, plabels, tdlabels

def x_set_class_map(data):
    plabels = set()
    tdlabels = set()
    tdlabels.add(('SHIFT', None))
    tdlabels.add(('REDUCE', None))
    for graph in data:
          for pnode in graph[1:-1]:
              plabels.add(pnode.tag)
              if pnode.parent == 0:
                  tdlabels.add(('LEFTARC', pnode.drel))
              elif pnode.id < pnode.parent:
                  tdlabels.add(('LEFTARC', pnode.drel))
              else:
                  tdlabels.add(('RIGHTARC', pnode.drel))
    return plabels, tdlabels

def read(fname, lang=None):
    with io.open(fname, encoding='utf-8') as fp:
        inputGenTrain = re.finditer("(.*?)\n\n", fp.read(), re.S)

    data = []
    for i,sentence in enumerate(inputGenTrain):
        graph = list(dependencyGraph(sentence.group(1), lang))
        try:
            pgraph = graph[:1]+projectivize(graph[1:-1])+graph[-1:]

        except:
            sys.stderr.write('Error Sent :: %d\n' %i)
            sys.stdout.flush()
            continue
        data.append(pgraph)
    return data

def dependencyGraph(sentence, lang=None):
    leaf = namedtuple('leaf', ['id','form','lemma','tag','ctag','lang','parent','pparent', 'drel','pdrel','left','right', 'visit'])
    PAD = leaf._make([-1,'__PAD__','__PAD__','__PAD__','__PAD__',defaultdict(lambda:'__PAD__'),-1,-1,'__PAD__','__PAD__',[None],[None], False])
    yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', defaultdict(str), -1, -1, '__ROOT__', '__ROOT__', PAD, [None], False])

    for node in sentence.split("\n"):
        if lang:
            id_,form,lemma,tag,ctag,_,parent,drel = node.split("\t")[:8]
            tlang = lang
        else:
            id_,lemma,form,tag,ctag,_,parent,drel,tlang = node.split("\t")[:9]
            tlang = tlang.split('|')[0]
            if tlang != 'hi' and tlang!='bn':
                tlang = 'en'
        if ':' in drel and drel != 'acl:relcl':
            drel = drel.split(':')[0]
        node = leaf._make([int(id_),form,lemma,tag,ctag,tlang,int(parent),-1,drel,drel,[None],[None], False])
        yield node
    yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', defaultdict(str), -1, -1, '__ROOT__', '__ROOT__', [None], [None], False])

def trainer_defn():
    trainers = {
            'momsgd'   : dy.MomentumSGDTrainer,
            'adam'     : dy.AdamTrainer,
            'simsgd'   : dy.SimpleSGDTrainer,
            'adagrad'  : dy.AdagradTrainer,
            'adadelta' : dy.AdadeltaTrainer
            }
    act_fn = {
            'sigmoid' : dy.logistic,
            'tanh'    : dy.tanh,
            'relu'    : dy.rectify,
            }
    return trainers, act_fn

def tree_eval(sentence, scores):
    for node in sentence:
        if node.parent == node.pparent:
            scores['rightAttach'] += 1
            if node.drel.strip('%') == node.pdrel.strip('%'):
                scores['rightLabeledAttach'] += 1
            else:
                scores['wrongLabeledAttach'] += 1
        else:
            scores['wrongAttach'] += 1
            scores['wrongLabeledAttach'] += 1

        if node.drel.strip('%') == node.pdrel.strip('%'):
            scores['rightLabel'] += 1
        else:
            scores['wrongLabel'] += 1
    return scores
