from lib.arc_eager import ArcEager
import dynet as dy
import pickle
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import random


class Meta:
    def __init__(self):
        self.c_dim = 32  # character-rnn input dimension
        self.window = 2  # arc-eager feature window
        self.add_words = 1  # additional lookup for missing/special words
        self.p_hidden = 64  # pos-mlp hidden layer dimension
        self.n_hidden = 128  # parser-mlp hidden layer dimension
        self.lstm_wc_dim = 128  # LSTM (word-char concatenated input) output dimension
        self.lstm_char_dim = 64  # char-LSTM output dimension
        self.transitions = {'SHIFT':0,'LEFTARC':1,'RIGHTARC':2,'REDUCE':3}  # parser transitions
        ################################# STACKING-MODEL-DIMS ##################################
        self.xc_dim = 32
        self.xp_hidden = 64
        self.xn_hidden = 128
        self.xlstm_wc_dim = 128
        self.xlstm_char_dim = 64

class Parser(ArcEager):
    def __init__(self, model=None, meta=None, new_meta=None, test=False):
        self.model = dy.Model()
        if new_meta:
            self.meta = new_meta
        else:
            self.meta = pickle.load(open('%s.meta' %model, 'rb')) if model else meta

        # define pos-mlp
        self.ps_pW1 = self.model.add_parameters((self.meta.p_hidden, self.meta.lstm_wc_dim*2))
        self.ps_pb1 = self.model.add_parameters(self.meta.p_hidden)
        self.ps_pW2 = self.model.add_parameters((self.meta.n_tags, self.meta.p_hidden))
        self.ps_pb2 = self.model.add_parameters(self.meta.n_tags)

        # define parse-mlp
        self.pr_pW1 = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_wc_dim*2*self.meta.window))
        self.pr_pb1 = self.model.add_parameters(self.meta.n_hidden)
        self.pr_pW2 = self.model.add_parameters((self.meta.n_outs, self.meta.n_hidden))
        self.pr_pb2 = self.model.add_parameters(self.meta.n_outs)

        # define char-rnns
        self.bhcfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.bhcbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.ecfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.ecbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)

        # define base Bi-LSTM for input word sequence (takes word+char-rnn embeddings as input)
        self.fwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.lstm_char_dim*2, self.meta.lstm_wc_dim, self.model)
        self.bwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.lstm_char_dim*2, self.meta.lstm_wc_dim, self.model)

        # define Bi-LSTM for POS feature representation (takes base Bi-LSTM output as input)
        self.ps_fwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2, self.meta.lstm_wc_dim, self.model)
        self.ps_bwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2, self.meta.lstm_wc_dim, self.model)

        # define Bi-LSTM for parser feature representation (takes base Bi-LSTM output and pos-hidden-state as input)
        self.pr_fwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2+self.meta.p_hidden, self.meta.lstm_wc_dim, self.model)
        self.pr_bwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2+self.meta.p_hidden, self.meta.lstm_wc_dim, self.model)

        # pad-node for missing nodes in partial parse tree
        self.PAD = self.model.add_parameters(self.meta.lstm_wc_dim*2)

        # define lookup tables
        self.ELOOKUP_WORD = self.model.add_lookup_parameters((self.meta.n_words_eng, self.meta.w_dim))
        self.HLOOKUP_WORD = self.model.add_lookup_parameters((self.meta.n_words_hin, self.meta.w_dim))
        self.BLOOKUP_WORD = self.model.add_lookup_parameters((self.meta.n_words_ben, self.meta.w_dim))
        self.ELOOKUP_CHAR = self.model.add_lookup_parameters((self.meta.n_chars_eng, self.meta.c_dim))
        self.BHLOOKUP_CHAR = self.model.add_lookup_parameters((self.meta.n_chars_bh, self.meta.c_dim))
        # load pretrained embeddings
        if model is None:
            for word, V in ewvm.vocab.items():
                self.ELOOKUP_WORD.init_row(V.index+self.meta.add_words, ewvm.syn0[V.index])
            for word, V in hwvm.vocab.items():
                self.HLOOKUP_WORD.init_row(V.index+self.meta.add_words, hwvm.syn0[V.index])
            for word, V in bwvm.vocab.items():
                self.BLOOKUP_WORD.init_row(V.index+self.meta.add_words, bwvm.syn0[V.index])

        # load pretrained dynet model
        if not test and model:
            self.model.populate('%s.dy' %model)
        ######################################### STACKING ##############################################
        self.xps_pW1 = self.model.add_parameters((self.meta.xp_hidden, self.meta.xlstm_wc_dim*2))
        self.xps_pb1 = self.model.add_parameters(self.meta.xp_hidden)
        self.xps_pW2 = self.model.add_parameters((self.meta.xn_tags, self.meta.xp_hidden))
        self.xps_pb2 = self.model.add_parameters(self.meta.xn_tags)

        self.xpr_pW1 = self.model.add_parameters((self.meta.xn_hidden, self.meta.xlstm_wc_dim*2*self.meta.window+self.meta.n_hidden))
        self.xpr_pb1 = self.model.add_parameters(self.meta.xn_hidden)
        self.xpr_pW2 = self.model.add_parameters((self.meta.xn_outs, self.meta.xn_hidden))
        self.xpr_pb2 = self.model.add_parameters(self.meta.xn_outs)

        self.xbhcfwdRNN = dy.LSTMBuilder(1, self.meta.xc_dim, self.meta.xlstm_char_dim, self.model)
        self.xbhcbwdRNN = dy.LSTMBuilder(1, self.meta.xc_dim, self.meta.xlstm_char_dim, self.model)
        self.xecfwdRNN = dy.LSTMBuilder(1, self.meta.xc_dim, self.meta.xlstm_char_dim, self.model)
        self.xecbwdRNN = dy.LSTMBuilder(1, self.meta.xc_dim, self.meta.xlstm_char_dim, self.model)

        self.xps_fwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.xlstm_char_dim*2+self.meta.p_hidden, self.meta.xlstm_wc_dim, self.model)
        self.xps_bwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.xlstm_char_dim*2+self.meta.p_hidden, self.meta.xlstm_wc_dim, self.model)

        self.xpr_fwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2+self.meta.w_dim+self.meta.xlstm_char_dim*2+self.meta.xp_hidden,
                            self.meta.xlstm_wc_dim, self.model)
        self.xpr_bwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2+self.meta.w_dim+self.meta.xlstm_char_dim*2+self.meta.xp_hidden,
                            self.meta.xlstm_wc_dim, self.model)

        self.XPAD = self.model.add_parameters(self.meta.xlstm_wc_dim*2)
        if test and model:
            self.model.populate('%s.dy' %model)

    def enable_dropout(self):
        self.fwdRNN.set_dropout(0.3)
        self.bwdRNN.set_dropout(0.3)
        self.ecfwdRNN.set_dropout(0.3)
        self.ecbwdRNN.set_dropout(0.3)
        self.bhcfwdRNN.set_dropout(0.3)
        self.bhcbwdRNN.set_dropout(0.3)
        self.ps_fwdRNN.set_dropout(0.3)
        self.ps_bwdRNN.set_dropout(0.3)
        self.pr_fwdRNN.set_dropout(0.3)
        self.pr_bwdRNN.set_dropout(0.3)
        self.ps_W1 = dy.dropout(self.ps_W1, 0.3)
        self.ps_b1 = dy.dropout(self.ps_b1, 0.3)
        self.pr_W1 = dy.dropout(self.pr_W1, 0.3)
        self.pr_b1 = dy.dropout(self.pr_b1, 0.3)
        ########################################
        self.xecfwdRNN.set_dropout(0.3)
        self.xecbwdRNN.set_dropout(0.3)
        self.xbhcfwdRNN.set_dropout(0.3)
        self.xbhcbwdRNN.set_dropout(0.3)
        self.xps_fwdRNN.set_dropout(0.3)
        self.xps_bwdRNN.set_dropout(0.3)
        self.xpr_fwdRNN.set_dropout(0.3)
        self.xpr_bwdRNN.set_dropout(0.3)
        self.xps_W1 = dy.dropout(self.xps_W1, 0.3)
        self.xps_b1 = dy.dropout(self.xps_b1, 0.3)
        self.xpr_W1 = dy.dropout(self.xpr_W1, 0.3)
        self.xpr_b1 = dy.dropout(self.xpr_b1, 0.3)

    def disable_dropout(self):
        self.fwdRNN.disable_dropout()
        self.bwdRNN.disable_dropout()
        self.ecfwdRNN.disable_dropout()
        self.ecbwdRNN.disable_dropout()
        self.bhcfwdRNN.disable_dropout()
        self.bhcbwdRNN.disable_dropout()
        self.ps_fwdRNN.disable_dropout()
        self.ps_bwdRNN.disable_dropout()
        self.pr_fwdRNN.disable_dropout()
        self.pr_bwdRNN.disable_dropout()
        ################################
        self.xecfwdRNN.disable_dropout()
        self.xecbwdRNN.disable_dropout()
        self.xbhcfwdRNN.disable_dropout()
        self.xbhcbwdRNN.disable_dropout()
        self.xps_fwdRNN.disable_dropout()
        self.xps_bwdRNN.disable_dropout()
        self.xpr_fwdRNN.disable_dropout()
        self.xpr_bwdRNN.disable_dropout()

    def initialize_graph_nodes(self):
        #  convert parameters to expressions
        self.pad = dy.parameter(self.PAD)
        self.ps_W1 = dy.parameter(self.ps_pW1)
        self.ps_b1 = dy.parameter(self.ps_pb1)
        self.ps_W2 = dy.parameter(self.ps_pW2)
        self.ps_b2 = dy.parameter(self.ps_pb2)
        self.pr_W1 = dy.parameter(self.pr_pW1)
        self.pr_b1 = dy.parameter(self.pr_pb1)
        self.pr_W2 = dy.parameter(self.pr_pW2)
        self.pr_b2 = dy.parameter(self.pr_pb2)
        #######################################
        self.xpad = dy.parameter(self.XPAD)
        self.xps_W1 = dy.parameter(self.xps_pW1)
        self.xps_b1 = dy.parameter(self.xps_pb1)
        self.xps_W2 = dy.parameter(self.xps_pW2)
        self.xps_b2 = dy.parameter(self.xps_pb2)
        self.xpr_W1 = dy.parameter(self.xpr_pW1)
        self.xpr_b1 = dy.parameter(self.xpr_pb1)
        self.xpr_W2 = dy.parameter(self.xpr_pW2)
        self.xpr_b2 = dy.parameter(self.xpr_pb2)

        # apply dropout
        if self.eval:
            self.disable_dropout()
        else:
            self.enable_dropout()

        # initialize the RNNs
        self.f_init = self.fwdRNN.initial_state()
        self.b_init = self.bwdRNN.initial_state()

        self.cf_init_eng = self.ecfwdRNN.initial_state()
        self.cb_init_eng = self.ecbwdRNN.initial_state()
        self.cf_init_bh = self.bhcfwdRNN.initial_state()
        self.cb_init_bh = self.bhcbwdRNN.initial_state()
        ################################################
        self.xcf_init_eng = self.xecfwdRNN.initial_state()
        self.xcb_init_eng = self.xecbwdRNN.initial_state()
        self.xcf_init_bh = self.xbhcfwdRNN.initial_state()
        self.xcb_init_bh = self.xbhcbwdRNN.initial_state()

        self.ps_f_init = self.ps_fwdRNN.initial_state()
        self.ps_b_init = self.ps_bwdRNN.initial_state()
        ###############################################
        self.xps_f_init = self.xps_fwdRNN.initial_state()
        self.xps_b_init = self.xps_bwdRNN.initial_state()

        self.pr_f_init = self.pr_fwdRNN.initial_state()
        self.pr_b_init = self.pr_bwdRNN.initial_state()
        ###############################################
        self.xpr_f_init = self.xpr_fwdRNN.initial_state()
        self.xpr_b_init = self.xpr_bwdRNN.initial_state()

    def word_rep_eng(self, w):
        if not self.eval and random.random() < 0.3:
            return self.ELOOKUP_WORD[0]
        idx = self.meta.ew2i.get(w, self.meta.ew2i.get(w.lower(), 0))
        return self.ELOOKUP_WORD[idx]

    def word_rep_hin(self, w):
        if not self.eval and random.random() < 0.3:
            return self.HLOOKUP_WORD[0]
        idx = self.meta.hw2i.get(w, 0)
        return self.HLOOKUP_WORD[idx]

    def word_rep_ben(self, w):
        if not self.eval and random.random() < 0.3:
            return self.BLOOKUP_WORD[0]
        idx = self.meta.bw2i.get(w, 0)
        return self.BLOOKUP_WORD[idx]


    def char_rep_eng(self, w, f, b):
        no_c_drop = False
        if self.eval or random.random()<0.9:
            no_c_drop = True
        bos, eos, unk = self.meta.ec2i["bos"], self.meta.ec2i["eos"], self.meta.ec2i['unk']
        char_ids = [bos] + [self.meta.ec2i.get(c, unk) if no_c_drop else unk for c in w] + [eos]
        char_embs = [self.ELOOKUP_CHAR[cid] for cid in char_ids]
        fw_exps = f.transduce(char_embs)
        bw_exps = b.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def char_rep_bh(self, w, f, b):
        no_c_drop = False
        if self.eval or random.random()<0.9:
            no_c_drop = True
        bos, eos, unk = self.meta.bhc2i["bos"], self.meta.bhc2i["eos"], self.meta.bhc2i["unk"]
        char_ids = [bos] + [self.meta.bhc2i.get(c, unk) if no_c_drop else unk for c in w] + [eos]
        char_embs = [self.BHLOOKUP_CHAR[cid] for cid in char_ids]
        fw_exps = f.transduce(char_embs)
        bw_exps = b.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def get_char_embds(self, sentence, bhf, bhb, ef, eb):
        char_embs = []
        for node in sentence:
            if node.lang in ['hi', 'bn']:
                char_embs.append(self.char_rep_bh(node.form, bhf, bhb))
            else:
                char_embs.append(self.char_rep_eng(node.form, ef, eb))
        return char_embs

    def get_word_embds(self, sentence):
        word_embs = []
        for node in sentence:
            if node.lang == 'hi':
                word_embs.append(self.word_rep_hin(node.form))
            elif node.lang == 'bn':
                word_embs.append(self.word_rep_ben(node.form))
            else:
                word_embs.append(self.word_rep_eng(node.form))
        return word_embs

    def basefeaturesEager(self, nodes, stack, i):
	    #NOTE Stack nodes
        s0 = nodes[stack[-1]] if stack else nodes[0].left
        #NOTE Buffer nodes
        n0 = nodes[ i ] if nodes[ i: ] else nodes[0].left
        return [(nd.id, nd.form) for nd in [s0,n0]]

    def feature_extraction(self, sentence):
        self.initialize_graph_nodes()

        # get word/char embeddings
        wembs = self.get_word_embds(sentence)
        cembs = self.get_char_embds(sentence, self.cf_init_bh, self.cb_init_bh, self.cf_init_eng, self.cb_init_eng)
        lembs = [dy.concatenate([w,c]) for w,c in zip(wembs, cembs)]
        # feed word vectors into base biLSTM
        fw_exps = self.f_init.transduce(lembs)
        bw_exps = self.b_init.transduce(reversed(lembs))
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

        # feed biLSTM embeddings into POS biLSTM (pretrained)
        ps_fw_exps = self.ps_f_init.transduce(bi_exps)
        ps_bw_exps = self.ps_b_init.transduce(reversed(bi_exps))
        ps_bi_exps = [dy.concatenate([f,b]) for f,b in zip(ps_fw_exps, reversed(ps_bw_exps))]

        # get pos-hidden representation and pos loss
        pos_errs, pos_hidden = [], []
        for xi,node in zip(ps_bi_exps, sentence):
            xh = self.ps_W1 * xi
            pos_hidden.append(xh)

        # get word/char embeddings (stacked)
        xcembs = self.get_char_embds(sentence, self.xcf_init_bh, self.xcb_init_bh,
                                               self.xcf_init_eng, self.xcb_init_eng)
        xwcp_exps = [dy.concatenate([w,c,p]) for w,c,p in zip(wembs, xcembs, pos_hidden)]
        xfw_exps = self.xps_f_init.transduce(xwcp_exps)
        xbw_exps = self.xps_b_init.transduce(reversed(xwcp_exps))
        xbi_exps = [dy.concatenate([f,b]) for f,b in zip(xfw_exps, reversed(xbw_exps))]

        pos_errs, xpos_hidden = [], []
        for xi,node in zip(xbi_exps, sentence):
            xh = self.xps_W1 * xi
            xpos_hidden.append(xh)
            xo = self.xps_W2*xh + self.xps_b2
            err = dy.softmax(xo).npvalue() if self.eval else dy.pickneglogsoftmax(xo, self.meta.p2i[node.tag])
            pos_errs.append(err)

        # concatenate pos hidden-layer with base biLSTM
        wcp_exps = [dy.concatenate([w,p]) for w,p in zip(bi_exps, pos_hidden)]
        # feed concatenated embeddings into parse biLSTM
        pr_fw_exps = self.pr_f_init.transduce(wcp_exps)
        pr_bw_exps = self.pr_b_init.transduce(reversed(wcp_exps))
        pr_bi_exps = [dy.concatenate([f,b]) for f,b in zip(pr_fw_exps, reversed(pr_bw_exps))]

        xwcp_exps = [dy.concatenate(list(z)) for z in zip(wembs, xcembs, pr_bi_exps, xpos_hidden)]
        xbi_fw_exps = self.xpr_f_init.transduce(xwcp_exps)
        xbi_bw_exps = self.xpr_b_init.transduce(reversed(xwcp_exps))
        xpr_bi_exps = [dy.concatenate([f,b]) for f,b in zip(xbi_fw_exps, reversed(xbi_bw_exps))]

        return pr_bi_exps, xpr_bi_exps, pos_errs
