#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from GL2 import GLJudge2 as GLJudge
from PoetryTool import PoetryTool

import numpy as np
import tensorflow as tf

from model import PoemModel
from model import HParams
from Yun import Yun
import codecs
import os
import pickle

class Generator(object):
    """construction for PoemTrainer"""

    def __init__(self, vocab, ivocab, hps, model, sess):

        self.model = model
        self.vocab = vocab
        self.ivocab = ivocab
        self.sess = sess

        self.PAD_ID = self.vocab['PAD']
        self.GO_ID = self.vocab['GO']
        self.EOS_ID = self.vocab['</S>']
        self.UNK_ID = self.vocab['UNK']

        self.sen_len = hps.buckets[0][0]

        # loading addentional data
        self.__getping()
        self.__getze()
        self.__getyun()

        # GL types
        # the type of the first sentence can determine the whole types
        self.GLTYPE = [[0, 1, 3, 2], [1, 2, 0, 1], [2, 1, 3, 2], [3, 2, 0, 1]]
        self.SENGL = {7: [[-1, 1, -1, 0, 0, 1, 1], [-1, 0, -1, 1, 1, 0, 0], [-1, 1, 0, 0, 1, 1, 0], [-1, 0, -1, 1, 0, 0, 1]],
                      5: [[-1, 0, 0, 1, 1], [-1, 1, 1, 0, 0], [0, 0, 1, 1, 0], [-1, 1, 0, 0, 1]]}

        print("laoding GL judger...")
        self.__GL = GLJudge()

        self.prob_esp = 1e-10  # the minimum value of problity

        self.tool = PoetryTool()

        #load poemlib
        print ("load poemlib...")
        self.__poemLib = {}
        fin = open("data/other/DuplicateCheckLib.txt")
        lines = fin.readlines()
        fin.close()
        for line in lines:
            line = line.strip()
            self.__poemLib[line] = 1
        
        # load dic for poem model
        vocab_file = open('data/ivocab.pkl', 'rb')
        self.idic = pickle.load(vocab_file,encoding='utf8')
        vocab_file.close()

        vocab_file = open('data/vocab.pkl', 'rb')
        self.dic = pickle.load(vocab_file,encoding='utf8')
        vocab_file.close()

        #self.scorer = LMScore()
    
    def __getze(self):
        f = open("data/other/zesheng.txt", 'r')
        self.zelist = []
        ze = f.read()
        # get ping-toned char list
        for key, val in self.vocab.items():
            if key in ze:
                self.zelist.append(int(val))
        f.close()

    def __getping(self):
        f = open("data/other/pingsheng.txt", 'r')
        self.pinglist = []
        ping = f.read()
        # get ping-toned char list
        for key, val in self.vocab.items():
            if key in ping:
                self.pinglist.append(int(val))
        f.close()

    def __getyun(self):
        print("loading yun dict...")
        f = open("data/other/YunList.txt", 'r')
        line = f.readline()
        self.yundic = {}
        self.iyundic = {}
        while line:
            para = line.split(" ")
            if para[0] not in self.vocab:
                line = f.readline()
                continue

            val = self.vocab[para[0]]
            key = int(para[1])

            if val not in self.iyundic:
                self.iyundic[val] = key

            if key in self.yundic:
                self.yundic[key].append(val)
            else:
                temp = []
                temp.append(val)
                self.yundic[key] = temp
            line = f.readline()

        f.close()

    def addtionFilter(self, trans, pos, yan):
        pos -= 1
        preidx = range(0, pos)

        batch_size = len(trans)
        # print(batch_size)

        forbidden_list = [[] for i in range(batch_size)]

        for i in range(0, batch_size):  # iter ever batch
            prechar = [trans[i][c] for c in preidx]

            newprechar = []
            if pos % 2 != 0:# or (pos == yan-1 and trans[i][pos-1] != trans[i][pos-2]):
                okpredix = trans[i][pos-1]
                for c in prechar:
                    if c != okpredix:
                        newprechar.append(c)
            else:
                newprechar = prechar

            forbidden_list[i] = newprechar

        return forbidden_list

    def beam_select(self, probs, trans, yan, k, beam_size, repeatidxvec, gls, yun):
        V = np.shape(probs)[1]  # vocabulary size
        n_samples = np.shape(probs)[0]
        if k == 1:
            n_samples = beam_size

        # trans_indices, word_indices, costs
        hypothesis = []  # (char_idx, which beam, prob)

        cost_eps = 1e5
        # control inner repeat

        forbidden_list = self.addtionFilter(trans, k, yan)
        for i in range(0, np.shape(probs)[0]):
            probs[i, forbidden_list[i]] = cost_eps

        # control global repeat
        probs[:, repeatidxvec] = cost_eps

        # control yun
        if yun != -1 and k == yan and yun in self.yundic:
            probs *= cost_eps
            probs[:, self.yundic[yun]] /= float(cost_eps)

        # control gls
        if k <= yan and gls != -1:
            gl = gls[k-1]

            if gl == 0:  # ping
                probs[:, self.zelist] = cost_eps
            elif gl == 1:  # ze
                probs[:, self.pinglist] = cost_eps

        flat_next_costs = probs.flatten()
        best_costs_indices = np.argpartition(
            flat_next_costs.flatten(), n_samples)[:n_samples]

        trans_indices = [int(idx)
                         for idx in best_costs_indices / V]  # which beam line
        word_indices = best_costs_indices % V
        costs = flat_next_costs[best_costs_indices]

        for i in range(0, n_samples):
            hypothesis.append((word_indices[i], trans_indices[i], costs[i]))

        return hypothesis

    def beam_search(self, sess, sen, beam_size, encoder_mask, senlen, repeatidxvec, gls, batchTopic, batchYun, yun=-1):
        encoder_mask_unpacked = np.array(encoder_mask)[:, :, 0]
        # generate the final state of encoder, and the attention_states
        encoder_state, attn_states = self.model.encoder_state_computer(
            sess, sen, encoder_mask)

        inputlen = np.shape(attn_states)[1]
        fin_attn_states = attn_states

        fin_trans = []
        fin_costs = []
        fin_align = []

        trans = [[] for i in range(beam_size)]
        costs = [0.0]
        align = []
        for i in range(beam_size):
            align.append(np.array([np.zeros(inputlen)], dtype=np.float32))

        n_samples = beam_size
        dim = self.model.hidden_size

        sess_tmp = tf.InteractiveSession(graph=tf.Graph())
        state_c = tf.concat([encoder_state[0],batchTopic,batchYun], 1)
        state_h = tf.concat([encoder_state[1],batchTopic,batchYun], 1)
        state = sess_tmp.run(tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h))
        sess_tmp.close()

        inp = np.array([self.vocab['GO'] for _ in range(beam_size)])
        #print(batchTopic.shape)
        output, state, alignments = self.model.decoder_state_output_computer(
            sess, inp, state, attn_states, encoder_mask_unpacked, batchYun)

        for k in range(1, 5*len(sen)):

            if n_samples == 0:
                break

            output = self.tool.softmax(output)
            output += 1e-5

            if k == 1:
                output = output[0, :]

            log_probs = np.log(output)
            #print(log_probs.shape)
           
            next_costs = np.array(costs)[:, None] - log_probs

            # Form a beam for the next iteration
            new_trans = [[] for i in range(0, n_samples)]
            #print(type(n_samples))
            new_costs = np.zeros(n_samples)
            new_states_c = np.zeros((n_samples, dim+self.model.num_topic+30+1), dtype="float32")
            new_states_m = np.zeros((n_samples, dim+self.model.num_topic+30+1), dtype="float32")

            new_align = [[] for i in range(n_samples)]

            inputs = np.zeros(n_samples, dtype="int64")
            hypothesis = self.beam_select(
                next_costs, trans, senlen, k, beam_size, repeatidxvec, gls, yun)

            for i, (next_word, orig_idx, next_cost) in enumerate(hypothesis):
                # print("%d %d %d %f" % (i, next_word, orig_idx, next_cost))
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                new_align[i] = np.concatenate(
                    (align[orig_idx], [alignments[orig_idx, :]]), axis=0)

                new_states_c[i] = state[0][orig_idx, :]
                new_states_m[i] = state[1][orig_idx, :]
                inputs[i] = next_word

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            align = []

            for i in range(n_samples):
                if new_trans[i][-1] != self.EOS_ID:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                    align.append(new_align[i])
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
                    fin_align.append(new_align[i])
            if len(indices)==0:
                break
            inputs = inputs[indices]
            new_states_c = new_states_c[indices]
            new_states_m = new_states_m[indices]
            attn_states = attn_states[indices, :, :]
            encoder_mask_unpacked = encoder_mask_unpacked[indices, :]
            batchTopic = batchTopic[indices,:]
            batchYun = batchYun[indices,:]

            new_states = tf.nn.rnn_cell.LSTMStateTuple(
                new_states_c, new_states_m)

            output, state, alignments= self.model.decoder_state_output_computer(
                sess, inputs, new_states, attn_states, encoder_mask_unpacked, batchYun)

        for i in range(len(fin_align)):
            talign = fin_align[i]
            fin_align[i] = talign[1:, :]

        index = np.argsort(fin_costs)
        tfin_align = []
        for i in range(0, len(index)):
            tfin_align.append(fin_align[index[i]])
        fin_align = tfin_align

        fin_trans = np.array(fin_trans)[index]
        fin_costs = np.array(sorted(fin_costs))

        if len(fin_trans) == 0:
            index = np.argsort(costs)
            fin_align = np.array(align)[index]
            fin_trans = np.array(trans)[index]
            fin_costs = np.array(sorted(costs))

        return fin_trans, fin_costs, fin_align, fin_attn_states

    def glFilter(self, trans, costs, align, gltyp, senlen):

        new_trans = []
        new_costs = []
        new_align = []

        for i in range(len(trans)):
            if len(trans[i]) < senlen:
                continue
            tran = trans[i][0:senlen]
            sen = self.tool.idxes2senlist(tran, self.ivocab)
            sen = "".join(sen)
            if sen in self.__poemLib:
                continue
            sen2 = sen
            gl = self.__GL.gelvJudge(sen2)

            if gltyp != -1 and gl != gltyp:
                continue

            if gltyp == -1 and gl < 0:
                continue

            new_trans.append(trans[i])
            new_costs.append(costs[i])
            new_align.append(align[i])

        return new_trans, new_costs, new_align

    def getGLbyIds(self, idxes):
        sen = self.tool.idxes2senlist(idxes, self.ivocab)
        sen = "".join(sen)
        sen2 = sen
        gl = self.__GL.gelvJudge(sen2)
        return gl

    def getYun(self, idxes):
        tail = idxes[-1]
        # print tail
        if tail in self.iyundic:
            return self.iyundic[tail]
        else:
            return -1


    def generate_one(self, sentence, manu_topic=-1, beam_size=20, all_topic=False, manu=False): #all_topic: if true, use all generated sentences for next topic
        sentence = sentence.strip()
        
        thisYun=int(self.model.yunjiao.getYun(sentence)[0])
        if thisYun<0:
            thisYun=0
        print(sentence)
        print(thisYun)
        
        ans = []
        repeatidx = []
        ans.append(sentence)

        # generate the second line
        #______________________________
        sentence = self.tool.lineSplit2list(sentence)
        print(sentence)
        sen = self.tool.senvec2idxes(sentence, self.vocab)
        inputlen = len(sen)

        repeatidx.extend(sen)
        yun = self.getYun(sen)
        gl = self.getGLbyIds(sen)
        #gl = -1

        if gl == -1:
            print(
                "The sentence #%s# you input does'n obey gl, Please input again!! ", ans[0])
            # continue
            gl = 0

        gltypes = self.GLTYPE[gl]
        all_sen = sen # all sentences include generated ones
        
        #thisYun = BatchYun([sen],self.ivocab,self.vocab['PAD'])
        
        batch_sen_nopad = []
        for length_idx in range(len(sen)):
            batch_sen_nopad.append(np.array([sen[length_idx] for _ in range(beam_size)], dtype=np.int32))
        embs = self.sess.run(tf.add_n([tf.nn.embedding_lookup(self.model.encoder_embedding, x) for x in batch_sen_nopad]))
        topics = self.model.classifier_state_computer_simple(self.sess, embs, np.array([len(embs)]))
        topics_pooling = np.zeros_like(topics)
        amax = np.argmax(topics,axis=1)
        for i in range(beam_size):
            topics_pooling[i][amax[i]] = 1.0
        batchTopic = topics_pooling
        if manu_topic != -1:
            batchTopic = np.zeros_like(batchTopic)
            batchTopic[0,manu_topic]=1
        print(batchTopic[0])
        
        for step in range(1, 4):
            print("generating %d line..." % (step+1))
            
            batch_sen, encoder_mask, encoder_lda = self.tool.gen_batch_beam(
                sen, self.sen_len, self.PAD_ID, self.GO_ID, self.EOS_ID, self.UNK_ID, self.ivocab, all_sen, all_topic, beam_size)

            #batchTopic = self.model.inferTopic(encoder_lda)
                        
            batchYun = self.model.yunjiao.getBatchYun(batch_sen,self.ivocab,self.vocab['PAD'])
            batchYun = np.zeros_like(batchYun)
            numBatch = batchYun.shape[0]
            if step == 2:                
                for iter in range(numBatch):
                    batchYun[iter,0] = 1.0
            else:
                for iter in range(numBatch):
                    batchYun[iter,thisYun] = 1.0
            # print (batch_sen)
            if step == 2:
                current_yun = -1
            elif step == 0:
                current_yun = -1
            else:
                current_yun = yun
            #print (current_yun)

            gls = self.SENGL[inputlen][gltypes[step]]
            if step==0:
                gls=-1

            trans, costs, align,  attn_states = self.beam_search(
                self.sess, batch_sen, beam_size, encoder_mask, inputlen, repeatidx, gls, batchTopic, batchYun, current_yun)
            
            if step == 0:
                trans, costs, align = self.glFilter(
                trans, costs, align, -1, inputlen)
            else:
                trans, costs, align = self.glFilter(trans, costs, align, gltypes[step], inputlen)

            if len(trans) == 0:
                return [], ("line %d generation failed!" % (step+1))

            which = 0
            if manu:
                for i in range(0, len(trans)):
                    sen = self.tool.beam_get_sentence(
                        trans[i], self.ivocab, self.EOS_ID)
                    print("%d  sen:  %s   cost:  %f" % (i, sen, costs[i]))
                which = input(
                    "Please input select the %d sentence: \n" % (step+1))

            sentence = self.tool.beam_get_sentence(
                trans[which], self.ivocab, self.EOS_ID)
            sentence = sentence.strip()
            if step == 1:
                thisYun = int(self.model.yunjiao.getYun(sentence)[0])
            if thisYun<0:
                thisYun=0
            sentmp = self.tool.lineSplit2list(sentence)
            sentence = "".join([sentmp[ch] for ch in range(inputlen)])
            ans.append(sentence)

            if step==3:
                return ans, "ok"

            #sentence = sentences[step] #comment if generate 3 sentences directly
            sentence = self.tool.lineSplit2list(sentence)
            sen = self.tool.senvec2idxes(sentence, self.vocab)
            repeatidx = list(set(repeatidx).union(set(sen)))

            if yun == -1 and step == 1:
                yun = self.getYun(sen)
            if step == 0:
                yun = self.getYun(sen)
            all_sen = all_sen+sen
        return ans, "ok"

def main(_):
    pass

if __name__ == "__main__":
    tf.app.run()
