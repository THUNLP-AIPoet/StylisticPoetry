# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import time
from collections import namedtuple


import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

from Yun import Yun
'''
some global function copy from tensorflow.seq2seq
'''
HParams = namedtuple('HParams',
                     'vocab_size, emb_size, hidden_size,'
                     'device, learning_rate, '
                     'max_gradient_norm, buckets, batch_size, num_topic, mode')


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        average_across_timesteps: If set, divide the returned cost by the total label weight.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
          to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, default: "sequence_loss_by_example".

        Returns:
            1D batch-sized float Tensor: The log-perplexity for each sequence.

        Raises:
            ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
            "%d, %d, %d." % (len(logits), len(weights), len(targets)))

    with ops.name_scope("sequence_loss_by_example"):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logit,labels= target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)

        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size

    return log_perps

def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):

    with ops.name_scope("sequence_loss"):
        cost = math_ops.reduce_sum(sequence_loss_by_example(logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))

        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, cost.dtype)
        else:
            return cost

class PoemModel(object):
    def __init__(self, hps):
        # Create the model
        self.yunjiao = Yun()
        self.num_topic = hps.num_topic
        self.batchYun = tf.placeholder(tf.float32, shape=[None,30+1])
        self.vocab_size = hps.vocab_size
        self.emb_size = hps.emb_size
        self.hidden_size = hps.hidden_size
        self.device = hps.device
        self.learning_rate = hps.learning_rate
        self.learning_rate_init = self.learning_rate
        self.max_gradient_norm = hps.max_gradient_norm
        self.buckets = hps.buckets
        self.batch_size = hps.batch_size
        self.mode = hps.mode

        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        output_projection = None
        softmax_loss_function = None
        
        decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size+self.num_topic+30+1)       
        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob = self.keep_prob)
        self.balance = tf.placeholder(tf.float32)

        # build placeholders
        self.encoder_inputs = []

        for i in range(self.buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

        self.decoder_inputs = []
        for i in range(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],name="decoder{0}".format(i)))

        self.encoder_mask = tf.placeholder(tf.float32, shape=[None, self.buckets[0][0], 1], name="encoder_mask")
        self.encoder_mask_unpacked = tf.placeholder(tf.float32, shape=[None, self.buckets[0][0]], name="encoder_mask_unpacked")

        #self.topicEmbed= tf.Variable(tf.random_normal([self.num_topic,self.hidden_size], stddev=0.35),name="topicEmbed")
        self.inp = tf.placeholder(tf.int32, shape=[None], name="inp")

        if self.mode == 'decode': # for generating
            # Remember! Since we use the bidirectional_rnn, so encoder state is of double size 
            self.attentions = tf.placeholder(tf.float32, shape=[None, self.buckets[0][0], self.hidden_size*2], name="attentions")
            self.prev_state = (tf.placeholder(tf.float32, shape=[None, self.hidden_size+self.num_topic+30+1] , name="prev_state_c"),
                                                        tf.placeholder(tf.float32, shape=[None, self.hidden_size+self.num_topic+30+1] , name="prev_state_m") )            
        else: # training
            self.target_weights = []
            for i in range(self.buckets[-1][1] + 1):
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            self.targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]


        self.embs_sum = tf.placeholder(tf.float32, shape=[None,self.emb_size])
        self.embs_len = tf.placeholder(tf.float32, shape=[1,])
        # input and output use the same word embedding
        #with tf.variable_scope('word_embedding_encoder'), tf.device('/cpu:0'):
        self.encoder_embedding = tf.get_variable('enc_embedding', [self.vocab_size, self.emb_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))

        #with tf.variable_scope('word_embedding_decoder'), tf.device('/cpu:0'):
        self.decoder_embedding = tf.get_variable('dec_embedding', [self.vocab_size, self.emb_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))

        self.emb_encoder_inputs = [tf.nn.embedding_lookup(self.encoder_embedding, x) for x in self.encoder_inputs]
        self.emb_decoder_inputs = [tf.nn.embedding_lookup(self.decoder_embedding, x) for x in self.decoder_inputs]
        self.einp = tf.nn.embedding_lookup(self.decoder_embedding, self.inp) 

        print ("using device: %s" % self.device)
        with tf.device(self.device):
            if self.mode == 'train': # training
                self.outputs, self.loss, self.loss_class, self.debug = self.__build_seq2seq(decoder_cell, self.emb_encoder_inputs[0:self.buckets[0][0]], 
                    self.emb_decoder_inputs[0:self.buckets[0][1]], self.targets, self.target_weights, self.encoder_mask, self.buckets[0], self.batchYun)

                params = tf.trainable_variables()
                #opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=0.95, epsilon=1e-06)
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.gradient_norm = norm         
                self.update = opt.apply_gradients( zip(clipped_gradients, params), global_step=self.global_step)

            else:
                self.encoder_state, self.attention_states = self.__build_encoder_state_computer(self.emb_encoder_inputs, self.encoder_mask)
                self.next_output, self.next_state, self.next_align = self.__build_decoder_state_output_computer(decoder_cell, self.attentions, self.einp, self.prev_state, self.encoder_mask_unpacked, self.batchYun)
        self.infer_score = self.__build_classifier_state_computer_simple(self.embs_sum,self.embs_len)

        # saver
        self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V1)
    '''
    def build_classifier_state_computer(self, emb_encoder_inputs):


        with variable_scope.variable_scope(variable_scope.get_variable_scope(),  reuse=None):
            with variable_scope.variable_scope("seq2seq_Classifier"):
                encoder_cell_fw  = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                encoder_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

                encoder_cell_fw =  tf.nn.rnn_cell.DropoutWrapper(encoder_cell_fw, output_keep_prob= self.keep_prob)
                encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper(encoder_cell_bw, output_keep_prob= self.keep_prob)

                (outputs , encoder_state_fw, encoder_state_bw)  = rnn.static_bidirectional_rnn(
                    encoder_cell_fw, encoder_cell_bw, emb_encoder_inputs, dtype=tf.float32)

                encoder_outputs = outputs


                encoder_state_c =  encoder_state_bw[0]
                encoder_state_m = encoder_state_bw[1]

                with variable_scope.variable_scope("initial_transfor_c"):
                    final_state_c = core_rnn_cell._linear(encoder_state_c,  self.hidden_size, True)
                    final_state_c  = tf.tanh(final_state_c)

                with variable_scope.variable_scope("initial_transfor_m"):
                    final_state_m = core_rnn_cell._linear(encoder_state_m, self.hidden_size, True)
                    final_state_m  = tf.tanh(final_state_m)
                
                final_state = tf.nn.rnn_cell.LSTMStateTuple(final_state_c, final_state_m)

                infer_score = core_rnn_cell._linear(final_state, self.num_topic, True)
                return infer_score
    '''

    def __build_classifier_state_computer_simple(self, emb_encoder_inputs_sum, length):
        infer_score = core_rnn_cell._linear(emb_encoder_inputs_sum / length, self.num_topic, True)
        return infer_score

    def classifier_state_computer_simple(self, sess, emb_encoder_inputs_sum, emb_len):
        input_feed = {}
        input_feed[self.embs_sum]= emb_encoder_inputs_sum
        input_feed[self.embs_len]= emb_len
        return sess.run(self.infer_score, input_feed)

    def __build_encoder_state_computer(self, emb_encoder_inputs, encoder_mask):
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),  reuse=None):
            with variable_scope.variable_scope("seq2seq_Encoder"):
                encoder_cell_fw  = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                encoder_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

                encoder_cell_fw =  tf.nn.rnn_cell.DropoutWrapper(encoder_cell_fw, output_keep_prob= self.keep_prob)
                encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper(encoder_cell_bw, output_keep_prob= self.keep_prob)

                (outputs , encoder_state_fw, encoder_state_bw)  = rnn.static_bidirectional_rnn(
                    encoder_cell_fw, encoder_cell_bw, emb_encoder_inputs, dtype=tf.float32)

                encoder_outputs = outputs


                encoder_state_c =  encoder_state_bw[0]
                encoder_state_m = encoder_state_bw[1]

                with variable_scope.variable_scope("initial_transfor_c"):
                    final_state_c = core_rnn_cell._linear(encoder_state_c,  self.hidden_size, True)
                    final_state_c  = tf.tanh(final_state_c)

                with variable_scope.variable_scope("initial_transfor_m"):
                    final_state_m = core_rnn_cell._linear(encoder_state_m, self.hidden_size, True)
                    final_state_m  = tf.tanh(final_state_m)
                
                final_state = tf.nn.rnn_cell.LSTMStateTuple(final_state_c, final_state_m)


                # First calculate a concatenation of encoder outputs to put attention on.
                # cell.output_size is embedding_size
                top_states = [array_ops.reshape(e, [-1, 1, encoder_cell_fw.output_size*2]) for e in encoder_outputs]

                attention_states = array_ops.concat(top_states, 1)
              
                final_attention_states = tf.multiply(encoder_mask, attention_states)
                return final_state, final_attention_states

    def encoder_state_computer(self, session, encoder_inputs, encoder_mask):
        encoder_size = len(encoder_inputs) 
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

        input_feed[self.encoder_mask.name] = encoder_mask
        input_feed[self.keep_prob] = 1.0
        output_feed = [self.encoder_state, self.attention_states]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]  # encoder_state, attention_states

    def __build_decoder_state_output_computer(self, cell, attentions, inp, prev_state, encoder_mask, batchYun, num_heads=1):
        '''
        attentions: encoder states  
        '''
        if num_heads < 1:
            raise ValueError("With less than 1 heads, use a non-attention decoder.")
        with variable_scope.variable_scope("seq2seq_Decoder"):
            output_size = self.vocab_size # num_decoder_symbols is vocabulary size
            if not attentions.get_shape()[1:3].is_fully_defined():
                raise ValueError("Shape[1] and [2] of attention_states must be known: %s" % attentions.get_shape())

            batch_size = array_ops.shape(attentions)[0]
            attn_length = attentions.get_shape()[1].value  # the length of a input sentence
            attn_size = attentions.get_shape()[2].value  # hidden state size of encoder, that is 2*size
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            # Remember we use bidirectional RNN 
            hidden = array_ops.reshape(attentions, [-1, attn_length, 1, attn_size]) 
            hidden_features = []
            v = []
            # Size of query vectors for attention
            # query vector is decoder state
            attention_vec_size = attn_size

            for a in range(num_heads):
                k = variable_scope.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
                hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

            # calculate attention
            def attention(query):
                """Put attention masks on hidden using hidden_features and query."""
                ds = []  # Results of attention reads will be stored here.
                if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                    query_list = nest.flatten(query)
                    for q in query_list:  # Check that ndims == 2 if specified.
                        ndims = q.get_shape().ndims
                        if ndims:
                            assert ndims == 2
                    query = array_ops.concat(query_list, 1)

                for a in range(num_heads):
                    with variable_scope.variable_scope("Attention_%d" % a):
                        y = core_rnn_cell._linear(query, attention_vec_size, True)
                        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = math_ops.reduce_sum( v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3]) 
                        a = nn_ops.softmax(s)
                        #a = a + 1e-5
                        
                        a1 = tf.multiply(a, encoder_mask)
                        #print (mask_a.get_shape())
                        floor = math_ops.reduce_sum(a1, axis = 1)
                        floor = tf.stack([floor], axis = 1)

                        #print (floor.get_shape())
                        a2 = tf.truediv(a1, floor)
                        nan_bool = tf.is_nan(a2)
                        #mask_a = tf.select(nan_bool, a1+0.1, a2)
                        mask_a = a2

    
                        #print (mask_a.get_shape())
                        #print ("_____________")

                        # Now calculate the attention-weighted vector d.
                        d = math_ops.reduce_sum( array_ops.reshape(mask_a, [-1, attn_length, 1, 1]) * hidden, [1, 2])

                        ds.append(array_ops.reshape(d, [-1, attn_size]))  #remember this size
                return ds, mask_a

            # calculate one step
            prev_attns, _ = attention(prev_state)


            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            # Merge input and previous attentions into one vector of the right size.
            x = core_rnn_cell._linear([inp] + prev_attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, prev_state)   
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                attns, align = attention(state)


            with variable_scope.variable_scope("AttnOutputProjection"):
                #output_lm = tf.nn.softmax(rnn_cell_impl._linear([cell_output] + attns, output_size, True))
                output_lm = core_rnn_cell._linear([cell_output] + attns, output_size, True)

            return output_lm, state, align

    def decoder_state_output_computer(self, sess, prev_inp, prev_state, attention_states, encoder_mask_unpacked, batchYun):
        input_feed = {}

        input_feed[self.prev_state[0].name] = prev_state[0]
        input_feed[self.prev_state[1].name] = prev_state[1]
        
        input_feed[self.attentions.name] = attention_states
        input_feed[self.inp.name] = prev_inp
        input_feed[self.keep_prob] = 1.0
        input_feed[self.encoder_mask_unpacked.name] = encoder_mask_unpacked
        input_feed[self.batchYun] = batchYun
        
        if not self.attentions.get_shape()[1:3].is_fully_defined():
          raise ValueError("Shape[1] and [2] of attention_states must be known: %s" % self.attentions.get_shape())
          
        
        output_feed = [self.next_output, self.next_state, self.next_align]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0], outputs[1], outputs[2] # next_output, next_state, next_align

    def infer_loss(self, decoder_inputs, weights, expected_seq_j, col):
        seqtmp=[]
        for i in range(len(decoder_inputs)):
            seqtmp.append(tf.multiply(tf.matmul(expected_seq_j[i],tf.stop_gradient(self.decoder_embedding)), tf.stop_gradient(tf.tile(tf.expand_dims(weights[i],-1), [1,self.emb_size]))))
        retloss = -tf.reduce_mean(tf.slice(tf.transpose(tf.nn.log_softmax(self.__build_classifier_state_computer_simple(tf.add_n(seqtmp), len(seqtmp)))),[col,0],[1,-1]))
        return retloss
    def __build_seq2seq(self, decoder_cell, encoder_inputs, decoder_inputs, targets, weights, encoder_mask, bucket, batchYun):
        
        with tf.variable_scope("Find"):
            decoder_inputs_simp = decoder_inputs
            for i in range(len(decoder_inputs)):
                decoder_inputs_simp[i] = tf.multiply(decoder_inputs[i],tf.tile(tf.expand_dims(weights[i],-1), [1,self.emb_size]))
            ifScore = self.__build_classifier_state_computer_simple(tf.add_n(decoder_inputs_simp[1:]),len(decoder_inputs_simp[1:]))
            ifTopic = tf.one_hot(tf.argmax(input=ifScore,dimension=1),depth=self.num_topic)

        encoder_state, attention_states  = self.__build_encoder_state_computer(encoder_inputs, encoder_mask)
        state_c = tf.concat([encoder_state[0],ifTopic,batchYun], 1)
        state_h = tf.concat([encoder_state[1],ifTopic,batchYun], 1)
        state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
        attn_weights = []

        encoder_mask_unpack =  tf.unstack(encoder_mask, axis = 2)[0]

        decoder_outputs = []
        for i, inp in enumerate(decoder_inputs):
            if i>0:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    output, state, align = self.__build_decoder_state_output_computer(decoder_cell, attention_states, inp, state, encoder_mask_unpack, batchYun)
            else:
                output, state, align = self.__build_decoder_state_output_computer(decoder_cell, attention_states, inp, state, encoder_mask_unpack, batchYun)

            decoder_outputs.append(output)
            attn_weights.append([align])

        loss = sequence_loss(decoder_outputs, targets[: bucket[1]], weights[ : bucket[1]], softmax_loss_function=None)
        
        loss_class = tf.zeros_like(loss)
        expected_seq = [[] for j in range(self.num_topic)]

        with tf.variable_scope("Infer"):
            for j in range(self.num_topic):
                if j > 0: tf.get_variable_scope().reuse_variables()
                theTopic = np.zeros((self.batch_size,self.num_topic))   
                for iter in range(theTopic.shape[0]):
                    theTopic[iter,j] =  1.0
                #print(theTopic)
                state_c2 = tf.concat([encoder_state[0],theTopic,batchYun], 1)
                state_h2 = tf.concat([encoder_state[1],theTopic,batchYun], 1)
                state2 = tf.nn.rnn_cell.LSTMStateTuple(state_c2, state_h2)

                for i in range(len(decoder_inputs)):
                    if i>0:
                        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                            output2, state2, align2 = self.__build_decoder_state_output_computer(decoder_cell, attention_states, tf.matmul(output2,self.decoder_embedding), state2, encoder_mask_unpack, batchYun)
                    else:
                        output2, state2, align2 = self.__build_decoder_state_output_computer(decoder_cell, attention_states, decoder_inputs[0], state2, encoder_mask_unpack, batchYun)
                    expected_seq[j].append(output2)
                # optional loss term
                # loss += 1.0/self.num_topic * sequence_loss(expected_seq[j], targets[: bucket[1]], weights[ : bucket[1]], softmax_loss_function=None)
                loss_class += 1.0/self.num_topic * self.infer_loss(decoder_inputs,weights,expected_seq[j],j)
                
        loss = self.balance * loss + (1-self.balance) * loss_class
        return decoder_outputs, loss, loss_class, attn_weights
    
    def step(self, session, encoder_inputs, decoder_inputs, target_weights, encoder_mask, vocab, ivocab, forward_only, balance):
        encoder_size = self.buckets[0][0]
        decoder_size = self.buckets[0][1]

        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))

        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
        
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
          input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
          input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
          input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        input_feed[self.encoder_mask.name] = encoder_mask
        input_feed[self.balance] = balance
        if balance < 1.0:
            self.learning_rate = self.learning_rate_init / 10.0 # fine tune
        
        # Yun
        input_feed[self.batchYun] = self.yunjiao.getBatchYun(decoder_inputs,ivocab,vocab['PAD'])
        
        if forward_only:
            keep_prob = 1.0
        else:
            keep_prob = 0.8

        input_feed[self.keep_prob] = keep_prob

        if not forward_only:
            output_feed = [self.update,  # Update Op that does Ada.
                     self.gradient_norm,  # Gradient norm.
                     self.loss, self.loss_class, self.debug ]  # Loss for this batch.

            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[l])
        else:
            output_feed = [self.loss, self.loss_class]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[l])

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5:]  # Gradient norm, loss, loss_class, debug, outputs
        else:
            return outputs[0], outputs[1], outputs[2:] # loss, loss_class, outputs.

