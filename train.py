# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from model import PoemModel
from model import HParams 

from state import FLAGS


class PoemTrainer(object):
    """construction for PoemTrainer"""
    def __init__(self):
        self.FLAGS = FLAGS

        self.batch_size = self.FLAGS.batch_size

        # training data
        self.vocab, self.ivocab, self.data = self.load_data(self.FLAGS.data_dir)
        self.dic_size =  len(self.vocab)

        self.PAD_ID = self.vocab['PAD']
        self.GO_ID = self.vocab['GO']
        self.EOS_ID = self.vocab['</S>']
        self.UNK_ID = self.vocab['UNK']

        # development data
        pkl_file = open(self.FLAGS.data_dir + '/text_dev.pkl', 'rb')
        self.dev_data  = pickle.load(pkl_file)
        pkl_file.close()

        print(np.shape(self.data))
        #print(np.shape(self.dev_data))

        self.batch_check = np.zeros(len(self.data))
        self.batch_check2 = np.zeros(len(self.data), dtype=np.float32)


        # construct HParams 
        self.hps = HParams(
          vocab_size = len(self.vocab),
          emb_size = self.FLAGS.emb_size, 
          hidden_size = self.FLAGS.hidden_size,
          device = self.FLAGS.device,
          learning_rate = self.FLAGS.learning_rate,
          max_gradient_norm = self.FLAGS.max_gradient_norm,
          buckets = [ (8, 9)], 
          batch_size = self.FLAGS.batch_size,
          num_topic = self.FLAGS.num_topic,
          mode = 'train'
        )

        print ("Params  sets: ")
        print ("___________________")
        print ("learning_rate:%s  max_gradient_norm:%s   " % (str(self.FLAGS.learning_rate), self.FLAGS.max_gradient_norm))
        print ("batch_size:%d" % (self.FLAGS.batch_size))
        print ("hidden_size:%d   emb_size:%d   " % (self.FLAGS.hidden_size, self.FLAGS.emb_size))
        print ("steps_per_checkpoint:%d" % (self.FLAGS.steps_per_checkpoint))
        print ("steps_per_sample:%d" % (self.FLAGS.steps_per_sample))
        print ("sample_num:%d" % (self.FLAGS.sample_num))
        print ("device:%s" % (self.FLAGS.device))
        print ("Vocabulary size: %d  data size: %d "% (len(self.vocab), len(self.data)))
        print("___________________")

        self.buckets =  self.buckets = [(8, 9)]


    def load_data(self, file_dir):
        """
        loading  training data, including vocab, inverting vocab and corpus
        """
        vocab_file = open(file_dir + '/vocab.pkl', 'rb') #dictionary word->id
        dic = pickle.load(vocab_file,encoding='utf8')
        vocab_file.close()

        ivocab_file = open(file_dir + '/ivocab.pkl', 'rb') #dictionary id->word
        idic = pickle.load(ivocab_file,encoding='utf8')
        ivocab_file.close()

        corpus_file = open(file_dir + '/text_train.pkl', 'rb')
        corpus = pickle.load(corpus_file,encoding='utf8')
        #print(corpus[0])
        corpus_file.close()
        
        return dic, idic, corpus



    def get_next_batch_sentence(self, inputs, outputs, batch_size, bucket_id=0):
        assert len(inputs) == len(outputs) == self.batch_size
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        encoder_mask = []
        encoder_lda = []

        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for i in range(batch_size):

            encoder_input = inputs[i]
            decoder_input = outputs[i] + [self.EOS_ID]
            #print(len(encoder_input))

            # get lda format corpus
            dict_tmp={}
            single_list=[]
            for j in range(len(encoder_input)):
                if encoder_input[j] == self.PAD_ID or encoder_input[j] == self.GO_ID or encoder_input[j] == self.EOS_ID or encoder_input[j] ==self.UNK_ID:
                    continue
                if encoder_input[j] in dict_tmp:
                    dict_tmp[encoder_input[j]] = dict_tmp[encoder_input[j]] + 1.0
                else:
                    dict_tmp[encoder_input[j]] = 1.0
            for word,word_cnt in dict_tmp.items():
                single_list.append((word,word_cnt))
            encoder_lda.append(single_list)
            # Encoder inputs are padded and then reversed.
            
            encoder_pad_size = encoder_size - len(encoder_input)
            encoder_pad = [self.PAD_ID] * encoder_pad_size
            encoder_inputs.append(encoder_input + encoder_pad)
            mask = [1.0] * (len(encoder_input)) + [0.0] * (encoder_pad_size)
            mask = np.reshape(mask, [encoder_size,1])
            encoder_mask.append(mask)

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([self.GO_ID] + decoder_input +
                            [self.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == self.PAD_ID:
                    batch_weight[batch_idx] = 0.0

            batch_weights.append(batch_weight)

        #
        encoder_mask = np.array(encoder_mask)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, encoder_mask, encoder_lda

    def get_next_batch(self, step, data, batch_size, train_flag = True):
        """
        get next batch, we use global_step to determin the batch id
        """
        batch_poems = []
        data_size = len(data)
        idx = (step * batch_size) % data_size

        encoder_inputs = []
        decoder_inputs = []
        encoder_lda = []

        for i in range(0, batch_size):
            if train_flag:
                self.batch_check[(idx + i) % data_size] = 1
                self.batch_check2[(idx + i) % data_size] += 1.0
            encoder_inputs.append(  data[ (idx + i) % data_size][0]  )
            #print(len(data[ (idx + i) % data_size][0]))
            decoder_inputs.append(  data[ (idx + i) % data_size][1]  )
            encoder_lda.append(  data[ (idx + i) % data_size][2]  )

        batch_encoder_inputs, batch_decoder_inputs, batch_weights, encoder_mask, _  = self.get_next_batch_sentence(encoder_inputs, decoder_inputs, self.batch_size)
        # TO DO: return lda format list of tuples

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, encoder_mask, encoder_lda
        
    def create_model(self, session):
        """Create the model and initialize or load parameters in session."""
        model = PoemModel(self.hps)
        ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        return model

    def idx2sentece(self, idx):
        sentence = []
        for i in idx:
            sentence.append(self.ivocab[i])

        return sentence

    def sentence2idx(self, sentence):
        idx = []
        for c in sentence:
            if c in self.vocab:
                idx.append(self.vocab[c])
            else:
                idx.append(self.vocab['UNK'])

        return idx

    '''
    search a output sentence by the simple greedy_decode
    '''
    def greedy_decode(self, outputs):
        outidx = [int(np.argmax(logit, axis=0)) for logit in outputs]
        #print (outidx)
        if self.EOS_ID in outidx:
            outidx = outidx[:outidx.index(self.EOS_ID)]

        sentence = self.idx2sentece(outidx)
        sentence = " ".join(sentence)
        return sentence

    def sample(self, encoder_inputs, decoder_inputs, outputs):
        sample_num = self.FLAGS.sample_num
        if sample_num > self.batch_size:
            sample_num = self.batch_size

        idxes = []  #Random select some examples
        while (len(idxes) < self.FLAGS.sample_num):
            which = np.random.randint(self.batch_size)
            if not which in idxes:
                idxes.append(which)
    

        for idx in idxes:
            input1 = [ c[idx] for c in encoder_inputs]
            input1 = " ".join(self.idx2sentece(input1))

            target1 = [ c[idx] for c in decoder_inputs ]
            target1 = " ".join(self.idx2sentece(target1))

            outline1 = [ c[idx] for c in outputs]
            line1 = self.greedy_decode(outline1)
            print ("#" + input1  + "#" + "       #"  + target1 + "#      #" + line1 + "#")

    def train(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.98)
        gpu_options.allow_growth = True

        with tf.Session(config = tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options) ) as sess:

            # Create model.
            print ("create model") 
            model = self.create_model(sess)

            # This is the training loop.
            step_time, loss, loss_class = 0.0, 0.0, 0.0
            current_step = 0
            time1 = time.time()
            balance = 1.0

            while True:
                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights, encoder_mask, encoder_lda = self.get_next_batch(model.global_step.eval() , self.data, self.batch_size)
                #print(decoder_inputs.shape)
                
                #print ("training!!!!!")
                if current_step >50000:# and current_step <= 1000000:
                    balance = 0.9
                else:
                    balance = 1.0  
                _, step_loss, step_loss_class,  debug, outputs  = model.step(sess, encoder_inputs, decoder_inputs, target_weights, encoder_mask, self.vocab, self.ivocab, False, balance)
                #print (np.shape(debug))
  
                # do sample and validation
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                loss_class += step_loss_class / FLAGS.steps_per_checkpoint 
                current_step += 1

                #ever steps_per_sample steps to sample and show the generate answers
                if current_step % self.FLAGS.steps_per_sample == 0:
                    print ("running %d iterations" % (model.global_step.eval()))
                    debug = np.array(debug)
                    which = random.randint(0, self.batch_size-1)
                    EOS = int(np.sum(encoder_mask[which, :, 0]))
                    print (EOS)
                    align = debug[0:EOS, 0, which, 0:EOS]
                    print (align)


                    self.sample(encoder_inputs, decoder_inputs, outputs)
                    eval_loss = step_loss
                    time2 = time.time()
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("training perplexity %.2f" % (eval_ppx))
                    print("training loss %.2f" % (eval_loss))
                    print("training loss class %.5f" % (step_loss_class))
                    print ("%f seconds per iteration" % (float(time2-time1) /  self.FLAGS.steps_per_sample ))
                    print ("%d poems in training data have been used!" % (np.sum(self.batch_check)))
                    print ("using avg %f   max %f  min %f  " % (np.mean(self.batch_check2), np.max(self.batch_check2), np.min(self.batch_check2)    )   )
                    sys.stdout.flush()
                    time1 = time.time()

                if current_step % self.FLAGS.steps_per_train_log == 0:
                    fout = open("trainlog.txt", 'a')
                    fout.write(self.hps.mode + " " + str(model.global_step.eval()) + " " + str(eval_loss) + " " + str(eval_ppx) + " \n")
                    fout.close()

                # ever FLAGS.steps_per_checkpoint steps, we save checkpoint, print necessary information.
                if current_step % self.FLAGS.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                    print ("global step %d step-time %.2f perplexity "
                        "%.2f" % (model.global_step.eval(), step_time, perplexity))
                    perplexity_class = math.exp(loss_class) if loss_class < 300 else float('inf')
                    print ("global step %d step-time %.2f perplexity class "
                        "%.2f" % (model.global_step.eval(), step_time, perplexity_class))

                    # Save checkpoint and zero timer and loss.
                    print ("saving model...")
                    checkpoint_path = os.path.join(self.FLAGS.model_dir, "poem.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0

                
                if current_step % self.FLAGS.steps_per_validate == 0:
                    # Run on development set.    
                    tt_ppl = 0.0
                    tt_ppl_class = 0.0
                    dev_bnum = 1000
                    print ("run dev batch")
                    for iterdev in range(dev_bnum):
                        encoder_inputs, decoder_inputs, target_weights, encoder_mask, encoder_lda  = self.get_next_batch(np.random.randint(0,2000), self.dev_data, self.batch_size, False)

                        if current_step > 150000 and current_step <= 200000:
                            balance = 0.0
                        else:
                            balance = 1.0
                        step_loss, step_loss_class, outputs  = model.step(sess, encoder_inputs, decoder_inputs, target_weights, encoder_mask, self.vocab, self.ivocab, True, balance)
               

                        eval_loss = step_loss
                        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                        tt_ppl += eval_ppx

                        eval_ppx_class = math.exp(step_loss_class) if step_loss_class < 300 else float('inf')
                        tt_ppl_class += eval_ppx_class
                    tt_ppl = tt_ppl / dev_bnum
                    tt_ppl_class = tt_ppl_class / dev_bnum
                    print("dev perplexity %.2f" % (tt_ppl))
                    print("dev perplexity %.5f" % (tt_ppl_class))

                    sys.stdout.flush()

                    fout = open("devlog.txt", 'a')
                    fout.write(self.hps.mode + " " + str(current_step) + " " + str(eval_loss) + " " + str(tt_ppl) + " \n")
                    fout.close()
                
def main(_):
    #tf.reset_default_graph()
    trainer = PoemTrainer()
    trainer.train()

if __name__ == "__main__":
  tf.app.run()
