# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import sys
import pickle
from generate_base import Generator
from model import PoemModel
from model import HParams
from state import FLAGS
from PoetryTool import PoetryTool
import numpy as np

global g1
g1 = tf.Graph()

class GeneratorUI(object):

    def __init__(self):
        self.FLAGS = FLAGS
        print(self.FLAGS.data_dir)
        self.vocab, self.ivocab = self.load_dic(
            self.FLAGS.data_dir)
        self.dic_size = len(self.vocab)
        self.hps = HParams(
            vocab_size=len(self.vocab),
            emb_size=self.FLAGS.emb_size,
            hidden_size=self.FLAGS.hidden_size,
            device=self.FLAGS.device,
            learning_rate=self.FLAGS.learning_rate,
            max_gradient_norm=self.FLAGS.max_gradient_norm,
            buckets=[(8, 9)],
            batch_size=self.FLAGS.batch_size,
            num_topic = self.FLAGS.num_topic,
            mode='decode'
        )

        self.tool = PoetryTool()
        self.load_already=False

    def load_dic(self, file_dir):
        """
        loading  training data, including vocab, inverting vocab and corpus
        """
        vocab_file = open(file_dir + '/vocab.pkl', 'rb')
        dic = pickle.load(vocab_file,encoding='utf8')
        vocab_file.close()

        ivocab_file = open(file_dir + '/ivocab.pkl', 'rb')
        idic = pickle.load(ivocab_file,encoding='utf8')
        ivocab_file.close()

        return dic, idic

    def load_model(self, session, beam_size):
        """load parameters in session."""
        decode_hps = self.hps._replace(batch_size=beam_size)
        model = PoemModel(decode_hps)

        ckpt = tf.train.get_checkpoint_state("model/")

        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                  ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise ValueError("%s not found! " % ckpt.model_checkpoint_path)

        return model

    def generate_one(self, all_topic):
        beam_size = input("please input beam size>")
        beam_size = int(beam_size)

        self.sess = tf.InteractiveSession(graph=tf.Graph())
        self.model = self.load_model(self.sess, beam_size)
        self.generator = Generator(
            self.vocab, self.ivocab, self.hps, self.model, self.sess)

        while True:
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            ans, info = self.generator.generate_one(sentence, beam_size=beam_size, all_topic=all_topic, manu=False)
            if len(ans) == 0:
                print("generation failed!")
                print(info)
                continue

            for sen in ans:
                print(sen)
    
    def generate_whole_file(self, infile, outfile, all_topic, beam_size):
        self.sess = tf.InteractiveSession(graph=g1)
        self.model = self.load_model(self.sess, beam_size)
        self.generator = Generator(
            self.vocab, self.ivocab, self.hps, self.model, self.sess)

        fin = open(infile, 'r')
        lines = fin.readlines()
        fin.close()

        for manu in range(10):
            fout = open(outfile+str(manu)+".txt", 'w')
            for line in lines:
                line = line.strip()
                if len(line)<5:
                    continue
                if line == "failed!":
                    continue
                #try:
                ans, info = self.generator.generate_one(line, manu, beam_size, all_topic, False)
                if len(ans) == 0:
                    fout.write(info + "\n")
                else:
                    fout.write(" ".join(ans) + "\n")
                fout.flush()
                    #except:
                    #    print(line)

            fout.close()


def main(_):   
    ui = GeneratorUI()
    #ui.generate_whole_file("poemtestin.txt", "poemout", True, 20)
    ui.generate_one(False)
    
if __name__ == "__main__":
    tf.app.run()

