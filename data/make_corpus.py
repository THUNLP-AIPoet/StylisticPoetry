from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import cPickle
import codecs

import numpy as np
from six.moves import xrange

reload(sys)
sys.setdefaultencoding('utf-8')
    
def loadData(fname,splitSent):
    f = codecs.open(fname,'r','utf-8')
    
    line = 0
    onePoem={}
    onePoemList=[]
    oneSen=[]
    originPoem=[]
    for s in f.readlines():
        line = line + 1      
        #print(line)
        s=s.strip()
        if(len(s)>7):
            print('illegal format')
            print(s)
            break
        for i in xrange(len(s)):
            #print(s)
           
            if(s[i]!=" " and s[i]!="\n" and vocab.has_key(s[i])):
                #print(ivocab[vocab[s[i]]])
                #print(s[i])
                oneSen.append(vocab[s[i]])
                if onePoem.has_key(vocab[s[i]]):
                    onePoem[vocab[s[i]]] = onePoem[vocab[s[i]]]+1.0
                else:
                    onePoem[vocab[s[i]]] = 1.0
        originPoem.append(oneSen)
        #print(oneSen)
        oneSen=[]
        
        if line % splitSent == 0:
            for word,count in onePoem.iteritems():
                onePoemList.append((word,count))   

            for j in xrange(splitSent-1):
                corpus.append((originPoem[j],originPoem[j+1],onePoemList))

            oneSen=[]
            originPoem=[]
            onePoem={}
            onePoemList=[]
        
    f.close()

def loadData_oneTopic(fname,splitSent):
    f = codecs.open(fname,'r','utf-8')
    
    line = 0
    onePoem={}
    onePoemList=[]
    TopicPoemList=[]
    oneSen=[]
    originPoem=[]
    for s in f.readlines():
        line = line + 1      
        #print(line)
        s=s.strip()
        if(len(s)>7):
            print('wtf')
            print(s)
            break
        for i in xrange(len(s)):
            #print(s)
           
            if(s[i]!=" " and s[i]!="\n" and vocab.has_key(s[i])):
                #print(ivocab[vocab[s[i]]])
                #print(s[i])
                oneSen.append(vocab[s[i]])
                if onePoem.has_key(vocab[s[i]]):
                    onePoem[vocab[s[i]]] = onePoem[vocab[s[i]]]+1.0
                else:
                    onePoem[vocab[s[i]]] = 1.0
        onePoemList.append(onePoem)
        onePoem={}
        originPoem.append(oneSen)
        oneSen=[]
        
        if line % splitSent == 0:
            for j in xrange(splitSent-1):
                for word,count in onePoemList[j+1].iteritems():
                    TopicPoemList.append((word,count))   
                corpus.append((originPoem[j],originPoem[j+1],TopicPoemList))
                TopicPoemList=[]

            oneSen=[]
            originPoem=[]
            onePoem={}
            onePoemList=[]
            TopicPoemList=[]
        
    f.close()    
    
def loadDictionary():
    #vocab_file = open('vocab.pkl', 'rb')
    #dic = cPickle.load(vocab_file)
    #vocab_file.close()
    dic={}

    ivocab_file = open('ivocab.pkl', 'rb')
    idic = cPickle.load(ivocab_file)
    ivocab_file.close()
    
    for id,word in idic.items():
        idic[id] = word.decode('utf-8')
        dic[word.decode('utf-8')] = id
    return dic,idic

def loadCorpus():
    corpus_file = open('corpus.pkl', 'rb')
    cps = cPickle.load(corpus_file)
    corpus_file.close()
    return cps

if __name__ == "__main__":    
    # preprocess corpus
    vocab,ivocab=loadDictionary()
       
    corpus=[]
    loadData_oneTopic('poems.txt',4)

    for i in xrange(len(corpus[0][0])):
        print(ivocab[corpus[0][0][i]])
    for i in xrange(len(corpus[0][1])):
        print(ivocab[corpus[0][1][i]])
    print(corpus[0][2])
    # 8:1:1 train dev test
    np.random.seed(1234)
    np.random.shuffle(corpus)
    print(int(math.floor(0.8*len(corpus))))
    corpus_train = corpus[0:int(math.floor(0.8*len(corpus)))]
    corpus_dev = corpus[int(math.floor(0.8*len(corpus))):int(math.floor(0.9*len(corpus)))]
    print(len(corpus_dev))
    corpus_test = corpus[int(math.floor(0.9*len(corpus))):]
    cPickle.dump(corpus_train,open('text_train.pkl','wb'))
    cPickle.dump(corpus_dev,open('text_dev.pkl','wb'))
    cPickle.dump(corpus_test,open('text_test.pkl','wb'))
    