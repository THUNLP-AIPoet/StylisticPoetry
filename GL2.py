#coding=utf-8
from __future__ import division
'''
Created on 2015年10月31日

@author: sony
'''

class GLJudge2(object):
    '''
    获取一首诗的格律
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        # 读入平声字文件
        print("loading pingsheng.txt")
        f = open("data/other/pingsheng.txt",'r')
        self.__ping = f.read()
        print(type(self.__ping))
        f.close()
        #self.__ping = self.__ping.decode("utf-8")
        #print self.__ping
        f = open("data/other/zesheng.txt",'r')
        self.__ze = f.read()
        f.close()
        print(type(self.__ze))
        #self.__ze = self.__ze.decode("utf-8")
        
        
        
    def gelvJudge(self, sentence):
        # print "#" + sentence + "#"
        if(len(sentence) == 5):
			#1
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #2
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #3
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #4
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 0
            #5
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 0
            #6
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping):
                return 1
            #7
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping):
                return 1
            #8
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #9
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #10
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #11
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #12
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2
            #13
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2
            #14
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2


        elif (len(sentence) == 7):
            #1
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #2
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #3
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #4
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 0
            #5
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 0
            #6
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ping):
                return 1
            #7
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ping):
                return 1
            #8
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #9
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #10
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ping and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #11
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #12
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
            #13
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
            #14
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
        else:
            return -2
        return -1
        
