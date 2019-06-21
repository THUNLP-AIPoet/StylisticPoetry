import numpy as np

'''
The tool class for poetry project
'''
class PoetryTool(object):
    def __init__(self):
        pass

    '''
    split line to character and save as list
    '''
    def lineSplit2list(self, line):
        sentence = []
        for i in range(len(line)):
            #c = line[i:i + 3]
            sentence.append(line[i])
        return sentence

    """Compute softmax values for each sets of scores in x."""
    ''' Each line of x is a data line '''
    def softmax(self, x):
        ans = np.zeros(np.shape(x), dtype=np.float32)
        for i in range(np.shape(x)[0]):
            c = x[i, :]
            ans[i, :] = np.exp(c) / np.sum(np.exp(c), axis=0)

        return ans

    ''' normalize each line of matrix '''
    def norm_matrxi(self, matrix):
        l = np.shape(matrix)[0]
        for i in range(0, l):
            if np.sum(matrix[i, :]) == 0:
                matrix[i, :] = 0
            else:
                matrix[i, :] = matrix[i, :] / np.sum(matrix[i, :])
        return matrix

    '''
    change id to a sentence
    ws: if use white space to split chracters
    '''
    def beam_get_sentence(self, idxes, ivocab, EOS_ID, ws=False):
        if idxes is not list:
            idxes = list(idxes)
        if EOS_ID in idxes:
          idxes = idxes[:idxes.index(EOS_ID)]

        sentence = self.idxes2senlist(idxes, ivocab)
        if ws:
            sentence = " ".join(sentence)
        else:
            sentence = "".join(sentence)
        return sentence

    ''' idxes to character list '''
    def idxes2senlist(self, idxes, idic):
        sentence = []
        for idx in idxes:
            if idx in idic:
                sentence.append(idic[idx])
            else:
                sentence.append('UNK')
        return sentence

    ''' character list to idx list '''
    def senvec2idxes(self, sentence, dic):
        idxes = []
        for c in sentence:
            if c in dic:
                idxes.append(dic[c])
            else:
                idxes.append(dic['UNK'])
        return idxes

    ''' generate batch input for encoder '''
    def gen_batch_beam(self, sentence, encoder_len, PAD_ID, GO_ID, EOS_ID, UNK_ID, ivocab, all_sen, all_topic, batch_size = 10):
        inputs = [sentence] * batch_size
        all_inputs = [all_sen] * batch_size
        encoder_inputs = []
        encoder_size = encoder_len
        encoder_mask = []
        
        encoder_lda = []

        for i in range(len(inputs)):
            # in genearting, there is no EOS in the input sentence
            encoder_input = inputs[i]  
            #lda format
            if not all_topic:
                all_inputs = inputs
            dict_tmp={}
            single_list=[]
            for j in range(len(all_inputs[i])):
                if all_inputs[i][j] == PAD_ID or all_inputs[i][j] == GO_ID or all_inputs[i][j] == EOS_ID or all_inputs[i][j] ==UNK_ID:
                    continue
                if all_inputs[i][j] in dict_tmp:
                    dict_tmp[all_inputs[i][j]] = dict_tmp[all_inputs[i][j]] + 1.0
                else:
                    dict_tmp[all_inputs[i][j]] = 1.0
            for word,word_cnt in dict_tmp.items():
                single_list.append((word,word_cnt))
            encoder_lda.append(single_list)
            # Encoder inputs are padded
            encoder_pad_size = encoder_size - len(encoder_input)
            encoder_pad = [PAD_ID] * encoder_pad_size
            encoder_inputs.append(encoder_input + encoder_pad)
            mask = [1.0] * (len(encoder_input)) + [0.0] * (encoder_pad_size)
            #print(mask)
            #print(encoder_size)
            #print(self.beam_get_sentence(inputs[i],ivocab,EOS_ID))
            mask = np.reshape(mask, [encoder_size,1])
            encoder_mask.append(mask)

        # create batch-major vectors from the data
        batch_encoder_inputs= []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(batch_size)], dtype=np.int32))
          
        return batch_encoder_inputs, encoder_mask, encoder_lda

