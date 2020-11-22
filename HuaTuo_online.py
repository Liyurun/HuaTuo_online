from transformers import  BertModel, BertConfig,BertTokenizer
from torch import nn,load,tensor
from os import getpid 
import numpy
import gc
import sys
import warnings 
warnings.filterwarnings("ignore")

import psutil
import re

model_name = 'bert-base-chinese'
count = 1
process = psutil.Process(getpid())
def ouput_memory():
    print(count,'Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')

class TextNet(nn.Module):
    def __init__(self,  code_length): #code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('bert-base-chinese', cache_dir="./")
        self.textExtractor = BertModel.from_pretrained(
            'bert-base-chinese', cache_dir="./")
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output=self.textExtractor(tokens, token_type_ids=segments,
                                 		attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]  
        #output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features=self.tanh(features)
        return output

def easy_test(texts):

    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text) #用tokenizer对句子分词
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)#索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens]) #最大的句子长度

    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding
    #segments列表全0，因为只有一个句子1，没有句子2
    #input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
    #相当于告诉BertModel不要利用后面0的部分
        
    #转换成PyTorch tensors
    tokens_tensor = tensor(tokens)
    segments_tensors = tensor(segments)
    input_masks_tensors = tensor(input_masks)


    #——————提取文本特征——————
    text_hashCodes = textNet(tokens_tensor , segments_tensors , input_masks_tensors ) #text_hashCodes是一个32-dim文本特征
    text_hashCodes = text_hashCodes[0][:,0,:]
    #c = sum((text_hashCodes[0] - text_hashCodes[1])*(text_hashCodes[0] - text_hashCodes[1]))/text_hashCodes.shape[1]

    return  text_hashCodes

if __name__ == "__main__":
    print('start HuaTuo_online')
    print('loading pretrain model...')
    textNet = TextNet(code_length=32)

    #——————输入处理——————
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print('loading completed')
    hope = input('希望的药物主治功效：')
    print('calculating...')
    hope_value = easy_test(['[CLS] '+ hope.replace('\n','') + ' [SEP]'])
 

    gc.collect()
    print('researching... ')
    record = load(open('record1.pkl','rb'))

    compare_dic = {}

    
    num = hope_value.shape[1]

    loss_fn = nn.MSELoss(reduce=False, size_average=True)
    for i,(name,value) in enumerate(record.items()):
        # record[name] = sum(t**2)/num
        record[name] = sum(loss_fn(value[0],hope_value[0]))/1000
        print('\r' + "正在搜索:" + ' %.2f%%\t' % ((i + 1) / len(record) * 100), end='')
        print('[' + '>' * int(i / len(record) * 30) + '-' * int(30 - i / len(record) * 30) + ']', end='')
        
    print(' ')
    hh = sorted(record.items(), key = lambda x:(x[1]))

    for i in range(5):
        string = '推荐药物排名{0}——"{1}",推荐程度{2:.3f}'.format(i+1,hh[i][0].replace('\n',''),float(hh[i][1].detach().numpy()))
        print(re.sub('说明书','',string))


# S = 0
# for i in dir():
#     S+=sys.getsizeof(i)
#     print(sys.getsizeof(i))
# print(S)