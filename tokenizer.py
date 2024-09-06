import pandas as pd
from transformers import BertTokenizer
import numpy as np
import tqdm, pickle, ast



class Mytokenizer():
    def __init__(self, word_dic):
        self.word_dic = word_dic
        self.tokens_to_int = []
        self.att_mask = []

    def create_tokens_to_int(self, tokens):
        for token in tokens:
            if token in self.word_dic:
                self.tokens_to_int.append(self.word_dic[token])
            else:
                self.tokens_to_int.append(self.word_dic['none'])

    def create_att_mask(self):
        for i in range(len(self.tokens_to_int)):
            self.att_mask.append(int(1))

    def padding(self, max_length):
        if len(self.tokens_to_int) < max_length:
            while len(self.tokens_to_int) < max_length:
                self.tokens_to_int.append(self.word_dic['padding'])
                self.att_mask.append(self.word_dic['padding'])
        else:
            self.tokens_to_int = self.tokens_to_int[:max_length]
            self.att_mask = self.att_mask[:max_length]

    def get_tti(self):
        return self.tokens_to_int

    def get_att_mask(self):
        return self.att_mask

    def reset(self):
        self.tokens_to_int = []
        self.att_mask = []



df = pd.read_csv("/home/iis/jemin/multimodal/data/dataframe.csv")
df = df.sample(frac=1).reset_index(drop=True)
with open('/home/iis/jemin/multimodal/chardic.pickle', 'rb') as fr:
    chardic = pickle.load(fr)


urls_tokenizer = Mytokenizer(chardic)
content_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


input_ids = []
attention = []
input_labels = []
url_ids = []
url_att = []

for idx, data in tqdm.tqdm(df.iterrows()):
    sentence = ""
    for token in data['contents']:
        sentence = sentence + token
    outputs = content_tokenizer(data['contents'], padding="max_length", truncation=True)
    input_ids.append(outputs['input_ids'])
    attention.append(outputs['attention_mask'])
    input_labels.append(data['label'])

    tmp_urls = ['[CLS]']
    ret_urls = []
    array_data = ast.literal_eval(data['urls'])
    for url in array_data:
        for s in url:
            tmp_urls.append(s)
        tmp_urls.append('[SEP]')
    urls_tokenizer.create_tokens_to_int(tmp_urls)
    urls_tokenizer.create_att_mask()
    urls_tokenizer.padding(512)
    url_ids.append(urls_tokenizer.get_tti())
    url_att.append(urls_tokenizer.get_att_mask())
    urls_tokenizer.reset()



input_ids = np.array(input_ids)
attention = np.array(attention)
input_labels = np.array(input_labels)
url_ids = np.array(url_ids)
url_att = np.array(url_att)


np.save("/home/iis/jemin/multimodal/data/contents/input_ids", input_ids)
np.save("/home/iis/jemin/multimodal/data/contents/attentions", attention)
np.save("/home/iis/jemin/multimodal/data/contents/input_labels", input_labels)
np.save("/home/iis/jemin/multimodal/data/contents/url_ids", url_ids)
np.save("/home/iis/jemin/multimodal/data/contents/url_att", url_att)