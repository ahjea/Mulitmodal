
from parse import getMetacontent, getContent, isEnglish
import os, sys, json, tqdm
from bs4 import BeautifulSoup
import pandas as pd


def read_file(path):
    ret = ""
    try:
        f = open(path, "rt")
        while True:
            c = f.read()
            if c == '':
                break
            ret = ret + c
        f.close()
    except:
        pass
    return ret



benign_path = "/home/iis/jemin/extra_dataset/pedia/benign/"
phish_path = "/home/iis/jemin/extra_dataset/pedia/phish_sample/"

benign_list = os.listdir(benign_path)
phishing_list = os.listdir(phish_path)
count = 0

# benign : 1 / phishing : 0
df = pd.DataFrame(columns=['url', 'urls', 'contents', 'label'])

for i in tqdm.tqdm(benign_list):
    try:
        html_path = benign_path + i + "/html.txt"
        content = getContent(html_path, True)
        if isEnglish(content):
            html_txt = read_file(html_path)
            soup = BeautifulSoup(html_txt, 'html.parser')
            links = soup.find_all('a')
            link_list = []
            for tag in links:
                link = tag.get('href', None)
                if link is not None:
                    link_list.append(link)
            df_len = len(df)
            df.loc[df_len] = [i, link_list, content, int(1)]
        else:
            pass
    except:
        pass

for i in tqdm.tqdm(phishing_list):
    try:
        html_path = phish_path + i + "/html.txt"
        content = getContent(html_path, True)
        if isEnglish(content):
            html_txt = read_file(html_path)
            soup = BeautifulSoup(html_txt, 'html.parser')
            links = soup.find_all('a')
            link_list = []
            for tag in links:
                link = tag.get('href', None)
                if link is not None:
                    link_list.append(link)
            df_len = len(df)
            df.loc[df_len] = [i, link_list, content, int(0)]
        else:
            pass
    except:
        pass

ret = df.sample(frac=1).reset_index(drop=True)
print(len(df))
df.to_csv("/home/iis/jemin/multimodal/dataframe.csv")