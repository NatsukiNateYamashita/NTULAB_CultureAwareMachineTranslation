import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import urllib.parse
import csv
from webdriver_manager.chrome import ChromeDriverManager
import zhconv
import sys
import os

def convertToZh(inputTsPath, outTsPath):
    with open(inputTsPath,'r', encoding='UTF-8') as f:
        content = f.read()
        with open(outTsPath,'w',encoding='UTF-8') as f1:
            f1.write(zhconv.convert(content, 'zh-cn'))
def convertToTw(inputTsPath, outTsPath):
    with open(inputTsPath,'r', encoding='UTF-8') as f:
        content = f.read()
        with open(outTsPath,'w',encoding='UTF-8') as f1:
            f1.write(zhconv.convert(content, 'zh-tw'))

# convertToZh('annotated_mpdd.csv', 'simplified_annotated_mpdd.csv')

class JAZHTranslator:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        # self.browser = webdriver.Chrome(options=self.options)
        self.browser = webdriver.Chrome(ChromeDriverManager().install())
        self.browser.implicitly_wait(3)

    def translate(self, text, dest='ja'):
        # 翻訳したい文をURLに埋め込んでからアクセスする
        
        # text_for_url = urllib.parse.quote_plus(text, safe='')
        url = "https://www.deepl.com/en/translator#ja/zh/{0}".format(
            text)
        self.browser.get(url)
        wait_time = 2 
        time.sleep(wait_time)
        self.browser.refresh()
        wait_time = 2 + len(text) / 5
        time.sleep(wait_time)
        # 翻訳結果を抽出する
        while 1:
       
            Output_selector = "#target-dummydiv"
            Outputtext = self.browser.find_element_by_css_selector(Output_selector).get_attribute("textContent")
            if Outputtext != "" :
                # Outputtext = "ERROR"
                break
            time.sleep(1)

        return Outputtext

    def quit(self):
        self.browser.quit()


class ZHJATranslator:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        # self.browser = webdriver.Chrome(options=self.options)
        self.browser = webdriver.Chrome(ChromeDriverManager().install())
        self.browser.implicitly_wait(3)

    def translate(self, text, dest='ja'):
        # 翻訳したい文をURLに埋め込んでからアクセスする
        url = "https://www.deepl.com/en/translator#zh/ja/{0}".format(
            text)
        self.browser.get(url)
        wait_time = 2 
        time.sleep(wait_time)
        self.browser.refresh()
        wait_time = 2 + len(text) / 5
        time.sleep(wait_time)

        while 1:
       
            Output_selector = "#target-dummydiv"
            Outputtext = self.browser.find_element_by_css_selector(Output_selector).get_attribute("textContent")
            if Outputtext != "" :
                # Outputtext = "ERROR"
                break
            time.sleep(1)

        return Outputtext

    def quit(self):
        self.browser.quit()

def read_csv(path):
    data = []
    f_path = path
    with open(f_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # if i == 0:
            #     pass
            # else:
            data.append(row)
    print("finish read csv")
    return data

def save_csv(path,data):
    with open(path, 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)


import re
code_regex = re.compile('["#$%&\'\\\\()*+,-./:;<=>@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠｀＋￥％]')



# translator = ZHJATranslator()
# # convertToZh('integrated_mpdd.csv', 'simplified_integrated_mpdd.csv')
# lastsave = 24000
# comp_data = read_csv("integrated_mpdd_translated_deepl.csv")
# uncomp_data = read_csv("integrated_mpdd.csv")
# data = comp_data[:lastsave] + uncomp_data[lastsave:]
# save_list = []
# for i, d in enumerate(data):
#     if i < lastsave:
#         continue
#     d[0] = code_regex.sub('', d[0])
#     d[0] = zhconv.convert(d[0], 'zh-cn')
#     temp = translator.translate(d[0])
#     temp = temp.rstrip()
#     data[i][0] = d[0]
#     data[i][1] = temp
#     # print(i, d[0])
#     print(i, temp)
#     if (i % 100 == 0) or (i == len(data)-1):
#         save_csv("integrated_mpdd_translated_deepl.csv", data)
#         print(i, "saved file!!")
#         print()
# save_csv("integrated_mpdd_translated_deepl.csv", data)
# translator.quit()


# 空白になってしまった部分を再翻訳
# translator = ZHJATranslator()
# lastsave = 0
# comp_data = read_csv("integrated_mpdd_translated_deepl.csv")
# # uncomp_data = read_csv("integrated_cejc.csv")
# # data = comp_data[:lastsave] + uncomp_data[lastsave:]
# data = comp_data
# print(len(data))
# for i, d in enumerate(data):
#     # if i < lastsave:
#     #     continue
#     if (d[1]=='') or (d[1] in ['。','。']):
#         d[0] = code_regex.sub('', d[0])
#         temp = translator.translate(d[0])
#         temp = temp.rstrip()
#         data[i][0] = d[0]
#         data[i][1] = temp
#         # print(i, d[0])
#         print(i, temp)
#         save_csv("integrated_mpdd_translated_deepl.csv", data)
#         print(i, "saved file!!")
#         print()
# # save_csv("integrated_mpdd_translated_deepl.csv", data)
# # convertToTw('simplified_annotated_cejc_translated_deepl.csv', 'annotated_cejc_translated_deepl.csv')
# translator.quit()

translator = JAZHTranslator()
lastsave = 86100


comp_data = read_csv("integrated_cejc_translated_deepl.csv")
uncomp_data = read_csv("integrated_cejc.csv")
data = comp_data[:lastsave] + uncomp_data[lastsave:]
print(len(data))
for i, d in enumerate(data):
    if i < lastsave:
        continue
    d[0] = code_regex.sub('', d[0])
    temp = translator.translate(d[0])
    temp = temp.rstrip()
    data[i][0] = d[0]
    data[i][1] = temp
    # print(i, d[0])
    print(i, temp)
    if (i % 100 == 0) or (i == len(data)-1):
        save_csv("integrated_cejc_translated_deepl.csv", data)
        print(i, "saved file!!")
        print()
save_csv("integrated_cejc_translated_deepl.csv", data)
# convertToTw('simplified_annotated_cejc_translated_deepl.csv', 'annotated_cejc_translated_deepl.csv')
translator.quit()


# 空白になってしまった部分を再翻訳
translator = JAZHTranslator()
lastsave = 0
comp_data = read_csv("integrated_cejc_translated_deepl.csv")
# uncomp_data = read_csv("integrated_cejc.csv")
# data = comp_data[:lastsave] + uncomp_data[lastsave:]
data = comp_data
print(len(data))
for i, d in enumerate(data):
    # if i < lastsave:
    #     continue
    if (d[1]=='') or (d[1] in ['。','。']):
        d[0] = code_regex.sub('', d[0])
        temp = translator.translate(d[0])
        temp = temp.rstrip()
        data[i][0] = d[0]
        data[i][1] = temp
        # print(i, d[0])
        print(i, temp)
        save_csv("integrated_cejc_translated_deepl.csv", data)
        print(i, "saved file!!")
        print()
# save_csv("integrated_cejc_translated_deepl.csv", data)
# convertToTw('simplified_annotated_cejc_translated_deepl.csv', 'annotated_cejc_translated_deepl.csv')
translator.quit()