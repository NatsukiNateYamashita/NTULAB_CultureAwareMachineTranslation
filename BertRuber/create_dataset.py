import pandas as pd
import os
import csv
import random


corpus = "mpdd"
f_path = '/Users/natsuki/Documents/WorkingSpace/afterprocess4adddataset/integrated_{}.csv'.format(corpus)
df = pd.read_csv(f_path, 
                names=['original','translated','speakerid','conversationid',
                'req_query_tag','req_query_unreadable', 'req_query_natural','req_query_rewrited', 'req_res_unreadable',   'req_res_natural',  'req_res_rewrited',
                'apo_query_tag','apo_query_unreadable', 'apo_query_natural','apo_query_rewrited', 'apo_res_unreadable',   'apo_res_natural',  'apo_res_rewrited',
                'tha_query_tag','tha_query_unreadable', 'tha_query_natural','tha_query_rewrited', 'tha_res_unreadable',   'tha_res_natural',  'tha_res_rewrited',
                'utteranceid']
                )

# print(df['original'])
### CREATE train_data
sit_list = ['request','apology','thanksgiving']
for sit in sit_list:
    save_dir = f'data/{corpus}/{sit}'
    os.makedirs(save_dir, exist_ok=True)
    save_dir += '/'
    # query
    tag_col = sit[:3] + '_query_tag'
    query = df[df[tag_col].str.contains('rewrite',na=False)]

    original_query = query['original']
    original_query.to_csv(save_dir +'original_query.csv',header=False, index=False,encoding='utf-8')

    translated_query = query['translated']
    translated_query.to_csv(save_dir +'translated_query.csv',header=False, index=False,encoding='utf-8')

    rewrited_query_col = sit[:3] + '_query_rewrited'
    rewrited_query = query[rewrited_query_col]
    rewrited_query.to_csv(save_dir +'rewrited_query.csv',header=False, index=False,encoding='utf-8')
    # response
    res_idx = query[tag_col].index+1
    original_res = df['original'][res_idx]
    original_res.to_csv(save_dir +'original_res.csv',header=False, index=False,encoding='utf-8')

    translated_res = df['translated'][res_idx]
    translated_res.to_csv(save_dir +'translated_res.csv',header=False, index=False,encoding='utf-8')
    
    rewrited_res_col = sit[:3] + '_res_rewrited'
    rewrited_res = query[rewrited_res_col]
    rewrited_res.to_csv(save_dir +'rewrited_res.csv',header=False, index=False,encoding='utf-8')

    original_neg = df[~df[tag_col].str.contains('rewrite',na=False)]
    original_neg = original_neg['original']
    original_neg.to_csv(save_dir +'original_neg.csv',header=False, index=False,encoding='utf-8')


# def get_data(f_path):
#     data = []
#     with open(f_path, 'r', encoding='utf_8_sig') as f:
#         reader = csv.reader(f)
#         for i, row in enumerate(reader):
#             data.append(row[0])
#     return data

# ### MIX
# data_kind = ['query','res','neg']
# sit_list = ['request','apology','thanksgiving']
# for sit in sit_list:
#     data_dir = '/nfs/nas-7.1/yamashita/LAB/BertRuber/data/'

#     path = data_dir + 'cejc/{}/rewrited_query.csv'.format( sit)
#     q_data = get_data(path)
#     path = data_dir + 'mpdd/{}/original_query.csv'.format( sit)
#     q_data.extend(get_data(path))
#     path = data_dir + 'cejc/{}/rewrited_res.csv'.format( sit)
#     r_data = get_data(path)
#     path = data_dir + 'mpdd/{}/rewrited_res.csv'.format( sit)
#     r_data.extend(get_data(path))

#     if len(q_data) != len(r_data):
#         print("Different length!")
#         break
#     else:
#         p = list(zip(q_data, r_data))
#         random.shuffle(p)
#         q_data, r_data = zip(*p)

#     save_dir = f'data/mix/{sit}'
#     os.makedirs(save_dir, exist_ok=True)
#     save_dir += '/'
#     save_path = save_dir + 'rewrited_query.csv'
#     with open(save_path, 'w', encoding='utf_8_sig')as f:
#         writer = csv.writer(f)
#         writer.writerows(q_data)
#     save_path = save_dir + 'rewrited_res.csv'
#     with open(save_path, 'w', encoding='utf_8_sig')as f:
#         writer = csv.writer(f)
#         writer.writerows(r_data)
    
    # path = data_dir + 'cejc/{}/original_neg.csv'.format( sit)
    # n_data = get_data(path)
    # path = data_dir + 'mpdd/{}/original_neg.csv'.format( sit)
    # n_data.extend(get_data(path))
    # save_path = save_dir + 'original_neg.csv'
    # with open(save_path, 'w', encoding='utf_8_sig')as f:
    #     writer = csv.writer(f)
    #     writer.writerows(n_data)
    

    


