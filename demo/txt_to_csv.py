# -*-coding:utf-8 -*-

"""
将txt文件转为csv
"""

import csv
import pandas as pd
import numpy as np

columns = ['Label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1', 'C2', 'C3',
           'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
           'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
# 读要转换的txt文件，文件每行各词间以tab分隔
print("开始转换.............")
i = 0
# with open('../data/criteo.txt', 'rb') as filein:
#     all_list = []
#     for line in filein:
#         line_list = line.decode().strip().split("\t")
#         all_list.append(line_list)
#         i = i + 1
#         print("txt转换为csv.............", i)
#         if len(all_list) == 6100000:
#             break
#     print("存储到list中............")
#
# test = pd.DataFrame(columns=columns, data=all_list)
# test.to_csv('../data/criteo/criteo.csv', index=False)
# print("完成了.........")


train_data = pd.read_table('../data/criteo.txt', iterator=True, header=None)
chunks = []
loop = True
while loop:
    try:
        chunkSize = 10000
        chunk = train_data.get_chunk(chunkSize)
        chunks.append(chunk)
        chunk.columns = ['Label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
                         'C1', 'C2', 'C3',
                         'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                         'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24',
                         'C25', 'C26']
        print("txt to csv......", len(chunks))
        if len(chunks) == 4501:
            loop = False
            del chunks
        chunk.to_csv('../data/criteo/criteo.csv', mode='a', header=False, index=None)
    except Exception as e:
        break
