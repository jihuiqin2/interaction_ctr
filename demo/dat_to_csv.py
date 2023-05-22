import pandas as pd
import numpy as np
from sklearn import preprocessing

"""
用户文件有以下字段，分别是用户ID、性别、年龄、职业和邮编（ UserID::Gender::Age::Occupation::Zip-code ）
"""
users = pd.read_table('../data/ml-1m/users.dat', sep='::', header=None, engine='python',
                      encoding='utf-8').to_numpy()
users = np.delete(users, 4, axis=1)  # 删除邮编列

"""
电影评分文件包含用户ID、电影ID、评分和时间戳（ UserID::MovieID::Rating::Timestamp ）
"""
# 处理电影
movies = pd.read_table('../data/ml-1m/movies.dat', sep='::', header=None, engine='python',
                       encoding='ISO-8859-1').to_numpy()
movies = np.delete(movies, 1, axis=1)  # 删除电影名称列
movies[:, 0] -= 1

"""
电影评分文件包含用户ID、电影ID、评分和时间戳（ UserID::MovieID::Rating::Timestamp ）
"""

# 处理用户评分
ratings = pd.read_table('../data/ml-1m/ratings.dat', sep='::', header=None, engine='python',
                        encoding='utf-8').to_numpy()[:, :3]
ratings[:, 0:2] -= 1
arr = []
for i in range(len(ratings)):
    if ratings[i][2] > 3:
        ratings[i][2] = 1
    elif ratings[i][2] < 3:
        ratings[i][2] = 0
    if ratings[i][2] == 3:
        arr.append(i)
ratings = np.delete(ratings, arr, axis=0)

"""
合并
"""
unames = ['userId', 'gender', 'age', 'occupation']
mnames = ['movieId', 'genres']
rnames = ['userId', 'movieId', 'rating']

users = pd.DataFrame(users, columns=unames)
movies = pd.DataFrame(movies, columns=mnames)
ratings = pd.DataFrame(ratings, columns=rnames)

data = pd.merge(movies, ratings, on=['movieId'])
data = pd.merge(users, data, on=['userId'])

print(data.columns)

column = ['userId', 'gender', 'age', 'occupation', 'movieId', 'genres', 'rating']

df4 = pd.DataFrame(data, columns=column)
df4.to_csv('../data/ml-1m/movie1m.csv', index=None)
