import pandas

# 读取csv
df = pandas.read_csv('train.csv')

# df['source'] = df['source']+1

# 统计每个source的数量
a = df['source'].value_counts()
print(a)
# 保存csv
# df.to_csv('train.csv', index=False)