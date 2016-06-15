#-*-coding:mbcs-*-


import operator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
matplotlib.style.use('ggplot')


# ת���ļ���ʽ������csv�ļ�
fp_origin = open("./Analysis.txt", 'r')
fp_modified = open("./Analysis.csv", 'w')

line = fp_origin.readline()
while(line):
    temp = line.strip().split()
    temp = ','.join(temp)+'\n'
    fp_modified.write(temp)
    line = fp_origin.readline()
    
fp_origin.close()
fp_modified.close()


# �����������ݣ�����ͺ���ֵ��
name_category = ["season", "river_size", "river_speed"]
name_value = ["mxPH", "mnO2", "Cl", "NO3", "NH4", "oPO4", "PO4", "Chla", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]
# �洢7�ֺ����Ӧ������
name_seaweed = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]

# ��ȡ����
data_origin = pd.read_csv("./Analysis.csv", 
                   names = name_category+name_value,
                   na_values = "XXXXXXX")

# ���ַ�����ת��Ϊcategory
for item in name_category:
    data_origin[item] = data_origin[item].astype('category')


# ����ժҪ**
# 
# - �Ա�����ԣ�����ÿ������ȡֵ��Ƶ��

for item in name_category:
    print item, '��Ƶ��Ϊ��\n', pd.value_counts(data_origin[item].values), '\n'


# - ����ֵ���ԣ����������С����ֵ����λ�����ķ�λ����ȱʧֵ�ĸ���

# ���ֵ
data_show = pd.DataFrame(data = data_origin[name_value].max(), columns = ['max'])
# ��Сֵ
data_show['min'] = data_origin[name_value].min()
# ��ֵ
data_show['mean'] = data_origin[name_value].mean()
# ��λ��
data_show['median'] = data_origin[name_value].median()
# �ķ�λ��
data_show['quartile'] = data_origin[name_value].describe().loc['25%']
# ȱʧֵ����
data_show['missing'] = data_origin[name_value].describe().loc['count'].apply(lambda x : 200-x)

print data_show


# ���ݿ��ӻ�

# ֱ��ͼ
fig = plt.figure(figsize = (20,11))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 5, i)
    data_origin[item].plot(kind = 'hist', title = item, ax = ax)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/histogram.jpg')
print 'histogram saved at ./image/histogram.jpg'


# qqͼ
fig = plt.figure(figsize = (20,12))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 5, i)
    sm.qqplot(data_origin[item], ax = ax)
    ax.set_title(item)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/qqplot.jpg')
print 'qqplot saved at ./image/qqplot.jpg'

# ��qqͼ�п��Կ�����ֻ��mxPH��mnO2����ֵ������̬�ֲ�������ֵ��������

# - ���ƺ�ͼ������Ⱥֵ����ʶ��
# ��ͼ
fig = plt.figure(figsize = (20,12))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 5, i)
    data_origin[item].plot(kind = 'box')
    i += 1
fig.savefig('./image/boxplot.jpg')
print 'boxplot saved at ./image/boxplot.jpg'


# - ��7�ֺ��壬�ֱ�������������Ʊ�������size��������ͼ
# ������ͼ
fig = plt.figure(figsize = (10, 27))
i = 1
for seaweed in name_seaweed:
    for category in name_category:
        ax = fig.add_subplot(7, 3, i)
        data_origin[[seaweed, category]].boxplot(by = category, ax = ax)
        ax.set_title(seaweed)
        i += 1
plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
fig.savefig('./image/boxplot_condition.jpg')
print 'boxplot_condition saved at ./image/boxplot_condition.jpg'


# ����ȱʧ�Ĵ���**

# �ҳ�����ȱʧֵ��������Ŀ����ֵ
nan_list = pd.isnull(data_origin).any(1).nonzero()[0]

# ��ʾ����ȱʧֵ��ԭʼ������Ŀ
# data_origin.iloc[nan_list].style.highlight_null(null_color='red')


# ��ȱʧֵ��Ӧ�������޳������������ݼ�
data_filtrated = data_origin.dropna()

# ��ͼ
fig = plt.figure(figsize = (20,15))

i = 1
# ��������ͼ
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1

i = 6
#����ֱ��ͼ
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'filtrated', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# ����ͼ��ʹ��������
fig.savefig('./image/missing_data_delete.jpg')
data_filtrated.to_csv('./data_output/missing_data_delete.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print 'filted_missing_data1 saved at ./image/missing_data_delete.jpg'
print 'data after analysis saved at ./data_output/missing_data_delete.csv'


# �����Ƶ��ֵ���ȱʧֵ


# ����ԭʼ���ݵĿ���,�������򻯴���
data_filtrated = data_origin.copy()
# ��ÿһ�����ݣ��ֱ���д���
for item in name_category+name_value:
    # �������Ƶ�ʵ�ֵ
    most_frequent_value = data_filtrated[item].value_counts().idxmax()
    # �滻ȱʧֵ
    data_filtrated[item].fillna(value = most_frequent_value, inplace = True)

# ���ƿ��ӻ�ͼ
fig = plt.figure(figsize = (20,15))

i = 1
# �Ա�����ԣ���������ͼ
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1    

i = 6
# ����ֵ���ԣ�����ֱ��ͼ
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# ����ͼ��ʹ��������
fig.savefig('./image/missing_data_most.jpg')
data_filtrated.to_csv('./data_output/missing_data_most.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print 'filted_missing_data2 saved at ./image/missing_data_most.jpg'
print 'data after analysis saved at ./data_output/missing_data_most.csv'


# 4.3 ͨ�����Ե���ع�ϵ���ȱʧֵ
# 
# ʹ��pandas��Series��***interpolate()***����������ֵ���Խ��в�ֵ���㣬���滻ȱʧֵ��
# 
# ��ֱ��ͼ�п��Կ��������������ݣ���������ɸ�ֵ��ͬ��ֵ�����Ҿ�ֵ�仯����


# ����ԭʼ���ݵĿ���
data_filtrated = data_origin.copy()
# ����ֵ�����Ե�ÿһ�У����в�ֵ����
for item in name_value:
    data_filtrated[item].interpolate(inplace = True)

# ���ƿ��ӻ�ͼ
fig = plt.figure(figsize = (15,10))

i = 1
# �Ա�����ԣ���������ͼ
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1   
    
i = 6
# ����ֵ���ԣ�����ֱ��ͼ
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# ����ͼ��ʹ��������
fig.savefig('./image/missing_data_corelation.jpg')
data_filtrated.to_csv('./data_output/missing_data_corelation.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print 'filted_missing_data3 saved at ./image/missing_data_corelation.jpg'
print 'data after analysis saved at ./data_output/missing_data_corelation.csv'


# ͨ�����ݶ���֮������������ȱʧֵ

# ��ȱʧֵ��Ϊ0�������ݼ���������

# ����ԭʼ���ݵĿ������������򻯴���
data_norm = data_origin.copy()
# ����ֵ���Ե�ȱʧֵ�滻Ϊ0
data_norm[name_value] = data_norm[name_value].fillna(0)
# �����ݽ�������
data_norm[name_value] = data_norm[name_value].apply(lambda x : (x - np.mean(x)) / (np.max(x) - np.min(x)))

# ���������
score = {}
range_length = len(data_origin)
for i in range(0, range_length):
    score[i] = {}
    for j in range(0, range_length):
        score[i][j] = 0    

# �ڴ����������У���ÿ����������Ŀ��������Ե÷֣���ֵԽ�߲�����Խ��
for i in range(0, range_length):
    for j in range(i, range_length):
        for item in name_category:
            if data_norm.iloc[i][item] != data_norm.iloc[j][item]:
                score[i][j] += 1
        for item in name_value:
            temp = abs(data_norm.iloc[i][item] - data_norm.iloc[j][item])
            score[i][j] += temp
        score[j][i] = score[i][j]

# ����ԭʼ���ݵĿ���
data_filtrated = data_origin.copy()

# ����ȱʧֵ����Ŀ���ú������ƶ���ߣ��÷���ͣ���������Ŀ�ж�Ӧ���Ե�ֵ�滻
for index in nan_list:
    best_friend = sorted(score[index].items(), key=operator.itemgetter(1), reverse = False)[1][0]
    for item in name_value:
        if pd.isnull(data_filtrated.iloc[index][item]):
            if pd.isnull(data_origin.iloc[best_friend][item]):
                data_filtrated.ix[index, item] = data_origin[item].value_counts().idxmax()
            else:
                data_filtrated.ix[index, item] = data_origin.iloc[best_friend][item]

# ���ƿ��ӻ�ͼ
fig = plt.figure(figsize = (16,16))

i = 1
# �Ա�����ԣ���������ͼ
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1   
    
i = 6
# ����ֵ���ԣ�����ֱ��ͼ
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# ����ͼ��ʹ��������
fig.savefig('./pic/missing_data_similarity.jpg')
data_filtrated.to_csv('./data.csv/missing_data_similarity.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print 'filted_missing_data4 saved at ./pic/filted_missing_data4.jpg'
print 'data after analysis saved at ./data_output/missing_data_similarity.csv'
