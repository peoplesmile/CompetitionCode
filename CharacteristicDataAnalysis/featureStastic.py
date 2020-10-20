'''
参考：https://www.jianshu.com/p/9ec363617da8
分布分析 ：研究数据的分布特征和分布类型，分定量数据、定性数据区分基本统计量
    定量数据：了解其分布形式，对称或者非对称，发现某些特大或特小的异常值,可通过绘制散点图，频率分布直方图，茎叶图直观的分析
    定性数据：可用饼图或和条形图直观的显示分布情况
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.NANprocessing import NaN_process
from utils.memoryUtils import reduce_mem_usage

#读取数据
baseInfo = pd.read_csv('../data/train/base_info.csv',engine = 'python')

#缺省值处理
baseInfo=NaN_process(baseInfo,'base_info')
baseInfo=reduce_mem_usage(baseInfo)
# print(baseInfo.info())
#一共18个字段
#定量字段：empnum,regcap,reccap
#定性字段：oplocdistrict,industryphy,enttype,enttypeitem,state,orgid,jobid,adbusign,
#        townsign,regtype,compform,opform,venind,enttypeminu,enttypegb

# 定量数据的分布分析
cols=['empnum','regcap','reccap']
#1)极差
def d_range(df,cols):
    for col in cols:
        crange = df[col][df[col]!=-1].max() - df[col][df[col]!=-1].min()
        # print(df[col][df[col]!=-1].max())
        # print(df[col][df[col]!=-1].min())
        # print(df[col][df[col]!=-1].mean())
        print('%s极差为 %f' % (col, crange))
d_range(baseInfo,cols)
# empnum极差为 1500.000000
# regcap极差为 5000100.000000
# reccap极差为 1278900.000000
# 极差较大，考虑使用归一化/标准化之前使用log处理

# 2）箱形图
# baseInfo['empnum'][baseInfo['empnum']>0].apply(np.log10).plot.box()
# baseInfo['regcap'][baseInfo['regcap']>0].apply(np.log10).plot.box()
# baseInfo['reccap'][baseInfo['reccap']>0].apply(np.log10).plot.box()
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()
#可以看出三个feature均有较多异常点，且异常点都较大，所以使用log，并用中位数/众数填充NAN


# 定性数据的分布分析
cols=['oplocdistrict','industryphy','enttype','enttypeitem','state','orgid','jobid','adbusign',
       'townsign','regtype','compform','opform','venind','enttypeminu','enttypegb']

#1）频率分布
col='enttypegb'
cx_g = baseInfo[col].value_counts(sort=True)
# 统计频率
r_cx = pd.DataFrame(cx_g)
r_cx.rename(columns ={cx_g.name:'频数'}, inplace = True)  # 修改频数字段名
r_cx['频率'] = r_cx / r_cx['频数'].sum()  # 计算频率
r_cx['频率%'] = r_cx['频率'].apply(lambda x: "%.2f%%" % (x*100))  # 以百分比显示频率
# r_cx.style.bar(subset=['频率'], color='#d65f5f',width=100)
# print(r_cx)
# 绘制直方图
plt.figure(num = 1,figsize = (12,2))
r_cx['频率'].plot(kind = 'bar',
                 width = 0.8,
                 rot = 0,
                 color = 'k',
                 grid = True,
                 alpha = 0.5,
                 figsize = (12,2))
plt.show()
# 从直方图可看出，有的feature分布较为正常，有的feature分布只有几个值常见，其他值非常少，
# 考虑将这样的数据重新划分枚举，出现少的值统一划到一类中
