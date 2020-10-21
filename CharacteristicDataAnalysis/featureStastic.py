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
# baseInfo['empnum'][baseInfo['empnum']>=0].apply(np.log10).plot.box()
# baseInfo['regcap'][baseInfo['regcap']>=0].apply(np.log10).plot.box()
# baseInfo['reccap'][baseInfo['reccap']>0].apply(np.log10).plot.box()
# TODO reccap的中位数为0，使用>=会报错，考虑去掉reccap
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()
#可以看出三个feature均有较多异常点，且异常点都较大，所以使用log，并用中位数/众数填充NAN

# 3）集中度趋势  指一组数据向某一中心靠拢的倾向，核心在于寻找数据的代表值或中心值
#1.均值/算术平均数  均值作为一个统计量，对极端值很敏感，如果数据中存在极端值或者数据是偏态分布，均值不能很好的度量数据的集中趋势，需去掉极端异常值
#只简单用频率作为权重感觉不合适
#2.中位数 将一组观测值按从小到大的顺序排列，位于中间的那个数。即在全部数据中，小于和大于中位数的数据个数相等
#3.众数 数据集中出现最频繁的值，众数不具有唯一性，并不经常用来度量定性变量的中心位置，更适用于定性变量，一般用于离散型变量而非连续性变量
# print(baseInfo['reccap'][baseInfo['reccap']>=0].median())
# print(baseInfo['reccap'][baseInfo['reccap']>=0].mean())
# print(baseInfo['reccap'][baseInfo['reccap']>=0].mode().tolist())
#TODO reccap中位数，众数为0？？奇怪的样子 考虑去掉

# 4）标准差
# a_std = baseInfo['empnum'][baseInfo['empnum']>=0].std()
# b_std = baseInfo['regcap'][baseInfo['regcap']>=0].std()
# c_std = baseInfo['reccap'][baseInfo['reccap']>=0].std()
# print('empnum的标准差为：%.2f, regcap的标准差为：%.2f, reccap的标准差为：%.2f' % (a_std,b_std,c_std))
#empnum的标准差为：15.39, regcap的标准差为：67770.86, reccap的标准差为：36537.98
# TODO 标准差都较大，必须考虑异常值

# 5）画密度曲线，看是否符合正态分布
baseInfo['reccap'][baseInfo['reccap']>0].apply(np.log10).plot(kind = 'kde',style = '--k',grid = True)
# plt.show()
#像正态分布，不加log看不出来
# 如果数据符合正太分布，超过距离平均值3个标准差的值出现的概率为P 小于0.3%，属于极个别小概率事件，定于为异常值

# 6）四分卫间距（分位差）  是上四分卫数Qu与下四分卫数Ql之差，期间包含了全部观测值的一半，其值越大，说明数据的变异程度越大。
#实际就是箱型图的上下两个箱体线，通过箱型图识别异常值，通常被定义为小于QL-1.5IQR或大于Qu+1.5IQR的值。IQR即为四分卫间距。
# a_iqr =baseInfo['empnum'][baseInfo['empnum']>=0].quantile(0.75)-baseInfo['empnum'][baseInfo['empnum']>=0].quantile(0.25)
# b_iqr =baseInfo['regcap'][baseInfo['regcap']>=0].quantile(0.75)- baseInfo['regcap'][baseInfo['regcap']>=0].quantile(0.25)
# c_iqr =baseInfo['reccap'][baseInfo['reccap']>=0].quantile(0.75)- baseInfo['reccap'][baseInfo['reccap']>=0].quantile(0.25)
# print('empnum的分位差为：%.2f, regcap的分位差为：%.2f, reccap的分位差为：%.2f' % (a_iqr,b_iqr,c_iqr))
#empnum的分位差为：3.00, regcap的分位差为：485.00, reccap的分位差为：100.00
#由此感觉去除异常值后的数据变异程度并不大

#正态性检验 利用观测数据判断总体是否服从正态分布的检验称为正态性检验，它是统计判决中重要的一种特殊的拟合优度假设检验。常用判断方法直方图初判 / QQ图判断 / K-S检验
#1.直方图初判
from scipy import stats
from scipy.stats import norm,skew
# sns.set()
# sns.distplot(baseInfo['empnum'][baseInfo['empnum']>0].apply(np.log10),fit = norm) #displot集合了直方图和拟合曲线
# (mu, sigma) = norm.fit(baseInfo['empnum'][baseInfo['empnum']>0].apply(np.log10)) #求出正太分布的均值和标准差
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.show()
#2.QQ图判断
# QQ图通过把测试样本数据的分位数与已知分布相比较，从而来检验数据的分布情况
# QQ图是一种散点图，对应于正态分布的QQ图，就是由标准正态分布的分位数为横坐标，样本值为纵坐标的散点图
# 参考直线：如果数据严格意义上服从正太分布，点将形成一条直线，该正太分布的均值是直线在Y轴上的截距，标准差是该直线的斜率
# fig = plt.figure()
# res = stats.probplot(baseInfo['reccap'][baseInfo['reccap']>0].apply(np.log10),plot=plt)  #样本为Series，默认dist='norm' 拟合直线为正太分布
# plt.show()
#3.KS检验
# u = baseInfo['empnum'][baseInfo['empnum']>=0].mean()  # 计算均值
# std = baseInfo['empnum'][baseInfo['empnum']>=0].std()  # 计算标准差
# print(stats.kstest(baseInfo['empnum'][baseInfo['empnum']>=0], 'norm', (u, std)))
# .kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
# 结果返回两个值：statistic → D值，pvalue → P值
# p值大于0.05，为正态分布
#综上，从图看，勉强算正态分布，但从p值看，不是正态分布（不论加不加log），最终认为，不是正态分布



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
# 绘制直方图
plt.figure(num = 1,figsize = (12,2))
r_cx['频率'].plot(kind = 'bar',
                 width = 0.8,
                 rot = 0,
                 color = 'k',
                 grid = True,
                 alpha = 0.5,
                 figsize = (12,2))
# plt.show()
# 从直方图可看出，有的feature分布较为正常，有的feature分布只有几个值常见，其他值非常少，
# 考虑将这样的数据重新划分枚举，出现少的值统一划到一类中
