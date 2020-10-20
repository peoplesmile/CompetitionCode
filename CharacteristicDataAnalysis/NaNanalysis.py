import pandas as pd

#读取数据
baseInfo = pd.read_csv('../data/train/base_info.csv',engine = 'python')
# annualReportInfo = pd.read_csv('../data/train/annual_report_info.csv',engine = 'python')
# taxInfo = pd.read_csv('../data/train/tax_info.csv',engine = 'python')
# changeInfo = pd.read_csv('../data/train/change_info.csv',engine = 'python')
# newsInfo = pd.read_csv('../data/train/news_info.csv',engine = 'python')
# otherInfo = pd.read_csv('../data/train/other_info.csv',engine = 'python')

#查看数据
# print(baseInfo.head())
# print(annualReportInfo.head())
# print(taxInfo.head())
# print(changeInfo.head())
# print(newsInfo.head())
# print(otherInfo.head())

#查看id是否有重复 False 每条数据都不重复
# print(baseInfo['id'].duplicated().any())
# print(annualReportInfo[['id','ANCHEYEAR']].duplicated().any())
# print(annualReportInfo[annualReportInfo.duplicated()])
# print(annualReportInfo[annualReportInfo[['id','ANCHEYEAR']].duplicated()])
# print(annualReportInfo['STATE'].drop_duplicates())
# print(taxInfo['id'].duplicated().any())
# print(taxInfo[taxInfo.duplicated()])
# print(changeInfo[changeInfo.duplicated()])
# print(newsInfo[newsInfo[['id','public_date']].duplicated()])
# print(newsInfo[newsInfo.duplicated()])
# print(otherInfo[otherInfo['id'].duplicated()])
# print(otherInfo[otherInfo.duplicated()])

#查看每个特征的空值，当int型数据列包含空值时，会将该列转化为float类型
# print(baseInfo.info(verbose=True,null_counts=True))
# print(annualReportInfo.info(verbose=True,null_counts=True))
# print(taxInfo.info(verbose=True,null_counts=True))
# print(changeInfo.info(verbose=True,null_counts=True))
# print(newsInfo.info(verbose=True,null_counts=True))
# print(otherInfo.info(verbose=True,null_counts=True))
