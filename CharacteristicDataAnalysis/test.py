from utils.NANprocessing import NaN_process
import pandas as pd
from utils.memoryUtils import reduce_mem_usage

#读取数据
baseInfo = pd.read_csv('../data/train/base_info.csv',engine = 'python')

#缺省值处理
baseInfo=NaN_process(baseInfo,'base_info')
baseInfo=reduce_mem_usage(baseInfo)

