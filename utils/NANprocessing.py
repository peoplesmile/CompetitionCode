def NaN_process(df,df_name):
    if df_name=='base_info':
        # 去除特征 parnum,exenum,ptbusscope,midpreindcode,protype,forreccap,forregcap,congro
        df.drop(
            labels=['parnum', 'exenum', 'ptbusscope', 'midpreindcode', 'protype', 'forreccap', 'forregcap', 'congro'],
            axis=1,
            inplace=True
        )
        #oplocdistrict,共16个值，无缺省值，转换为catogory类型
        df['oplocdistrict']=df['oplocdistrict'].astype('category')
        #industryphy,共20个值，无缺省值，转换为catogory类型
        df['industryphy'] = df['industryphy'].astype('category')
        #TODO industryco 行业细类代码,有346个值，枚举太多了，暂时去掉这个特征，有缺省值
        df.drop(
            labels=['industryco'],
            axis=1,
            inplace=True
        )
        # TODO dom 经营地址,字符串类型，有23278个值，枚举太多了，暂时去掉这个特征
        df.drop(
            labels=['dom'],
            axis=1,
            inplace=True
        )
        # TODO opscope 经营范围,中文文字，涉及自然语言处理技术，暂时去掉这个特征
        df.drop(
            labels=['opscope'],
            axis=1,
            inplace=True
        )
        # enttype,共17个值，无缺省值，转换为catogory类型
        df['enttype'] = df['enttype'].astype('category')
        # enttypeitem,共32个值，有缺省值，转换为catogory类型
        df.fillna(value={'enttypeitem':-1},inplace=True)
        df['enttypeitem'] = df['enttypeitem'].astype('int')
        df['enttypeitem'] = df['enttypeitem'].astype('category')
        # TODO opfrom 经营期限起,6597种值，无缺省值，暂时去掉这个特征，后续考虑使用距今的天数（int）代替，缺省值使用平均值等
        df.drop(
            labels=['opfrom'],
            axis=1,
            inplace=True
        )
        # TODO opto 经营期限止,5747种值，有缺省值，暂时去掉这个特征，后续考虑使用距今的天数（int）代替，需要查看缺省的原因决定缺省值是用什么代替
        df.drop(
            labels=['opto'],
            axis=1,
            inplace=True
        )
        # state,共6个值，无缺省值，转换为catogory类型
        df['state'] = df['state'].astype('category')
        # orgid,共11个值，无缺省值，转换为catogory类型
        df['orgid'] = df['orgid'].astype('category')
        # jobid,共7个值，无缺省值，转换为catogory类型
        df['jobid'] = df['jobid'].astype('category')
        # adbusign,共2个值，无缺省值，转换为catogory类型
        df['adbusign'] = df['adbusign'].astype('category')
        # townsign,共2个值，无缺省值，转换为catogory类型
        df['townsign'] = df['townsign'].astype('category')
        # regtype,共3个值，无缺省值，转换为catogory类型
        df['regtype'] = df['regtype'].astype('category')
        # TODO empnum 从业人数,使用-1代替NAN，转换为int类型,有异常值，极值大，不能用平均数填充
        df.fillna(value={'empnum': int(-1)}, inplace=True)
        df['empnum'] = df['empnum'].astype('int')

        # compform 组织形式,共3个值，转换为category类型
        df.fillna(value={'compform': -1}, inplace=True)
        df['compform'] = df['compform'].astype('int')
        df['compform'] = df['compform'].astype('category')
        # TODO opform,共34个值，有缺失值，转换为category类型,有nan类型和空格类型（*），空格是什么？
        df.fillna(value={'opform': -1}, inplace=True)
        df['opform'] = df['opform'].astype('category')
        # venind,共4个值，有缺省值，转换为category类型
        df.fillna(value={'venind': -1}, inplace=True)
        df['venind'] = df['venind'].astype('int')
        df['venind'] = df['venind'].astype('category')
        # enttypeminu,共27个值，有缺省值，转换为category类型
        df.fillna(value={'enttypeminu': -1}, inplace=True)
        df['enttypeminu'] = df['enttypeminu'].astype('int')
        df['enttypeminu'] = df['enttypeminu'].astype('category')
        # TODO oploc 经营尝试,5351种值，有缺省值，暂时去掉这个特征
        df.drop(
            labels=['oploc'],
            axis=1,
            inplace=True
        )
        # TODO regcap，float，使用-1代替NAN，有异常值，极值大，不能用平均数填充
        df.fillna(value={'regcap': -1}, inplace=True)
        # TODO reccap，float，有缺省值，使用-1代替NAN，有异常值，极值大，不能用平均数填充
        df.fillna(value={'reccap': -1}, inplace=True)
        # enttypegb,共53个值，无缺省值，转换为category类型
        df.fillna(value={'enttypegb': -1}, inplace=True)
        df['enttypegb'] = df['enttypegb'].astype('int')
        df['enttypegb'] = df['enttypegb'].astype('category')

        #共剩下18个特征+1 id
        # print(df.info())

        # print(df['enttypegb'])
        # print(len(df['enttypegb'].drop_duplicates()))
    return df
