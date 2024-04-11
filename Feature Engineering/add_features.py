import numpy as np

# 동별 어린이보호구역 개수
def child_area(train_df, test_df):
    ### 어린이보호구역개수 결측값 처리
    child_area_mean = train_df.groupby('구')['어린이보호구역개수'].mean()
    # 달서구와 동구가 아예 어린이보호구역개수가 nan임.
    child_mean_sum = child_area_mean.sum()/len(child_area_mean) 
    # 전체평균으로 대체 
    # 동구와 달성군 모두 면적이 넓음
    child_area_mean['달서구'] = child_mean_sum
    child_area_mean['동구'] = child_mean_sum
    def fillna_with_mean_child(row):
        if np.isnan(row['어린이보호구역개수']):
            row['어린이보호구역개수'] = child_area_mean[row['구']]
        return row
    train_df = train_df.apply(fillna_with_mean_child,axis=1)
    test_df = test_df.apply(fillna_with_mean_child,axis=1)
    return train_df, test_df

# 동별 주차장 급지구분
def parking_lot(train_df, test_df):
    # 급지구분 결측값 처리
    parking_mean = train_df.groupby('구')[['급지구분_1','급지구분_2','급지구분_3']].mean()
    cols = ['급지구분_1','급지구분_2','급지구분_3']
    for col in cols:
        def fillna_with_mean_parking(row):
            if np.isnan(row[col]):
                row[col] = parking_mean.loc[row['구'],col]
            return row
        train_df = train_df.apply(fillna_with_mean_parking,axis=1)
        test_df = test_df.apply(fillna_with_mean_parking,axis=1)
    return train_df, test_df


# 동별 보안등 개수
def light(train_df, test_df):
    # 보안등 개수 결측치 처리
    light_mean = train_df.groupby('구')['보안등개수'].mean()
    # 달성군과 서구가 NaN >> 달성군이 면적이 크므로 전체평균값 활용,
    # 서구는 작으므로 Q1 사용
    light_total_mean = light_mean.sum()/len(light_mean)
    light_mean['달성군'] = light_total_mean
    light_mean['서구'] = light_total_mean * 0.25
    def fillna_with_mean_light(row):
        if np.isnan(row['보안등개수']):
            row['보안등개수'] = light_mean[row['구']]
        return row
    train_df = train_df.apply(fillna_with_mean_light,axis=1)
    test_df = test_df.apply(fillna_with_mean_light,axis=1)
    return train_df, test_df


# 동별 횡단보도 개수 : 결측치 처리 방법 1
def cross(train_df,test_df):
    cross_mean = train_df.groupby('구')['횡단보도개수'].mean()
    def fillna_with_mean_cross(row):
        if np.isnan(row['횡단보도개수']):
            row['횡단보도개수']=cross_mean[row['구']]
        return row
    train_df = train_df.apply(fillna_with_mean_cross,axis=1)
    test_df = test_df.apply(fillna_with_mean_cross,axis=1)
    return train_df,test_df

# 동별 횡단보도 개수 : 결측치 처리 방법 2
def cross2(train_df,test_df):
    북구_횡단 = train_df[train_df['구']=='북구']['횡단보도개수'].mean()
    동구_횡단 = train_df[train_df['구']=='동구']['횡단보도개수'].mean()
    중구_횡단 = train_df[train_df['구']=='중구']['횡단보도개수'].mean()

    횡단_dict = {'중구': 중구_횡단, '동구': 동구_횡단, '북구': 북구_횡단}

    for i in ['중구', '동구', '북구']:
        train_df.loc[(train_df['구']==i) & (train_df['횡단보도개수'].isnull()), '횡단보도개수'] = 횡단_dict[i]
        test_df.loc[(test_df['구']==i) & (test_df['횡단보도개수'].isnull()), '횡단보도개수'] = 횡단_dict[i]
    
    return train_df,test_df

# 제한속도 결측값 처리
def speed_limit(train_df, test_df):
    
    speed_mean = train_df.groupby('구')['제한속도'].mean()
    def fillna_with_mean_speed(row):
        if np.isnan(row['제한속도']):
            row['제한속도'] = speed_mean[row['구']]
        return row
    train_df = train_df.apply(fillna_with_mean_speed,axis=1)
    test_df = test_df.apply(fillna_with_mean_speed,axis=1)

    return train_df, test_df

