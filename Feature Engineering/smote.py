# ECLO에 Smote 알고리즘 적용
def ECLO_range(x):
    '''
    input : x would be 'ECLO' column of df
    output : returns the range of ECLO (categorized)
    '''
    if x<=3:
        return 0
    elif x<=6:
        return 1
    elif x<=10:
        return 2
    elif x<=15:
        return 3
    elif x<=20:
        return 4
    else:
        return 5

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Initialize
smote = SMOTE()
# 일일이 객체를 지정해줘야 이후 디코딩 가능
le_dayofweek = LabelEncoder() 
le_weather = LabelEncoder()
le_road_state = LabelEncoder()
le_road_type = LabelEncoder()
le_acc_type = LabelEncoder()
le_gu = LabelEncoder()
le_dong = LabelEncoder()
le_season = LabelEncoder()

def smote(train_df, train_y):
    '''
    input : dataframe of train data (X,y)
    output : smote resampled train data (X,y)
    '''
    cat_list = ['요일','기상상태','노면상태','도로형태','사고유형','구','동','season']
    train_df['요일'] = le_dayofweek.fit_transform(train_df['요일'])
    train_df['기상상태'] = le_weather.fit_transform(train_df['기상상태'])
    train_df['노면상태'] = le_road_state.fit_transform(train_df['노면상태'])
    train_df['도로형태'] = le_road_type.fit_transform(train_df['도로형태'])
    train_df['사고유형'] = le_acc_type.fit_transform(train_df['사고유형'])
    train_df['구'] = le_road_state.fit_transform(train_df['구'])
    train_df['동'] = le_road_type.fit_transform(train_df['동'])
    train_df['season'] = le_acc_type.fit_transform(train_df['season'])
    X_train, X_test, y_train, y_test = train_test_split(train_df, train_y, test_size=0.2, random_state = 42)

    X_train_sm, Y_train_sm = smote.fit_resample(X_train, y_train)

    # eclo값은 1 이하이다.
    X_train_sm.loc[X_train_sm['ECLO']>1, 'ECLO'] = 1

    # 타겟인코딩을 위한 디코딩
    X_train_sm['요일'] = le_dayofweek.inverse_transform(X_train_sm['요일'])
    X_train_sm['기상상태'] = le_weather.inverse_transform(X_train_sm['기상상태'])
    X_train_sm['노면상태'] = le_road_state.inverse_transform(X_train_sm['노면상태'])
    X_train_sm['도로형태'] = le_road_type.inverse_transform(X_train_sm['도로형태'])
    X_train_sm['사고유형'] = le_acc_type.inverse_transform(X_train_sm['사고유형'])
    X_train_sm['구'] = le_road_state.inverse_transform(X_train_sm['구'])
    X_train_sm['동'] = le_road_type.inverse_transform(X_train_sm['동'])
    X_train_sm['season'] = le_acc_type.inverse_transform(X_train_sm['season'])

    return X_train_sm, Y_train_sm

