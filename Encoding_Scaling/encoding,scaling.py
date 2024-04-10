import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder

def encoding(train_df,test_df):
    '''
    input: 원래 train_df, test_df
    output: scaled train_df
    '''
    categorical_features = list(train_df.dtypes[train_df.dtypes == "object"].index)
    numeric_features = list(train_df.dtypes[train_df.dtypes == "int"].index)
    numeric_features.extend(list(train_df.dtypes[train_df.dtypes == "float"].index))

    # 원핫 인코딩
    for df in [['구','사고유형']]:
        # train_df와 test_df의 기상상태 및 노면상태 열 선택
        train_categorical_data = train_df[[df]]
        test_categorical_data = test_df[[df]]

        # OneHotEncoder 인스턴스 생성 및 fit_transform 수행
        encoder = OneHotEncoder()
        train_encoded = encoder.fit_transform(train_categorical_data)
        test_encoded = encoder.transform(test_categorical_data)

        # OneHotEncoder가 사용한 카테고리 목록을 가져와서 카테고리 이름을 열 이름으로 변환
        feature_names = encoder.get_feature_names_out([df])

        # 밀집 행렬로 변환 (선택 사항)
        train_encoded_dense = train_encoded.toarray()
        test_encoded_dense = test_encoded.toarray()

        # 데이터프레임으로 변환 (선택사항)
        train_encoded_df = pd.DataFrame(train_encoded_dense, columns=feature_names, index=train_df.index)
        test_encoded_df = pd.DataFrame(test_encoded_dense, columns=feature_names, index=test_df.index)

        # 기존 열 제거
        train_df = train_df.drop([df], axis=1)
        test_df = test_df.drop([df], axis=1)
        # 인코딩된 열 추가
        train_df = pd.concat([train_df, train_encoded_df], axis=1)
        test_df = pd.concat([test_df, test_encoded_df], axis=1)

    # 타겟 인코딩
    # categorical_features = ['도로형태','동','기상상태', '노면상태','season','요일']
    for i in categorical_features:
        tr_encoder = TargetEncoder(cols=[i])
        train_df[i] = tr_encoder.fit_transform(train_df[i], train_df['ECLO'])
        test_df[i] = tr_encoder.transform(test_df[i])
    
    # +) 주중, 주말로 변환(선택사항)
    def week_end(x):
        if x in ['월요일','화요일','수요일','목요일','금요일']:
            return 0
        else:
            return 1
    train_df['요일'] = train_df['요일'].apply(week_end)
    test_df['요일'] = test_df['요일'].apply(week_end)

    return train_df, test_df


from sklearn.preprocessing import RobustScaler
Scaler = RobustScaler()
def scaling(train_x, test_x):
    '''
    input : dataframe of train_x and test_x
    output : scaled train_x, test_x
    '''
    train_data_Scaled = Scaler.fit_transform(train_x)
    test_data_Scaled = Scaler.transform(test_x)
    return train_data_Scaled, test_data_Scaled
