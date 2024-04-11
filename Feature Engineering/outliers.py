### 이상치 처리
import numpy as np
def tukey_fences(data): # data는 1차원 배열
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    lf = q1 - (iqr*1.5)
    uf = q3 + (iqr*1.5)
    idxs = np.where((data>uf) | (data<lf))[0] # condition이 True일 때, data의 인덱스를 반환
    return idxs, lf, uf

def outlier_to_lf_uf(data, outlier_function=tukey_fences): # data는 Series객체
    idxs, lf, uf = outlier_function(data)
    print('lf와 uf: ',lf,uf)
    print(list(idxs))
    for i in idxs:
        if data.iloc[i] <= lf:
            data.iloc[i] = lf
        elif data.iloc[i] >= uf:
            data.iloc[i] = uf
            
    return data

### 이상치 처리 적용할 때 예시
'''
train_df['보안등개수'] = outlier_to_lf_uf(train_df['보안등개수'],tukey_fences)
'''

# 반복문 사용할 경우
'''
for col in outlier_columns:
    train_df[col] = outlier_to_lf_uf(train_df[col], tukey_fences)
    test_df[col] = outlier_to_lf_uf(test_df[col], tukey_fences)
'''