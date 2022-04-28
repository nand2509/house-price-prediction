import pandas as pd
import numpy as np
df = pd.read_csv("benguluruhome.csv")
print(df.head())


print(df.shape)
print(df.info())

for column in df.columns:
    print(df[column].value_counts())
    print("*"*20)

print(df.isna().sum())

print(df.drop(columns=['area_type','availability','society','balcony'],inplace=True))
print(df.describe)
print(df.info())

print(df['location'].value_counts())

df['location'] = df['location'].fillna('Sarjarpur Road')
print(df['size'].value_counts())

df['size'] = df['size'].fillna('2 BHK')
df['bath'] = df['bath'].fillna(df['bath'].median())
print(df.info())
print(df.head())

df['bhk'] = df['size'].str.split().str.get(0).astype(int)
print(df[df.bhk>20])

print(df['total_sqft'].unique())

def convertRange(x):
    temp = x.split('-')
    if len(temp) == 2:
        return(float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None


df['total_sqft']  = df['total_sqft'].apply(convertRange)
print(df.head())


# price per square feet

df['price_per_sqft'] = df['price'] *10000/df['total_sqft']
print(df['price_per_sqft'])

print(df.describe())

print(df['location'].value_counts())

df['location'] = df['location'].apply(lambda x:x.strip())
location_count = df['location'].value_counts()

location_count_less_10 = location_count[location_count<=10]
location_count_less_10

df['location'] = df['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)
print(df['location'].value_counts())

# OTLIERDETECTION AND REMOVAL

print(df.describe())
print((df['total_sqft']/df['bhk']).describe())
df = df[((df['total_sqft']/df['bhk']) >=300)]
print(df.describe())
print(df.shape)



def remove_outlier_sqft():

    df_output = pd.DataFrame()
    for key,subdf in df.gropby('location'):
        m = np.mean(subdf.price_per_sqft)

        st = np.std(subdf.price_per_sqft)

        gen_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output,gen_df],ignore_index='True')

    return df_output

def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for education,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }



        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df = bhk_outlier_remover(df)
print(df.shape)
print(df)


# clean the data

df.drop(columns=['size','price_per_sqft'],inplace=True)


X = df.drop(columns=['price'])
Y = df['price']

# TRAINIG THE MODEL

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
print(X_train.shape)
print(Y_train.shape)

# APPLY LINEAR REGRESSION


column_trans = make_column_transformer((OneHotEncoder(sparse=False),['location']),remainder='passthrough')
scaler = StandardScaler()
lr = LinearRegression(normalize=True)
pipe = make_pipeline(column_trans, scaler, lr)
print(pipe.fit(X_train, Y_train))
y_pred_lr = pipe.predict(X_test)
print(r2_score(Y_test, y_pred_lr))


# APPLYING LASSO


lasso = Lasso()
pipe = make_pipeline(column_trans,scaler,lasso)
print(pipe.fit(X_train,Y_train))

y_pred_lasso = pipe.predict(X_test)
print(r2_score(Y_test,y_pred_lasso))

print(df.head())

# APPLYING RIDGE

ridge = Ridge()
pipe = make_pipeline(column_trans,scaler,ridge)
print(pipe.fit(X_train,Y_train))
y_pred_ridge = pipe.predict(X_test)
print(r2_score(Y_test,y_pred_ridge))

import pickle

pickle.dump(pipe,open('RidgeModel.pkl','wb'))

