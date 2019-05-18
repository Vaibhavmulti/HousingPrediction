import sklearn
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

X=pd.read_csv("train.csv")
X_test=pd.read_csv("test.csv")

# drop id as it as nothing to do with saleprice
X=X.drop("Id",axis=1)
X_test=X_test.drop("Id",axis=1)

# Some numerical features are actually categorial
X=X.replace({'MSSubClass':{20:'MSSubClass20',30:'MSSubClass30',40:'MSSubClass40',45:'MSSubClass45',
                           50:'MSSubClass50',60:'MSSubClass60',70:'MSSubClass70',75:'MSSubClass75',
                           80:'MSSubClass80',85:'MSSubClass85',90:'MSSubClass90',
                            120:'MSSubClass120',150:'MSSubClass150',160:'MSSubClass160',
                            180:'MSSubClass180',190:'MSSubClass190'}})
X_test=X_test.replace({'MSSubClass':{20:'MSSubClass20',30:'MSSubClass30',40:'MSSubClass40',45:'MSSubClass45',
                           50:'MSSubClass50',60:'MSSubClass60',70:'MSSubClass70',75:'MSSubClass75',
                           80:'MSSubClass80',85:'MSSubClass85',90:'MSSubClass90',
                            120:'MSSubClass120',150:'MSSubClass150',160:'MSSubClass160',
                            180:'MSSubClass180',190:'MSSubClass190'}})

# Street : Pave or not pave
X=X.replace({'Street':{'Grvl':1,'Pave':2}})
X_test=X_test.replace({'Street':{'Grvl':1,'Pave':2}})

# Alley : no alley / paved or not
X['Alley']=X['Alley'].fillna(0)
X_test['Alley']=X_test['Alley'].fillna(0)
X=X.replace({'Alley':{'Grvl':1,'Pave':2}})
X_test=X_test.replace({'Alley':{'Grvl':1,'Pave':2}})

# Some categorical features are actually numerical .ExterQual, ExterCond
X=X.replace({'ExterQual':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'ExterQual':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X=X.replace({'ExterCond':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'ExterCond':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X['BsmtQual']=X['BsmtQual'].fillna(0)
X_test['BsmtQual']=X_test['BsmtQual'].fillna(0)
X=X.replace({'BsmtQual':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'BsmtQual':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X['BsmtCond']=X['BsmtCond'].fillna(0)
X_test['BsmtCond']=X_test['BsmtCond'].fillna(0)
X=X.replace({'BsmtCond':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'BsmtCond':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X['BsmtFinType2']=X['BsmtFinType2'].fillna(0)
X_test['BsmtFinType2']=X_test['BsmtFinType2'].fillna(0)
X=X.replace({'BsmtFinType2':{'Unf':0.5,'LwQ':1,'Rec':2,'BLQ':3,'ALQ':4,'GLQ':5}})
X_test=X_test.replace({'BsmtFinType2':{'Unf':0.5,'LwQ':1,'Rec':2,'BLQ':3,'ALQ':4,'GLQ':5}})

X['BsmtFinType1']=X['BsmtFinType1'].fillna(0)
X_test['BsmtFinType1']=X_test['BsmtFinType1'].fillna(0)
X=X.replace({'BsmtFinType1':{'Unf':0.5,'LwQ':1,'Rec':2,'BLQ':3,'ALQ':4,'GLQ':5}})
X_test=X_test.replace({'BsmtFinType1':{'Unf':0.5,'LwQ':1,'Rec':2,'BLQ':3,'ALQ':4,'GLQ':5}})

X['BsmtExposure']=X['BsmtExposure'].fillna(0)
X_test['BsmtExposure']=X_test['BsmtExposure'].fillna(0)
X=X.replace({'BsmtExposure':{'No':0,'Mn':1,'Av':2,'Gd':3}})
X_test=X_test.replace({'BsmtExposure':{'No':0,'Mn':1,'Av':2,'Gd':3}})

X=X.replace({'HeatingQC':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'HeatingQC':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X=X.replace({'CentralAir':{'N':0,'Y':1}})
X_test=X_test.replace({'CentralAir':{'N':0,'Y':1}})

X=X.replace({'KitchenQual':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'KitchenQual':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X['FireplaceQu']=X['FireplaceQu'].fillna(0)
X_test['FireplaceQu']=X_test['FireplaceQu'].fillna(0)
X=X.replace({'FireplaceQu':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'FireplaceQu':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X['GarageQual']=X['GarageQual'].fillna(0)
X_test['GarageQual']=X_test['GarageQual'].fillna(0)
X=X.replace({'GarageQual':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'GarageQual':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X['GarageCond']=X['GarageCond'].fillna(0)
X_test['GarageCond']=X_test['GarageCond'].fillna(0)
X=X.replace({'GarageCond':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'GarageCond':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

X['GarageFinish']=X['GarageFinish'].fillna(0)
X_test['GarageFinish']=X_test['GarageFinish'].fillna(0)
X=X.replace({'GarageFinish':{'Unf':1,'RFn':2,'Fin':3}})
X_test=X_test.replace({'GarageFinish':{'Unf':1,'RFn':2,'Fin':3}})

X['PoolQC']=X['PoolQC'].fillna(0)
X_test['PoolQC']=X_test['PoolQC'].fillna(0)
X=X.replace({'PoolQC':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})
X_test=X_test.replace({'PoolQC':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}})

#MoSold
X=X.replace({'MoSold':{1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                       7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}})
X_test=X_test.replace({'MoSold':{1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                       7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}})

#MasVnrArea if none then 0 area
X['MasVnrArea']=X['MasVnrArea'].fillna(0)
X_test['MasVnrArea']=X_test['MasVnrArea'].fillna(0)

#GarageYrBlt and YearBuilt are highly corelated 0.825667 using df.corr() so we can drop garageyrblt
X=X.drop('GarageYrBlt',axis=1)
X_test=X_test.drop('GarageYrBlt',axis=1)

features_to_fill_NAN=['MiscFeature','Fence','GarageType','MasVnrType']
X[features_to_fill_NAN]=X[features_to_fill_NAN].fillna("NotAvailable")
X_test[features_to_fill_NAN]=X_test[features_to_fill_NAN].fillna("NotAvailable")

#dropping Electrical as only 1 observation missing
X=X.dropna(subset=['Electrical'])
X_test=X_test.dropna(subset=['Electrical'])

#imputing LotArea with mean
X['LotFrontage']=X['LotFrontage'].fillna(X['LotFrontage'].mean())
X_test['LotFrontage']=X_test['LotFrontage'].fillna(X_test['LotFrontage'].mean())

# 2010 is the max 'YearBuilt' so the newest house built is in 2010
# subtracting the col from 2010 hence making it more meaningful
# doing the same with  YearRemodAdd (last modification added to the house)

X['YearBuilt']=X['YearBuilt']-2010
X['YearRemodAdd']=X['YearRemodAdd']-2010
X_test['YearBuilt']=X_test['YearBuilt']-2010
X_test['YearRemodAdd']=X_test['YearRemodAdd']-2010

X['YrSold']=X['YrSold'].astype("object")   # one hot encode this  as only 5 unique years

## OUTLIERS?

#print X.corr().SalePrice.sort_values(ascending=False)
#Lets look at the highest coorealted ones

'''X.plot.scatter('GrLivArea','SalePrice')
2 clear outliers where grlivarea > 4500
#X.plot.scatter('GrLivArea','SalePrice')
#plt.show()
'''
X=X.loc[X.GrLivArea<=4500]

# garage area and garage cars highly related so we can drop garage cars
X=X.drop("GarageCars",axis=1)
X_test=X_test.drop("GarageCars",axis=1)

'''
1 maybe totalbsmtsf > 3000 and saleprice < 300000
X.plot.scatter('TotalBsmtSF','SalePrice')
plt.show()
'''
#print X.loc[(X.TotalBsmtSF >  3000) & (X.SalePrice < 300000)].index.item()
X=X.drop(X.loc[(X.TotalBsmtSF >  3000) & (X.SalePrice < 300000)].index.item())


Y=X['SalePrice']
X=X.drop('SalePrice',axis=1)

'''
for one hot encoding we concat the 2 datasets in order to make the number of columns after get_dummies equal!
adding another column of all 0's and 1's in order to seperate the datasets after one hot encoding
'''

X['zo']=0
X_test['zo']=1
frames = [X, X_test]
X_concat=pd.concat(frames)
X_concat=pd.get_dummies(X_concat)
X=X_concat.loc[X_concat['zo']==0]
X_test=X_concat.loc[X_concat['zo']==1]
X=X.drop('zo',axis=1)
X_test=X_test.drop('zo',axis=1)



# LETS TUNE MODEL NOW
#lets find an estimate of number of estimatores first using the internal xgboost.cv function
#using the initial random/default parameters to find the estimators

parameters={
    'learning_rate':0.1,
    'n_estimators':1000,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':0.0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'seed':75
}

cv_folds=5
early_stopping_rounds=50

xgtrain = xgb.DMatrix(X.values, label=Y.values)
cvresult = xgb.cv(parameters, xgtrain, num_boost_round=parameters['n_estimators'], nfold=cv_folds,
                  early_stopping_rounds=early_stopping_rounds,seed=75,metrics='rmse')
parameters['n_estimators']=cvresult.shape[0]

'''param_test1 = {
 'max_depth':range(3,10,2),             #5  turns out to max
 'min_child_weight':range(1,6,2)        #5  turns out to max
}


gsearch1 = GridSearchCV(estimator = XGBRegressor(parameters),
 param_grid = param_test1, cv=5)
gsearch1.fit(X,Y)
print gsearch1.best_params_, gsearch1.best_score_
'''

parameters['max_depth']=5
parameters['min_child_weight']=5

'''
param_test2 = {
 'max_depth':[4,5,6],           # Still 5,5
 'min_child_weight':[4,5,6]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor(parameters),
 param_grid = param_test2, cv=5)
gsearch1.fit(X,Y)
print gsearch1.best_params_, gsearch1.best_score_
'''

'''
param_test3 = {
 'gamma': [i/10.0 for i in range(0,5)]      # 0 turns out to be best
}

gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
    n_estimators=parameters['n_estimators'],
    max_depth=5,
    min_child_weight=5,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=75),
 param_grid = param_test3, cv=5)
gsearch1.fit(X,Y)
print gsearch1.best_params_, gsearch1.best_score_
'''
parameters['gamma']=0.0

parameters['n_estimators']=1000
cvresult = xgb.cv(parameters, xgtrain, num_boost_round=parameters['n_estimators'], nfold=cv_folds,
                  early_stopping_rounds=early_stopping_rounds,seed=75,metrics='rmse')
parameters['n_estimators']=cvresult.shape[0]

'''
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],          #0.7
 'colsample_bytree':[i/10.0 for i in range(6,10)]       #0.6
}

gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
    n_estimators=parameters['n_estimators'],
    max_depth=5,
    min_child_weight=5,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=75),
 param_grid = param_test4, cv=5)
gsearch1.fit(X,Y)
print gsearch1.best_params_, gsearch1.best_score_
'''

parameters['subsample']=0.7
parameters['colsample_bytree']=0.6

'''
param_test5 = {
 'subsample':[i/100.0 for i in range(65,86,5)],  # still 0.7
 'colsample_bytree':[i/100.0 for i in range(50,76,5)]     # still 0.6
}

gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
    n_estimators=parameters['n_estimators'],
    max_depth=5,
    min_child_weight=5,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.6,
    seed=75),
 param_grid = param_test5, cv=5)
gsearch1.fit(X,Y)
print gsearch1.best_params_, gsearch1.best_score_
'''

'''
param_test6 = {
'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
 'reg_alpha':[0.2,0.25,0.3,0.35]    #0.2
}
print parameters
#0.1
print param_test6
gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,
    n_estimators=parameters['n_estimators'],
    max_depth=parameters['max_depth'],
    min_child_weight=parameters['min_child_weight'],
    gamma=parameters['gamma'],
    subsample=parameters['subsample'],
    colsample_bytree=parameters['colsample_bytree'],
    seed=75),
 param_grid = param_test6, cv=5)
gsearch1.fit(X,Y)
print gsearch1.best_params_, gsearch1.best_score_
'''
parameters['reg_alpha']=0.2
parameters['learning_rate']=0.01
parameters['n_estimators']=5000
cvresult = xgb.cv(parameters, xgtrain, num_boost_round=parameters['n_estimators'], nfold=cv_folds,
                  early_stopping_rounds=early_stopping_rounds,seed=75,metrics='rmse')
parameters['n_estimators']=cvresult.shape[0]
print parameters
finalmod=XGBRegressor(n_estimators=parameters['n_estimators'],subsample=parameters['subsample'],
reg_alpha=parameters['reg_alpha'],seed=parameters['seed'],colsample_bytree=parameters['colsample_bytree'],
gamma=parameters['gamma'],learning_rate=parameters['learning_rate'],max_depth=parameters['max_depth'],
min_child_weight=parameters['min_child_weight'])
finalmod.fit(X,Y)
predd= finalmod.predict(X_test)

submit=pd.DataFrame({"Id":[i for i in range(1461,2920)],"SalePrice":predd})
submit.to_csv("help.csv",index=False)
print submit
