#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:12:43 2017

@author: inovia
"""
import pandas as pd


#cust_na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NULL', 'NaN', 'nan']
data = pd.read_csv("/private/var/www/DSSPSubmission/data/train.csv")

#these are our X feature
X_df = data


import math
import random
from statsmodels.robust.scale import mad

pd.options.display.mpl_style = 'default'
X_df.boxplot()


#LotFrontage	True
X_df.hist(column='LotFrontage')

E_LotFrontage = X_df['LotFrontage'].dropna().median()
#E = 67

Sigma_LotFrontage = mad(X_df['LotFrontage'].dropna(), c=1)
#Sigma =12.5

def cleanLotFrontage(x):
    if math.isnan(x):
        return random.gauss(E_LotFrontage, Sigma_LotFrontage)
    return x

X_df['LotFrontage'] = X_df['LotFrontage'].apply(cleanLotFrontage)


#MasVnrType	True
X_df['MasVnrType'] = X_df['MasVnrType'].fillna('None')

#MasVnrArea	True
def cleanMasVnrArea(x):
    if math.isnan(x):
        return 0
    return x

X_df['MasVnrArea'] = X_df['MasVnrArea'].apply(cleanMasVnrArea)



#we drop the year of built of the garage
X_df = X_df.drop("GarageYrBlt",1)


#clean other data
X_df['Alley'] = X_df['Alley'].fillna('None')
X_df['BsmtQual'] = X_df['BsmtQual'].fillna('None')
X_df['BsmtCond'] = X_df['BsmtCond'].fillna('None')
X_df['BsmtExposure'] = X_df['BsmtExposure'].fillna('None')
X_df['BsmtFinType2'] = X_df['BsmtFinType2'].fillna('None')
X_df['FireplaceQu'] = X_df['FireplaceQu'].fillna('None')
X_df['GarageType'] = X_df['GarageType'].fillna('None')
X_df['GarageFinish'] = X_df['GarageFinish'].fillna('None')
X_df['GarageQual'] = X_df['GarageQual'].fillna('None')
X_df['GarageCond'] = X_df['GarageCond'].fillna('None')
X_df['PoolQC'] = X_df['PoolQC'].fillna('None')
X_df['Fence'] = X_df['Fence'].fillna('None')
X_df['MiscFeature'] = X_df['MiscFeature'].fillna('None')
X_df['MSZoning'] = X_df['MSZoning'].fillna('None')
X_df['Utilities'] = X_df['Utilities'].fillna('AllPub')
X_df['Exterior1st'] = X_df['Exterior1st'].fillna('None')
X_df['Exterior2nd'] = X_df['Exterior2nd'].fillna('None')
X_df['BsmtFinType1'] = X_df['BsmtFinType1'].fillna('None')
X_df['BsmtFinType1'] = X_df['BsmtFinType1'].fillna('None')
X_df['BsmtFinSF1'] = X_df['BsmtFinSF1'].fillna(0)
X_df['BsmtFinSF2'] = X_df['BsmtFinSF2'].fillna(0)
X_df['BsmtUnfSF'] = X_df['BsmtUnfSF'].fillna(0)
X_df['TotalBsmtSF'] = X_df['TotalBsmtSF'].fillna(0)
X_df['BsmtFullBath'] = X_df['BsmtFullBath'].fillna(0)
X_df['BsmtHalfBath'] = X_df['BsmtHalfBath'].fillna(0)
X_df['KitchenQual'] = X_df['KitchenQual'].fillna('UKN')
X_df['Functional'] = X_df['Functional'].fillna('Typ')
X_df['GarageCars'] = X_df['GarageCars'].fillna(0)
X_df['GarageArea'] = X_df['GarageArea'].fillna(0)
X_df['SaleType'] = X_df['SaleType'].fillna('Oth')



#we drop the id 
X_df = X_df.drop("Id",1)

#We create the new dimensions for MSSubClass
X_df= X_df.join(pd.get_dummies(X_df['MSSubClass'],prefix='MSSubClass'))
X_df = X_df.drop("MSSubClass",1)

#We create the new dimensions for MSZoning
X_df= X_df.join(pd.get_dummies(X_df['MSZoning'],prefix='MSZoning'))
X_df = X_df.drop("MSZoning",1)

#We create the new dimensions for Street
X_df= X_df.join(pd.get_dummies(X_df['Street'],prefix='Street'))
X_df = X_df.drop("Street",1)

#We create the new dimensions for Alley
X_df= X_df.join(pd.get_dummies(X_df['Alley'],prefix='Alley'))
X_df = X_df.drop("Alley",1)

#We create the new dimensions for LotShape
X_df= X_df.join(pd.get_dummies(X_df['LotShape'],prefix='LotShape'))
X_df = X_df.drop("LotShape",1)

#We create the new dimensions for LandContour
X_df= X_df.join(pd.get_dummies(X_df['LandContour'],prefix='LandContour'))
X_df = X_df.drop("LandContour",1)

#We create the new dimensions for Utilities
X_df= X_df.join(pd.get_dummies(X_df['Utilities'],prefix='Utilities'))
X_df = X_df.drop("Utilities",1)

#We create the new dimensions for LotConfig
X_df= X_df.join(pd.get_dummies(X_df['LotConfig'],prefix='LotConfig'))
X_df = X_df.drop("LotConfig",1)

#We create the new dimensions for LandSlope
X_df= X_df.join(pd.get_dummies(X_df['LandSlope'],prefix='LandSlope'))
X_df = X_df.drop("LandSlope",1)

#We create the new dimensions for Neighborhood
X_df= X_df.join(pd.get_dummies(X_df['Neighborhood'],prefix='Neighborhood'))
X_df = X_df.drop("Neighborhood",1)

#We create the new dimensions for Condition1
X_df= X_df.join(pd.get_dummies(X_df['Condition1'],prefix='Condition1'))
X_df = X_df.drop("Condition1",1)

#We create the new dimensions for Condition2
X_df= X_df.join(pd.get_dummies(X_df['Condition2'],prefix='Condition2'))
X_df = X_df.drop("Condition2",1)

#We create the new dimensions for BldgType
X_df= X_df.join(pd.get_dummies(X_df['BldgType'],prefix='BldgType'))
X_df = X_df.drop("BldgType",1)

#We create the new dimensions for HouseStyle
X_df= X_df.join(pd.get_dummies(X_df['HouseStyle'],prefix='HouseStyle'))
X_df = X_df.drop("HouseStyle",1)

#We create the new dimensions for HouseStyle
X_df= X_df.join(pd.get_dummies(X_df['YearBuilt'],prefix='YearBuilt'))
X_df = X_df.drop("YearBuilt",1)

#We create the new dimensions for YearRemodAdd
X_df= X_df.join(pd.get_dummies(X_df['YearRemodAdd'],prefix='YearRemodAdd'))
X_df = X_df.drop("YearRemodAdd",1)

#We create the new dimensions for RoofStyle
X_df= X_df.join(pd.get_dummies(X_df['RoofStyle'],prefix='RoofStyle'))
X_df = X_df.drop("RoofStyle",1)

#We create the new dimensions for RoofMatl
X_df= X_df.join(pd.get_dummies(X_df['RoofMatl'],prefix='RoofMatl'))
X_df = X_df.drop("RoofMatl",1)

#We create the new dimensions for Exterior1st
X_df= X_df.join(pd.get_dummies(X_df['Exterior1st'],prefix='Exterior1st'))
X_df = X_df.drop("Exterior1st",1)

#We create the new dimensions for Exterior2nd
X_df= X_df.join(pd.get_dummies(X_df['Exterior2nd'],prefix='Exterior2nd'))
X_df = X_df.drop("Exterior2nd",1)

#We create the new dimensions for MasVnrType
X_df= X_df.join(pd.get_dummies(X_df['MasVnrType'],prefix='MasVnrType'))
X_df = X_df.drop("MasVnrType",1)

#We create the new dimensions for MasVnrArea
X_df= X_df.join(pd.get_dummies(X_df['MasVnrArea'],prefix='MasVnrArea'))
X_df = X_df.drop("MasVnrArea",1)

#We create the new dimensions for ExterQual
X_df= X_df.join(pd.get_dummies(X_df['ExterQual'],prefix='ExterQual'))
X_df = X_df.drop("ExterQual",1)

#We create the new dimensions for ExterCond
X_df= X_df.join(pd.get_dummies(X_df['ExterCond'],prefix='ExterCond'))
X_df = X_df.drop("ExterCond",1)

#We create the new dimensions for Foundation
X_df= X_df.join(pd.get_dummies(X_df['Foundation'],prefix='Foundation'))
X_df = X_df.drop("Foundation",1)

#We create the new dimensions for BsmtQual
X_df= X_df.join(pd.get_dummies(X_df['BsmtQual'],prefix='BsmtQual'))
X_df = X_df.drop("BsmtQual",1)

#We create the new dimensions for BsmtCond
X_df= X_df.join(pd.get_dummies(X_df['BsmtCond'],prefix='BsmtCond'))
X_df = X_df.drop("BsmtCond",1)

#We create the new dimensions for BsmtExposure
X_df= X_df.join(pd.get_dummies(X_df['BsmtExposure'],prefix='BsmtExposure'))
X_df = X_df.drop("BsmtExposure",1)

#We create the new dimensions for BsmtFinType1
X_df= X_df.join(pd.get_dummies(X_df['BsmtFinType1'],prefix='BsmtFinType1'))
X_df = X_df.drop("BsmtFinType1",1)

#We create the new dimensions for BsmtFinType2
X_df= X_df.join(pd.get_dummies(X_df['BsmtFinType2'],prefix='BsmtFinType2'))
X_df = X_df.drop("BsmtFinType2",1)

#We create the new dimensions for Heating
X_df= X_df.join(pd.get_dummies(X_df['Heating'],prefix='Heating'))
X_df = X_df.drop("Heating",1)

#We create the new dimensions for HeatingQC
X_df= X_df.join(pd.get_dummies(X_df['HeatingQC'],prefix='HeatingQC'))
X_df = X_df.drop("HeatingQC",1)

#We create the new dimensions for CentralAir
X_df= X_df.join(pd.get_dummies(X_df['CentralAir'],prefix='CentralAir'))
X_df = X_df.drop("CentralAir",1)

#We create the new dimensions for Electrical
X_df= X_df.join(pd.get_dummies(X_df['Electrical'],prefix='Electrical'))
X_df = X_df.drop("Electrical",1)

#We create the new dimensions for FireplaceQu
X_df= X_df.join(pd.get_dummies(X_df['FireplaceQu'],prefix='FireplaceQu'))
X_df = X_df.drop("FireplaceQu",1)

#We create the new dimensions for GarageType
X_df= X_df.join(pd.get_dummies(X_df['GarageType'],prefix='GarageType'))
X_df = X_df.drop("GarageType",1)


#We create the new dimensions for GarageFinish
X_df= X_df.join(pd.get_dummies(X_df['GarageFinish'],prefix='GarageFinish'))
X_df = X_df.drop("GarageFinish",1)


#We create the new dimensions for GarageQual
X_df= X_df.join(pd.get_dummies(X_df['GarageQual'],prefix='GarageQual'))
X_df = X_df.drop("GarageQual",1)

#We create the new dimensions for GarageCond
X_df= X_df.join(pd.get_dummies(X_df['GarageCond'],prefix='GarageCond'))
X_df = X_df.drop("GarageCond",1)

#We create the new dimensions for PavedDrive
X_df= X_df.join(pd.get_dummies(X_df['PavedDrive'],prefix='PavedDrive'))
X_df = X_df.drop("PavedDrive",1)

#We create the new dimensions for PoolQC
X_df= X_df.join(pd.get_dummies(X_df['PoolQC'],prefix='PoolQC'))
X_df = X_df.drop("PoolQC",1)

#We create the new dimensions for Fence
X_df= X_df.join(pd.get_dummies(X_df['Fence'],prefix='Fence'))
X_df = X_df.drop("Fence",1)

#We create the new dimensions for MoSold
X_df= X_df.join(pd.get_dummies(X_df['MoSold'],prefix='MoSold'))
X_df = X_df.drop("MoSold",1)

#We create the new dimensions for YrSold
X_df= X_df.join(pd.get_dummies(X_df['YrSold'],prefix='YrSold'))
X_df = X_df.drop("YrSold",1)

#We create the new dimensions for SaleType
X_df= X_df.join(pd.get_dummies(X_df['SaleType'],prefix='SaleType'))
X_df = X_df.drop("SaleType",1)

#We create the new dimensions for Functionnal
X_df= X_df.join(pd.get_dummies(X_df['Functional'],prefix='Functional'))
X_df = X_df.drop("Functional",1)

#We create the new dimensions for KitchenQual
X_df= X_df.join(pd.get_dummies(X_df['KitchenQual'],prefix='KitchenQual'))
X_df = X_df.drop("KitchenQual",1)

#We create the new dimensions for SaleCondition
X_df= X_df.join(pd.get_dummies(X_df['SaleCondition'],prefix='SaleCondition'))
X_df = X_df.drop("SaleCondition",1)


#We create the new dimensions for MiscFeature
X_df= X_df.join(pd.get_dummies(X_df['MiscFeature'],prefix='MiscFeature'))
X_df = X_df.drop("MiscFeature",1)

if 'MiscFeature_Elev' in X_df.columns:
    X_df['MiscFeature_Elev'] = X_df[["MiscFeature_Elev", "MiscVal"]].product(axis=1)
if 'MiscFeature_Gar2' in X_df.columns:
    X_df['MiscFeature_Gar2'] = X_df[["MiscFeature_Gar2", "MiscVal"]].product(axis=1)
if 'MiscFeature_Othr' in X_df.columns:
    X_df['MiscFeature_Othr'] = X_df[["MiscFeature_Othr", "MiscVal"]].product(axis=1)
if 'MiscFeature_Shed' in X_df.columns:
    X_df['MiscFeature_Shed'] = X_df[["MiscFeature_Shed", "MiscVal"]].product(axis=1)
if 'MiscFeature_TenC' in X_df.columns:
    X_df['MiscFeature_TenC'] = X_df[["MiscFeature_TenC", "MiscVal"]].product(axis=1)
if 'MiscFeature_None' in X_df.columns:
    X_df['MiscFeature_None'] = X_df[["MiscFeature_None", "MiscVal"]].product(axis=1)

X_df = X_df.drop("MiscVal",1)