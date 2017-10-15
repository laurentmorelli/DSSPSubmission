import pandas as pd
import os
import math
import numpy as np
import random
from statsmodels.robust.scale import mad

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        
        pd.options.mode.chained_assignment = None  # default='warn'

       
        #####
        ## CLEAN MISSING VALUES BEGIN
        
        #LotFrontage	True
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

        #Just to be sure we're not dumb
        if 'SalePrice' in X_df.columns:
            X_df = X_df.drop('SalePrice',1)


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
        
        #####
        ## DROP UNECESSARY COLUMNS

        #we drop the year of built of the garage
        X_df = X_df.drop("GarageYrBlt",1)

        #we drop the id 
        X_df = X_df.drop("Id",1)

        ####
        ## ADD FEATURE

        #Let's create a columns for the age 
        X_df['AgeBuilt'] = 2010 - X_df['YearBuilt']
        X_df['AgeSold'] = 2010 - X_df['YrSold']
        #X_df['AgeBuilt'] = X_df['AgeSold'] - X_df['AgeBuilt']

        #Let's create heuristic columns
        X_df['LotShapeScore'] = X_df['LotShape'].replace({'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1})
        X_df['LandSlopeScore'] = X_df['LandSlope'].replace({'Gtl': 3, 'Mod': 2, 'Sev': 1})
        X_df['ExterQualScore'] = X_df['ExterQual'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1})
        X_df['ExterCondScore'] = X_df['ExterCond'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1})
        X_df['BsmtQualScore'] = X_df['BsmtQual'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1, 'None':0})
        X_df['BsmtCondScore'] = X_df['BsmtCond'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1, 'None':0})
        X_df['BsmtExposureScore'] = X_df['BsmtExposure'].replace({'Gd': 4,'Av': 3, 'Mn': 2, 'No': 1, 'None':0})
        X_df['HeatingQCScore'] = X_df['HeatingQC'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1})
        X_df['KitchenQualScore'] = X_df['KitchenQual'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1, 'UKN': 0})
        X_df['GarageQualScore'] = X_df['GarageQual'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1, 'None':0})
        X_df['GarageCondScore'] = X_df['GarageCond'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1, 'None':0})
        X_df['PoolQCScore'] = X_df['PoolQC'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1, 'None':0})
        X_df['FireplaceQuScore'] = X_df['FireplaceQu'].replace({'Ex': 5,'Gd': 4,'TA': 3, 'Fa': 2, 'Po': 1, 'None':0})
        X_df['FenceScore'] = X_df['Fence'].replace({'GdPrv': 5,'MnPrv': 4,'GdWo': 3, 'MnWw': 2, 'None': 1})
        X_df['BsmtFinType1Score'] = X_df['BsmtFinType1'].replace({'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None':0})
        X_df['BsmtFinType2Score'] = X_df['BsmtFinType2'].replace({'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None':0})
        X_df['FunctionalScore'] = X_df['Functional'].replace({'Typ': 7,'Min1': 6,'Min2': 5,'Mod': 4,'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal':0})
		
        #Let's lolilog the data
        #X_df['GrLivArea'] = np.log1p(X_df['GrLivArea']) -> no use with gradient boosting

        #####
        ## CREATE AND KEEP NUMERIC DIMENSIONS

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

        ####
        # ADD POTENTIAL NULL COLUMNS
        columns_expected = [
             'LotShapeScore',
            'LandSlopeScore',
            'ExterQualScore',
            'ExterCondScore',
            'BsmtQualScore',
            'BsmtCondScore',
            'BsmtCondScore',
            'HeatingQCScore',
            'KitchenQualScore',
            'GarageQualScore',
            'GarageCondScore',
            'PoolQCScore',
            'FireplaceQuScore',
            'FenceScore',
            'BsmtFinType1Score',
            'BsmtFinType2Score',
            'FunctionalScore' 
            'AgeBuilt',
            'AgeSold',
            'LotFrontage',
            'LotArea',
            'OverallQual',
            'OverallCond',
            'BsmtFinSF1',
            'BsmtFinSF2',
            'BsmtUnfSF',
            'TotalBsmtSF',
            '1stFlrSF',
            '2ndFlrSF',
            'LowQualFinSF',
            'GrLivArea',
            'BsmtFullBath',
            'BsmtHalfBath',
            'FullBath',
            'HalfBath',
            'BedroomAbvGr',
            'KitchenAbvGr',
            'TotRmsAbvGrd',
            'Fireplaces',
            'GarageCars',
            'GarageArea',
            'WoodDeckSF',
            'OpenPorchSF',
            'EnclosedPorch',
            '3SsnPorch',
            'ScreenPorch',
            'PoolArea',
            'MiscVal',
            'MSSubClass_20',
            'MSSubClass_30',
            'MSSubClass_40',
            'MSSubClass_45',
            'MSSubClass_50',
            'MSSubClass_60',
            'MSSubClass_70',
            'MSSubClass_75',
            'MSSubClass_80',
            'MSSubClass_85',
            'MSSubClass_90',
            'MSSubClass_120',
            'MSSubClass_150',
            'MSSubClass_160',
            'MSSubClass_180',
            'MSSubClass_190',
            'MSZoning_C (all)',
            'MSZoning_FV',
            'MSZoning_None',
            'MSZoning_RH',
            'MSZoning_RL',
            'MSZoning_RM',
            'Street_Grvl',
            'Street_Pave',
            'Alley_Grvl',
            'Alley_None',
            'Alley_Pave',
            'LotShape_IR1',
            'LotShape_IR2',
            'LotShape_IR3',
            'LotShape_Reg',
            'LandContour_Bnk',
            'LandContour_HLS',
            'LandContour_Low',
            'LandContour_Lvl',
            'Utilities_AllPub',
            'LotConfig_Corner',
            'LotConfig_CulDSac',
            'LotConfig_FR2',
            'LotConfig_FR3',
            'LotConfig_Inside',
            'LandSlope_Gtl',
            'LandSlope_Mod',
            'LandSlope_Sev',
            'Neighborhood_Blmngtn',
            'Neighborhood_Blueste',
            'Neighborhood_BrDale',
            'Neighborhood_BrkSide',
            'Neighborhood_ClearCr',
            'Neighborhood_CollgCr',
            'Neighborhood_Crawfor',
            'Neighborhood_Edwards',
            'Neighborhood_Gilbert',
            'Neighborhood_IDOTRR',
            'Neighborhood_MeadowV',
            'Neighborhood_Mitchel',
            'Neighborhood_NAmes',
            'Neighborhood_NPkVill',
            'Neighborhood_NWAmes',
            'Neighborhood_NoRidge',
            'Neighborhood_NridgHt',
            'Neighborhood_OldTown',
            'Neighborhood_SWISU',
            'Neighborhood_Sawyer',
            'Neighborhood_SawyerW',
            'Neighborhood_Somerst',
            'Neighborhood_StoneBr',
            'Neighborhood_Timber',
            'Neighborhood_Veenker',
            'Condition1_Artery',
            'Condition1_Feedr',
            'Condition1_Norm',
            'Condition1_PosA',
            'Condition1_PosN',
            'Condition1_RRAe',
            'Condition1_RRAn',
            'Condition1_RRNe',
            'Condition1_RRNn',
            'Condition2_Artery',
            'Condition2_Feedr',
            'Condition2_Norm',
            'Condition2_PosA',
            'Condition2_PosN',
            'BldgType_1Fam',
            'BldgType_2fmCon',
            'BldgType_Duplex',
            'BldgType_Twnhs',
            'BldgType_TwnhsE',
            'HouseStyle_1.5Fin',
            'HouseStyle_1.5Unf',
            'HouseStyle_1Story',
            'HouseStyle_2.5Unf',
            'HouseStyle_2Story',
            'HouseStyle_SFoyer',
            'HouseStyle_SLvl',
            'YearBuilt_1879',
            'YearBuilt_1880',
            'YearBuilt_1890',
            'YearBuilt_1895',
            'YearBuilt_1896',
            'YearBuilt_1900',
            'YearBuilt_1901',
            'YearBuilt_1902',
            'YearBuilt_1905',
            'YearBuilt_1907',
            'YearBuilt_1910',
            'YearBuilt_1912',
            'YearBuilt_1914',
            'YearBuilt_1915',
            'YearBuilt_1916',
            'YearBuilt_1917',
            'YearBuilt_1918',
            'YearBuilt_1919',
            'YearBuilt_1920',
            'YearBuilt_1921',
            'YearBuilt_1922',
            'YearBuilt_1923',
            'YearBuilt_1924',
            'YearBuilt_1925',
            'YearBuilt_1926',
            'YearBuilt_1927',
            'YearBuilt_1928',
            'YearBuilt_1929',
            'YearBuilt_1930',
            'YearBuilt_1931',
            'YearBuilt_1932',
            'YearBuilt_1934',
            'YearBuilt_1935',
            'YearBuilt_1936',
            'YearBuilt_1937',
            'YearBuilt_1938',
            'YearBuilt_1939',
            'YearBuilt_1940',
            'YearBuilt_1941',
            'YearBuilt_1942',
            'YearBuilt_1945',
            'YearBuilt_1946',
            'YearBuilt_1947',
            'YearBuilt_1948',
            'YearBuilt_1949',
            'YearBuilt_1950',
            'YearBuilt_1951',
            'YearBuilt_1952',
            'YearBuilt_1953',
            'YearBuilt_1954',
            'YearBuilt_1955',
            'YearBuilt_1956',
            'YearBuilt_1957',
            'YearBuilt_1958',
            'YearBuilt_1959',
            'YearBuilt_1960',
            'YearBuilt_1961',
            'YearBuilt_1962',
            'YearBuilt_1963',
            'YearBuilt_1964',
            'YearBuilt_1965',
            'YearBuilt_1966',
            'YearBuilt_1967',
            'YearBuilt_1968',
            'YearBuilt_1969',
            'YearBuilt_1970',
            'YearBuilt_1971',
            'YearBuilt_1972',
            'YearBuilt_1973',
            'YearBuilt_1974',
            'YearBuilt_1975',
            'YearBuilt_1976',
            'YearBuilt_1977',
            'YearBuilt_1978',
            'YearBuilt_1979',
            'YearBuilt_1980',
            'YearBuilt_1981',
            'YearBuilt_1982',
            'YearBuilt_1983',
            'YearBuilt_1984',
            'YearBuilt_1985',
            'YearBuilt_1986',
            'YearBuilt_1987',
            'YearBuilt_1988',
            'YearBuilt_1989',
            'YearBuilt_1990',
            'YearBuilt_1991',
            'YearBuilt_1992',
            'YearBuilt_1993',
            'YearBuilt_1994',
            'YearBuilt_1995',
            'YearBuilt_1996',
            'YearBuilt_1997',
            'YearBuilt_1998',
            'YearBuilt_1999',
            'YearBuilt_2000',
            'YearBuilt_2001',
            'YearBuilt_2002',
            'YearBuilt_2003',
            'YearBuilt_2004',
            'YearBuilt_2005',
            'YearBuilt_2006',
            'YearBuilt_2007',
            'YearBuilt_2008',
            'YearBuilt_2009',
            'YearBuilt_2010',
            'YearRemodAdd_1950',
            'YearRemodAdd_1951',
            'YearRemodAdd_1952',
            'YearRemodAdd_1953',
            'YearRemodAdd_1954',
            'YearRemodAdd_1955',
            'YearRemodAdd_1956',
            'YearRemodAdd_1957',
            'YearRemodAdd_1958',
            'YearRemodAdd_1959',
            'YearRemodAdd_1960',
            'YearRemodAdd_1961',
            'YearRemodAdd_1962',
            'YearRemodAdd_1963',
            'YearRemodAdd_1964',
            'YearRemodAdd_1965',
            'YearRemodAdd_1966',
            'YearRemodAdd_1967',
            'YearRemodAdd_1968',
            'YearRemodAdd_1969',
            'YearRemodAdd_1970',
            'YearRemodAdd_1971',
            'YearRemodAdd_1972',
            'YearRemodAdd_1973',
            'YearRemodAdd_1974',
            'YearRemodAdd_1975',
            'YearRemodAdd_1976',
            'YearRemodAdd_1977',
            'YearRemodAdd_1978',
            'YearRemodAdd_1979',
            'YearRemodAdd_1980',
            'YearRemodAdd_1981',
            'YearRemodAdd_1982',
            'YearRemodAdd_1983',
            'YearRemodAdd_1984',
            'YearRemodAdd_1985',
            'YearRemodAdd_1986',
            'YearRemodAdd_1987',
            'YearRemodAdd_1988',
            'YearRemodAdd_1989',
            'YearRemodAdd_1990',
            'YearRemodAdd_1991',
            'YearRemodAdd_1992',
            'YearRemodAdd_1993',
            'YearRemodAdd_1994',
            'YearRemodAdd_1995',
            'YearRemodAdd_1996',
            'YearRemodAdd_1997',
            'YearRemodAdd_1998',
            'YearRemodAdd_1999',
            'YearRemodAdd_2000',
            'YearRemodAdd_2001',
            'YearRemodAdd_2002',
            'YearRemodAdd_2003',
            'YearRemodAdd_2004',
            'YearRemodAdd_2005',
            'YearRemodAdd_2006',
            'YearRemodAdd_2007',
            'YearRemodAdd_2008',
            'YearRemodAdd_2009',
            'YearRemodAdd_2010',
            'RoofStyle_Flat',
            'RoofStyle_Gable',
            'RoofStyle_Gambrel',
            'RoofStyle_Hip',
            'RoofStyle_Mansard',
            'RoofStyle_Shed',
            'RoofMatl_CompShg',
            'RoofMatl_Tar&Grv',
            'RoofMatl_WdShake',
            'RoofMatl_WdShngl',
            'Exterior1st_AsbShng',
            'Exterior1st_AsphShn',
            'Exterior1st_BrkComm',
            'Exterior1st_BrkFace',
            'Exterior1st_CBlock',
            'Exterior1st_CemntBd',
            'Exterior1st_HdBoard',
            'Exterior1st_MetalSd',
            'Exterior1st_None',
            'Exterior1st_Plywood',
            'Exterior1st_Stucco',
            'Exterior1st_VinylSd',
            'Exterior1st_Wd Sdng',
            'Exterior1st_WdShing',
            'Exterior2nd_AsbShng',
            'Exterior2nd_AsphShn',
            'Exterior2nd_Brk Cmn',
            'Exterior2nd_BrkFace',
            'Exterior2nd_CBlock',
            'Exterior2nd_CmentBd',
            'Exterior2nd_HdBoard',
            'Exterior2nd_ImStucc',
            'Exterior2nd_MetalSd',
            'Exterior2nd_None',
            'Exterior2nd_Plywood',
            'Exterior2nd_Stone',
            'Exterior2nd_Stucco',
            'Exterior2nd_VinylSd',
            'Exterior2nd_Wd Sdng',
            'Exterior2nd_Wd Shng',
            'MasVnrType_BrkCmn',
            'MasVnrType_BrkFace',
            'MasVnrType_None',
            'MasVnrType_Stone',
            'MasVnrArea_0.0',
            'MasVnrArea_1.0',
            'MasVnrArea_3.0',
            'MasVnrArea_14.0',
            'MasVnrArea_16.0',
            'MasVnrArea_18.0',
            'MasVnrArea_20.0',
            'MasVnrArea_22.0',
            'MasVnrArea_23.0',
            'MasVnrArea_24.0',
            'MasVnrArea_28.0',
            'MasVnrArea_30.0',
            'MasVnrArea_32.0',
            'MasVnrArea_36.0',
            'MasVnrArea_38.0',
            'MasVnrArea_39.0',
            'MasVnrArea_40.0',
            'MasVnrArea_41.0',
            'MasVnrArea_44.0',
            'MasVnrArea_45.0',
            'MasVnrArea_47.0',
            'MasVnrArea_50.0',
            'MasVnrArea_51.0',
            'MasVnrArea_52.0',
            'MasVnrArea_53.0',
            'MasVnrArea_54.0',
            'MasVnrArea_56.0',
            'MasVnrArea_58.0',
            'MasVnrArea_60.0',
            'MasVnrArea_62.0',
            'MasVnrArea_65.0',
            'MasVnrArea_67.0',
            'MasVnrArea_68.0',
            'MasVnrArea_69.0',
            'MasVnrArea_70.0',
            'MasVnrArea_72.0',
            'MasVnrArea_74.0',
            'MasVnrArea_76.0',
            'MasVnrArea_80.0',
            'MasVnrArea_82.0',
            'MasVnrArea_84.0',
            'MasVnrArea_85.0',
            'MasVnrArea_86.0',
            'MasVnrArea_87.0',
            'MasVnrArea_88.0',
            'MasVnrArea_89.0',
            'MasVnrArea_90.0',
            'MasVnrArea_91.0',
            'MasVnrArea_94.0',
            'MasVnrArea_95.0',
            'MasVnrArea_96.0',
            'MasVnrArea_98.0',
            'MasVnrArea_99.0',
            'MasVnrArea_100.0',
            'MasVnrArea_101.0',
            'MasVnrArea_102.0',
            'MasVnrArea_104.0',
            'MasVnrArea_106.0',
            'MasVnrArea_108.0',
            'MasVnrArea_112.0',
            'MasVnrArea_113.0',
            'MasVnrArea_114.0',
            'MasVnrArea_115.0',
            'MasVnrArea_118.0',
            'MasVnrArea_119.0',
            'MasVnrArea_120.0',
            'MasVnrArea_121.0',
            'MasVnrArea_122.0',
            'MasVnrArea_123.0',
            'MasVnrArea_124.0',
            'MasVnrArea_125.0',
            'MasVnrArea_126.0',
            'MasVnrArea_128.0',
            'MasVnrArea_130.0',
            'MasVnrArea_132.0',
            'MasVnrArea_134.0',
            'MasVnrArea_135.0',
            'MasVnrArea_136.0',
            'MasVnrArea_138.0',
            'MasVnrArea_140.0',
            'MasVnrArea_141.0',
            'MasVnrArea_142.0',
            'MasVnrArea_143.0',
            'MasVnrArea_144.0',
            'MasVnrArea_145.0',
            'MasVnrArea_146.0',
            'MasVnrArea_148.0',
            'MasVnrArea_149.0',
            'MasVnrArea_150.0',
            'MasVnrArea_153.0',
            'MasVnrArea_156.0',
            'MasVnrArea_157.0',
            'MasVnrArea_158.0',
            'MasVnrArea_160.0',
            'MasVnrArea_161.0',
            'MasVnrArea_162.0',
            'MasVnrArea_163.0',
            'MasVnrArea_164.0',
            'MasVnrArea_165.0',
            'MasVnrArea_166.0',
            'MasVnrArea_168.0',
            'MasVnrArea_170.0',
            'MasVnrArea_172.0',
            'MasVnrArea_174.0',
            'MasVnrArea_176.0',
            'MasVnrArea_177.0',
            'MasVnrArea_178.0',
            'MasVnrArea_179.0',
            'MasVnrArea_180.0',
            'MasVnrArea_182.0',
            'MasVnrArea_184.0',
            'MasVnrArea_186.0',
            'MasVnrArea_187.0',
            'MasVnrArea_188.0',
            'MasVnrArea_189.0',
            'MasVnrArea_190.0',
            'MasVnrArea_192.0',
            'MasVnrArea_194.0',
            'MasVnrArea_196.0',
            'MasVnrArea_197.0',
            'MasVnrArea_198.0',
            'MasVnrArea_199.0',
            'MasVnrArea_200.0',
            'MasVnrArea_202.0',
            'MasVnrArea_203.0',
            'MasVnrArea_204.0',
            'MasVnrArea_205.0',
            'MasVnrArea_206.0',
            'MasVnrArea_209.0',
            'MasVnrArea_210.0',
            'MasVnrArea_212.0',
            'MasVnrArea_214.0',
            'MasVnrArea_215.0',
            'MasVnrArea_216.0',
            'MasVnrArea_217.0',
            'MasVnrArea_218.0',
            'MasVnrArea_221.0',
            'MasVnrArea_222.0',
            'MasVnrArea_226.0',
            'MasVnrArea_227.0',
            'MasVnrArea_228.0',
            'MasVnrArea_229.0',
            'MasVnrArea_230.0',
            'MasVnrArea_232.0',
            'MasVnrArea_234.0',
            'MasVnrArea_235.0',
            'MasVnrArea_236.0',
            'MasVnrArea_238.0',
            'MasVnrArea_240.0',
            'MasVnrArea_242.0',
            'MasVnrArea_244.0',
            'MasVnrArea_246.0',
            'MasVnrArea_248.0',
            'MasVnrArea_250.0',
            'MasVnrArea_251.0',
            'MasVnrArea_252.0',
            'MasVnrArea_253.0',
            'MasVnrArea_254.0',
            'MasVnrArea_256.0',
            'MasVnrArea_257.0',
            'MasVnrArea_258.0',
            'MasVnrArea_259.0',
            'MasVnrArea_260.0',
            'MasVnrArea_261.0',
            'MasVnrArea_264.0',
            'MasVnrArea_265.0',
            'MasVnrArea_268.0',
            'MasVnrArea_270.0',
            'MasVnrArea_272.0',
            'MasVnrArea_275.0',
            'MasVnrArea_276.0',
            'MasVnrArea_278.0',
            'MasVnrArea_279.0',
            'MasVnrArea_280.0',
            'MasVnrArea_283.0',
            'MasVnrArea_284.0',
            'MasVnrArea_285.0',
            'MasVnrArea_286.0',
            'MasVnrArea_288.0',
            'MasVnrArea_289.0',
            'MasVnrArea_290.0',
            'MasVnrArea_291.0',
            'MasVnrArea_292.0',
            'MasVnrArea_294.0',
            'MasVnrArea_295.0',
            'MasVnrArea_296.0',
            'MasVnrArea_298.0',
            'MasVnrArea_300.0',
            'MasVnrArea_302.0',
            'MasVnrArea_304.0',
            'MasVnrArea_305.0',
            'MasVnrArea_306.0',
            'MasVnrArea_308.0',
            'MasVnrArea_309.0',
            'MasVnrArea_310.0',
            'MasVnrArea_320.0',
            'MasVnrArea_322.0',
            'MasVnrArea_323.0',
            'MasVnrArea_327.0',
            'MasVnrArea_332.0',
            'MasVnrArea_340.0',
            'MasVnrArea_342.0',
            'MasVnrArea_352.0',
            'MasVnrArea_353.0',
            'MasVnrArea_355.0',
            'MasVnrArea_356.0',
            'MasVnrArea_359.0',
            'MasVnrArea_360.0',
            'MasVnrArea_364.0',
            'MasVnrArea_365.0',
            'MasVnrArea_366.0',
            'MasVnrArea_368.0',
            'MasVnrArea_371.0',
            'MasVnrArea_372.0',
            'MasVnrArea_378.0',
            'MasVnrArea_379.0',
            'MasVnrArea_380.0',
            'MasVnrArea_382.0',
            'MasVnrArea_383.0',
            'MasVnrArea_385.0',
            'MasVnrArea_394.0',
            'MasVnrArea_397.0',
            'MasVnrArea_400.0',
            'MasVnrArea_402.0',
            'MasVnrArea_405.0',
            'MasVnrArea_406.0',
            'MasVnrArea_410.0',
            'MasVnrArea_418.0',
            'MasVnrArea_420.0',
            'MasVnrArea_422.0',
            'MasVnrArea_423.0',
            'MasVnrArea_425.0',
            'MasVnrArea_430.0',
            'MasVnrArea_432.0',
            'MasVnrArea_434.0',
            'MasVnrArea_440.0',
            'MasVnrArea_442.0',
            'MasVnrArea_444.0',
            'MasVnrArea_450.0',
            'MasVnrArea_456.0',
            'MasVnrArea_466.0',
            'MasVnrArea_468.0',
            'MasVnrArea_470.0',
            'MasVnrArea_472.0',
            'MasVnrArea_473.0',
            'MasVnrArea_480.0',
            'MasVnrArea_492.0',
            'MasVnrArea_495.0',
            'MasVnrArea_500.0',
            'MasVnrArea_501.0',
            'MasVnrArea_502.0',
            'MasVnrArea_504.0',
            'MasVnrArea_506.0',
            'MasVnrArea_509.0',
            'MasVnrArea_510.0',
            'MasVnrArea_513.0',
            'MasVnrArea_514.0',
            'MasVnrArea_515.0',
            'MasVnrArea_518.0',
            'MasVnrArea_519.0',
            'MasVnrArea_522.0',
            'MasVnrArea_525.0',
            'MasVnrArea_526.0',
            'MasVnrArea_532.0',
            'MasVnrArea_549.0',
            'MasVnrArea_550.0',
            'MasVnrArea_554.0',
            'MasVnrArea_567.0',
            'MasVnrArea_568.0',
            'MasVnrArea_572.0',
            'MasVnrArea_600.0',
            'MasVnrArea_615.0',
            'MasVnrArea_621.0',
            'MasVnrArea_632.0',
            'MasVnrArea_634.0',
            'MasVnrArea_647.0',
            'MasVnrArea_652.0',
            'MasVnrArea_657.0',
            'MasVnrArea_662.0',
            'MasVnrArea_668.0',
            'MasVnrArea_674.0',
            'MasVnrArea_680.0',
            'MasVnrArea_692.0',
            'MasVnrArea_710.0',
            'MasVnrArea_714.0',
            'MasVnrArea_724.0',
            'MasVnrArea_726.0',
            'MasVnrArea_730.0',
            'MasVnrArea_734.0',
            'MasVnrArea_738.0',
            'MasVnrArea_754.0',
            'MasVnrArea_771.0',
            'MasVnrArea_877.0',
            'MasVnrArea_886.0',
            'MasVnrArea_902.0',
            'MasVnrArea_945.0',
            'MasVnrArea_970.0',
            'MasVnrArea_1050.0',
            'MasVnrArea_1095.0',
            'MasVnrArea_1110.0',
            'MasVnrArea_1159.0',
            'MasVnrArea_1224.0',
            'MasVnrArea_1290.0',
            'ExterQual_Ex',
            'ExterQual_Fa',
            'ExterQual_Gd',
            'ExterQual_TA',
            'ExterCond_Ex',
            'ExterCond_Fa',
            'ExterCond_Gd',
            'ExterCond_Po',
            'ExterCond_TA',
            'Foundation_BrkTil',
            'Foundation_CBlock',
            'Foundation_PConc',
            'Foundation_Slab',
            'Foundation_Stone',
            'Foundation_Wood',
            'BsmtQual_Ex',
            'BsmtQual_Fa',
            'BsmtQual_Gd',
            'BsmtQual_None',
            'BsmtQual_TA',
            'BsmtCond_Fa',
            'BsmtCond_Gd',
            'BsmtCond_None',
            'BsmtCond_Po',
            'BsmtCond_TA',
            'BsmtExposure_Av',
            'BsmtExposure_Gd',
            'BsmtExposure_Mn',
            'BsmtExposure_No',
            'BsmtExposure_None',
            'BsmtFinType1_ALQ',
            'BsmtFinType1_BLQ',
            'BsmtFinType1_GLQ',
            'BsmtFinType1_LwQ',
            'BsmtFinType1_None',
            'BsmtFinType1_Rec',
            'BsmtFinType1_Unf',
            'BsmtFinType2_ALQ',
            'BsmtFinType2_BLQ',
            'BsmtFinType2_GLQ',
            'BsmtFinType2_LwQ',
            'BsmtFinType2_None',
            'BsmtFinType2_Rec',
            'BsmtFinType2_Unf',
            'Heating_GasA',
            'Heating_GasW',
            'Heating_Grav',
            'Heating_Wall',
            'HeatingQC_Ex',
            'HeatingQC_Fa',
            'HeatingQC_Gd',
            'HeatingQC_Po',
            'HeatingQC_TA',
            'CentralAir_N',
            'CentralAir_Y',
            'Electrical_FuseA',
            'Electrical_FuseF',
            'Electrical_FuseP',
            'Electrical_SBrkr',
            'FireplaceQu_Ex',
            'FireplaceQu_Fa',
            'FireplaceQu_Gd',
            'FireplaceQu_None',
            'FireplaceQu_Po',
            'FireplaceQu_TA',
            'GarageType_2Types',
            'GarageType_Attchd',
            'GarageType_Basment',
            'GarageType_BuiltIn',
            'GarageType_CarPort',
            'GarageType_Detchd',
            'GarageType_None',
            'GarageFinish_Fin',
            'GarageFinish_None',
            'GarageFinish_RFn',
            'GarageFinish_Unf',
            'GarageQual_Fa',
            'GarageQual_Gd',
            'GarageQual_None',
            'GarageQual_Po',
            'GarageQual_TA',
            'GarageCond_Ex',
            'GarageCond_Fa',
            'GarageCond_Gd',
            'GarageCond_None',
            'GarageCond_Po',
            'GarageCond_TA',
            'PavedDrive_N',
            'PavedDrive_P',
            'PavedDrive_Y',
            'PoolQC_Ex',
            'PoolQC_Gd',
            'PoolQC_None',
            'Fence_GdPrv',
            'Fence_GdWo',
            'Fence_MnPrv',
            'Fence_MnWw',
            'Fence_None',
            'MiscFeature_Gar2',
            'MiscFeature_None',
            'MiscFeature_Othr',
            'MiscFeature_Shed',
            'MoSold_1',
            'MoSold_2',
            'MoSold_3',
            'MoSold_4',
            'MoSold_5',
            'MoSold_6',
            'MoSold_7',
            'MoSold_8',
            'MoSold_9',
            'MoSold_10',
            'MoSold_11',
            'MoSold_12',
            'YrSold_2006',
            'YrSold_2007',
            'YrSold_2008',
            'YrSold_2009',
            'YrSold_2010',
            'SaleType_COD',
            'SaleType_CWD',
            'SaleType_Con',
            'SaleType_ConLD',
            'SaleType_ConLI',
            'SaleType_ConLw',
            'SaleType_New',
            'SaleType_Oth',
            'SaleType_WD',
            'Functional_Maj1',
            'Functional_Maj2',
            'Functional_Min1',
            'Functional_Min2',
            'Functional_Mod',
            'Functional_Sev',
            'Functional_Typ',
            'KitchenQual_Ex',
            'KitchenQual_Fa',
            'KitchenQual_Gd',
            'KitchenQual_TA',
            'KitchenQual_UKN',
            'SaleCondition_Abnorml',
            'SaleCondition_AdjLand',
            'SaleCondition_Alloca',
            'SaleCondition_Family',
            'SaleCondition_Normal',
            'SaleCondition_Partial']

        for col in columns_expected:
            if col not in X_df.columns:
                X_df[col] = 0

        #CLEARLY THAT'S DUMB !!!!!!

        for col in X_df.columns:
            if col not in columns_expected:
                X_df = X_df.drop(col,1)

        #Normalizing remaining data
        X_df['GrLivArea'] = np.log1p(X_df['GrLivArea'])

        X_df["TotalSF"] = np.log1p(X_df["TotalBsmtSF"] + X_df["1stFlrSF"] + X_df["2ndFlrSF"])

        X_array = X_df.values
        return X_array
