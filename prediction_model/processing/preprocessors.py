#Import libraries
import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# Import other files/modules 
from prediction_model.config import config

# Numeric Imputer
class NumericalImputer(BaseEstimator,TransformerMixin):
    "Numerical Data Missing Value Imputer"
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.imputer_dict = {}
        for feature in self.variables:
            self.imputer_dict[feature] = X[feature].mean()
        return self
    
    def transform(self,X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict[feature],inplace=True)
        return X
    
    
#Categorical Imputer
class CategoricalImputer(BaseEstimator,TransformerMixin):
    """Categorical Data Missing Value Imputer"""
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X,y=None):
        self.imputer_dict_={}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X=X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature],inplace=True)
        return X
    
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Categorical Data Encoder"""
    
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y):
        self.encoder_dict_ = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """Feature Engineering"""
    
    def __init__(self, variables=None, reference_variable=None):
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        # No need to put anything, needed for Sklearn Pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var] + X[self.reference_variable]
        return X


class LogTransformation(BaseEstimator, TransformerMixin):
    """Transforming variables using Log Transformations"""
    
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y):
        return self
    #Need to check in advance if the features are <= 0
    #If yes, needs to be transformed properly (E.g., np.log1p(X[var]))
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.log(X[var])
        return X


class DropFeatures(BaseEstimator, TransformerMixin):
    """Dropping Features Which Are Less Significant"""
    
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables_to_drop, axis=1)
        return X