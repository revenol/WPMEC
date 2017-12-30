# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 19:32:44 2017

@author: Administrator
"""

from sklearn import datasets
iris = datasets.load_iris()
print(len(iris.data))