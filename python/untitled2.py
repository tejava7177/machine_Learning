# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:53:29 2024

@author: user
"""

import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')
fig.show(renderer="browser")