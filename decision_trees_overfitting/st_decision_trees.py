# -*- coding: utf-8 -*-
#  File: st_decision_trees.py
#  Project: 'Decision Trees'
#  Created by Gennady Matveev (gm@og.ly) on 16-12-2021.
#  Copyright 2021. All rights reserved.

# Import libraries
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


st.set_page_config(page_title='Decision trees overfitting demo', page_icon='../images/head.ico',
                   layout='centered', initial_sidebar_state='collapsed')

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
    
st.image('../images/head.png')
st.subheader('Decision trees overfitting demo')

# Set function
def f(x):
    return np.sin(x)

# Set sliders
random_state = st.slider('Random seed', min_value=1, max_value=42, value=17)
depth = st.slider('Decision tree depth', min_value=2, max_value=10, value=3)

st.markdown("""---""")

# Generate data
rng = np.random.RandomState(random_state)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = f(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))   

# Generate test smaple
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

# Create a model
reg = DecisionTreeRegressor(max_depth=depth)

# Fit the model
reg.fit(X, y)

# Make predictions
y_hat = reg.predict(X_test)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

plt.plot(X_test, y_hat, label=f'depth={depth}', color='g', linewidth=2)
plt.scatter(X, y, label='data', edgecolor="black", c="darkorange")

plt.xticks(size=12)
plt.xlabel("X", size=12)
plt.yticks(size=12)
plt.ylabel("y", size=12)
plt.legend(loc='best', fontsize=12)
plt.grid(b=1)

st.pyplot(fig)

# Show this code
show_me = st.checkbox('Show my code')
if show_me:
    st.code("""
# -*- coding: utf-8 -*-
#  File: st_decision_trees.py
#  Project: 'Decision Trees'
#  Created by Gennady Matveev (gm@og.ly) on 16-12-2021.
#  Copyright 2021. All rights reserved.

# Import libraries
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


st.set_page_config(page_title='Decision tree overfitting demo', page_icon='../ICONS/head.ico',
                   layout='centered', initial_sidebar_state='collapsed')

# Set function
def f(x):
    return np.sin(x)

# Set sliders
random_state = st.slider('Random seed', min_value=1, max_value=42, value=17)
depth = st.slider('Decision tree depth', min_value=2, max_value=10, value=3)

# Generate data
rng = np.random.RandomState(random_state)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = f(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))   

# Generate test smaple
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

# Create a model
reg = DecisionTreeRegressor(max_depth=depth)

# Fit the model
reg.fit(X, y)

# Make predictions
y_hat = reg.predict(X_test)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
# plt.figure(figsize=(10, 5))

plt.plot(X_test, y_hat, label=f'depth={depth}', color='g', linewidth=2)
plt.scatter(X, y, label='data', edgecolor="black", c="darkorange")#, label="data"

plt.xticks(size=12)
plt.xlabel("X", size=12)
plt.yticks(size=12)
plt.ylabel("y", size=12)
plt.legend(loc='best', fontsize=12)
plt.grid(b=1)

st.pyplot(fig)
"""
)