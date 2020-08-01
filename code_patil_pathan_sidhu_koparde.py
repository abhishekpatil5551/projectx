#!/usr/bin/env python
# coding: utf-8

# In[1]:
#ssds

import numpy as np
import pandas as pd
import plotly
import chart_studio
import chart_studio.plotly as py
import plotly.express as px
import plotly.io as pio


# In[2]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
df_clients = pd.read_csv("Clients.csv")
df_patches = pd.read_csv("Patches.csv")


# In[3]:


df_cl_dum = pd.get_dummies(df_clients[['job','marital','education',
                                         'housing','personal','term']])
df_cl_dum['balance'] = df_clients['balance']
df_cl_dum['age'] = df_clients['age']
corr1 = df_cl_dum.corr()
figc1 = px.imshow(corr1,labels= dict(x="features", y="features", color="Productivity"),
                      x=corr1.columns,
                      y=corr1.columns)
figc1.show()


# In[4]:


# Clustering optimality
er = []
for n in range(1,10):
    kmeans = KMeans(n)
    kmeans.fit(df_cl_dum)
    er.append(kmeans.inertia_)
fig1 = px.line(x = range(1,10), y= er)
fig1.update_layout(
    title_text = "Elbow",
    xaxis = {'title':'Number of Clusters'},
    yaxis = {'title':'Total Inertia'}
)

fig1.show()
fig1.write_html("inertia1.html")


# In[5]:


#found clusters
cl = 2
kmn = KMeans(n_clusters=cl)
kmn.fit(df_cl_dum)
cl_pred = kmn.predict(df_cl_dum)
fig2 = px.scatter(df_clients, x = "age", y= "balance", color = cl_pred)
fig2.update_layout(
    title_text = "Scatter Plot",
    xaxis = {'title':'Age'},
    yaxis = {'title':'Balance'}
)
fig2.show()
fig2.write_html("cluster1.html")


# In[6]:


pix = df_patches[['Elevation',	'Slope',
             'Horizontal_Distance_To_Hydrology',
             'Vertical_Distance_To_Hydrology',
             'Horizontal_Distance_To_Roadways',
             'Horizontal_Distance_To_Fire_Points']]
# finding
corr2 = df_patches.corr()
figc2 = px.imshow(corr2,labels= dict(x="features", y="features", color="Productivity"),
                      x=corr2.columns,
                      y=corr2.columns)
figc2.show()


# In[8]:


#elbow chart
er2 = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(pix)
    er2.append(kmeans.inertia_)
fig3 = px.line(x = range(1,10), y= er2)
fig3.update_layout(
    title_text = "Elbow",
    xaxis = {'title':'Number of Clusters'},
    yaxis = {'title':'Total Inertia'}
)
fig3.show()
fig3.write_html("intertia2.html")


# In[9]:


#clusters of patches
clust_p = 2
kmen = KMeans(n_clusters=clust_p)
kmen.fit(pix)
patch_pr = kmen.predict(pix)
fig4 = px.scatter(pix, x= "Horizontal_Distance_To_Roadways",y="Horizontal_Distance_To_Fire_Points", color=patch_pr)
fig4.update_layout(
    title_text = "ScatterPlot",
    xaxis = {'title':'Horizontal Distance To Roadways'},
    yaxis = {'title':'Horizontal Distance To Fire Points'}
)
fig4.show()
fig4.write_html("cluster2.html")


# In[ ]:




