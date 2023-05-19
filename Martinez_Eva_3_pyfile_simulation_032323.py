import os
from time import time
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import MeanShift, estimate_bandwidth
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import math
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from dateutil.relativedelta import relativedelta


def get_period(df_orders_final,weeks):

    date_most_recent_end = df_orders_final['date_formatted'].max()
    date_most_recent_begin = date_most_recent_end - relativedelta(weeks=weeks)

    date_less_recent_end = date_most_recent_begin
    date_less_recent_begin = date_less_recent_end- relativedelta(weeks=weeks)

    period_mr =(date_most_recent_begin,date_most_recent_end)
    period_lr= (date_less_recent_begin,date_less_recent_end)

    return period_lr, period_mr

def get_period_orders(df_orders_final,period):

    df_filtered_period =df_orders_final.loc[df_orders_final["date_formatted"].between(period[0], period[1])]

    return df_filtered_period


def compute_features_df(df_filtered_period):

    # Groupby
    df_recence = df_filtered_period[['date_formatted','customer_unique_id']]
    df_recence_gb = df_recence.groupby('customer_unique_id').max()
    df_recence_gb = df_recence_gb.rename(columns={"date_formatted": "recence_date"})

    df_frequence = df_filtered_period[['date_formatted','customer_unique_id']]
    df_frequence_gb = df_frequence.groupby('customer_unique_id').count()
    df_frequence_gb = df_frequence_gb.rename(columns={"date_formatted": "frequence"})

    df_montant = df_filtered_period[['payment_value','customer_unique_id']]
    df_montant_gb = df_montant.groupby('customer_unique_id').sum()
    df_montant_gb = df_montant_gb.rename(columns={"payment_value": "monetary"})

    df_review = df_filtered_period[['review_score','customer_unique_id']]
    df_review_gb = df_review.groupby('customer_unique_id').mean()

    # Change date to number of days for recence 
    recence_range = df_recence_gb['recence_date'].max()-df_recence_gb['recence_date'].min()

    datetime_relative_list=[]
    for i in df_recence_gb['recence_date']:
        relative_recence = df_recence_gb['recence_date'].max() - i
        relative_recence = relative_recence.days
        datetime_relative_list.append(relative_recence)

    df_recence_gb['recence']= datetime_relative_list
    del df_recence_gb['recence_date']

    # Merge all
    df_rf = df_recence_gb.join(df_frequence_gb)
    df_rfm = df_rf.join(df_montant_gb)
    df_rfmr = df_rfm.join(df_review_gb)
    df_rfmr = df_rfmr.dropna()

    # Refactoring
    # Frequency 
    frequency_refac = df_rfmr['frequence']
    list_frequency =[]
    for i in frequency_refac:
        if i ==1:
            list_frequency.append(0)
        else:
            list_frequency.append(1)
    del df_rfmr['frequence']
    df_rfmr['frequence'] = list_frequency

    # Monetary 
    df_rfmr['monetary_log']=np.log(df_rfmr['monetary'])
    del df_rfmr['monetary']
    df_rfmr.drop(df_rfmr[(df_rfmr['monetary_log'] == -float('inf'))].index, inplace=True)

    df_rfmr.loc[df_rfmr['monetary_log'] > 8, 'monetary_log'] = 8
    df_rfmr.loc[df_rfmr['monetary_log'] < 2.5, 'monetary_log'] = 2.5

    # Recence
    df_rfmr['recence_log']=np.log(df_rfmr['recence'])
    del df_rfmr['recence']
    df_rfmr.drop(df_rfmr[(df_rfmr['recence_log'] == -float('inf'))].index, inplace=True)
    
    df_rfmr_drop_nan = df_rfmr.dropna()
    df_rfmr = df_rfmr_drop_nan


    return df_rfmr


def compute_model(df_rfmr):
    numerical_features = ['monetary_log','recence_log','review_score']

    # Scaler
    scaler = MinMaxScaler()

    # Build the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', scaler, numerical_features)
        ],remainder='passthrough'
    )
    
    kmeans_model = Pipeline([("preprocessor", preprocessor),
                         ("kmeans", KMeans(5))])
    
    kmeans_model.fit(df_rfmr)
    return kmeans_model


def predict_and_score(model_lr,model_mr,df_filtered_period_mr):
    y_ref = model_mr.predict(df_filtered_period_mr)
    y_pred = model_lr.predict(df_filtered_period_mr)
    ari = adjusted_rand_score(y_ref, y_pred)
    return ari

