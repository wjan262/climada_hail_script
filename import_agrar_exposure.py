#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:51:58 2020

@author: jan
"""
import sys
sys.path.append("/home/jan/Documents/ETH/Masterarbeit/climada_python")
import pandas as pd
from climada.entity import Exposures, Entity, LitPop
from climada.entity import ImpactFuncSet, ImpactFunc
from climada.engine import Impact
import matplotlib.pyplot as plt
import_data = True

if import_data:
    path_data = "agrar_exposure/data_arealstatistik/AREA_NOLU04_46_191202.csv"
    df_data = pd.read_csv(path_data)
    

def is_agrar(row):
    """
    

    Parameters
    ----------
    row : dataframe
        DESCRIPTION.

    Returns
    -------
    int
        True for:
            200 Obstbau, Rebbau, Gartenbau 201, 202, 203
            220 Acker- und Futterbau 221, 222, 223
            240 Alpwirtschaft 241, 242, 243

    """    
    if row["LU18_46"] in [201, 202, 221]:#,222,223]:#,241,242,243]:
        return 1
    else:
        return 0

    #LV95 to WGS84
def transform_coord(E, N):
    """
    

    Parameters
    ----------
    E : pandas.core.series.Series
        X-Coordinates in LV95 format.
    N : pandas.core.series.Series
        Y-Coordinates in LV95 format.

    Returns
    -------
    lambd : pandas.core.series.Series
        X-Coordinates in WGS84 format.
    eps : pandas.core.series.Series
        Y-Coordinates in WGS84 format.

    """
    E_star = (E-2600000)/1000000
    N_star = (N-1200000)/1000000
    lambda_star = (2.6779094 + 4.728982 * E_star 
                         + 0.791484 * E_star * N_star
                         + 0.1306 * E_star * N_star**2
                         - 0.0436 * E_star**3)
    
    epsylon_star = (16.9023892 + 3.238272 * N_star 
                         - 0.270978 * E_star**2
                         - 0.002528 * N_star**2
                         - 0.0447 * E_star**2 * N_star
                         - 0.0140 * N_star**3)
    
    lambd = lambda_star * 100 / 36
    eps = epsylon_star * 100 / 36
    return lambd, eps

df_data = df_data[["X", "Y", "LU18_46", "E", "N"]]
df_data["is_agrar"] = df_data.apply(lambda row: is_agrar(row), axis = 1)

#reduce data to only include points specified in is_agrar() function
df_data = df_data[df_data["is_agrar"] == 1]

lambd, eps = transform_coord(E = df_data["E"], N = df_data["N"])


exp_hail_agr = Exposures()
exp_hail_agr["latitude"] = eps
exp_hail_agr["longitude"] = lambd
exp_hail_agr["region_id"] = df_data["LU18_46"]


# Giving value to the different sectors. Source: https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-0704000000_121/px-x-0704000000_121/px-x-0704000000_121.px
# Obst 201 (C.1.1.01.116 Obst): 559'104'082 (Rebbau muss abgezogen werden) -> 351'088'936
# Rebbau 202 (C1.1.01.1162 Weintrauben): 208'015'146
# ackerbau 221 (C1.1.01.11 Pflanzliche Erzeugung): 4'436'181'114 (Obst und Wein (C1.1.01.117 479'824'835))
# = 3955797175

value_obst = 559104082
avg_value_obst = value_obst / exp_hail_agr[exp_hail_agr["region_id"]==201].shape[0]
value_rebbau = 208015146
avg_value_rebbau = value_rebbau / exp_hail_agr[exp_hail_agr["region_id"]==202].shape[0]
value_ackerbau = 3955797175
avg_value_ackerbau = value_ackerbau / exp_hail_agr[exp_hail_agr["region_id"]==221].shape[0]
a = exp_hail_agr[exp_hail_agr["region_id"]==201].assign(value = avg_value_obst)
b = exp_hail_agr[exp_hail_agr["region_id"]==202].assign(value = avg_value_rebbau)
c = exp_hail_agr[exp_hail_agr["region_id"]==221].assign(value = avg_value_ackerbau)

exp_hail_agr = pd.concat([a,b,c]).sort_index()
exp_hail_agr = Exposures(exp_hail_agr)
exp_hail_agr.check()
exp_hail_agr.head()
exp_hail_agr.write_hdf5("exp_hail_agr.hdf5")

test = exp_hail_agr[exp_hail_agr["value"]>0]

plt.scatter(test[test["region_id"].isin([221, 222, 223])]["longitude"], test[test["region_id"].isin([221, 222, 223])]["latitude"])
plt.scatter(test[test["region_id"]==201]["longitude"], test[test["region_id"]==201]["latitude"])
plt.scatter(test[test["region_id"]==202]["longitude"], test[test["region_id"]==202]["latitude"])
plt.legend(labels = ["Acker - Futterbau", "Obstbau", "Rebbau"])
