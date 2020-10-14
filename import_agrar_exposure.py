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
    #200 Obstbau, Rebbau, Gartenbau 201, 202, 203
    #220 Acker- und Futterbau 221, 222, 223
    #240 Alpwirtschaft 241, 242, 243
    
    if row["LU18_46"] in [201,202,203,221,222,223,241,242,243]:
        return 1
    else:
        return 0


#LV95 to WGS84
df_data = df_data[["X", "Y", "LU18_46", "E", "N"]]
df_data["E*"] = (df_data["E"]-2600000)/1000000
df_data["N*"] = (df_data["N"]-1200000)/1000000
df_data["lambda*"] = (2.6779094 + 4.728982 * df_data["E*"] 
                     + 0.791484 * df_data["E*"] * df_data["N*"]
                     + 0.1306 * df_data["E*"] * df_data["N*"]**2
                     - 0.0436 * df_data["E*"]**3)

df_data["epsylon*"] = (16.9023892 + 3.238272 * df_data["N*"] 
                     - 0.270978 * df_data["E*"]**2
                     - 0.002528 * df_data["N*"]**2
                     - 0.0447 * df_data["E*"]**2 * df_data["N*"]
                     - 0.0140 * df_data["N*"]**3)

df_data["lambda"] = df_data["lambda*"] * 100 / 36
df_data["epsylon"] = df_data["epsylon*"] * 100 / 36

exp_hail_agr = Exposures()

df_data["is_agrar"] = df_data.apply(lambda row: is_agrar(row), axis = 1)
exp_hail_agr["agr_type"] = df_data["LU18_46"]


exp_hail_agr["latitude"] = df_data["epsylon"]
exp_hail_agr["longitude"] = df_data["lambda"]

# exp_hail_agr = exp_hail_agr[(
#     exp_hail_agr["longitude"]>= 5.9625) & (
#         exp_hail_agr["longitude"]<= 10.4625)]
        
# exp_hail_agr = exp_hail_agr[(
#     exp_hail_agr["latitude"]>= 45.829167) & (
#         exp_hail_agr["latitude"]<= 47.795833)]


tot_val_agr = 3.5*10**9
exp_hail_agr["value"] = tot_val_agr/sum(df_data["is_agrar"])

epx_hail_agr.check()

exp_hail_agr.write_hdf5("exp_hail_agr.hdf5")



