#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:54:40 2020

@author: jan
"""

#%% Import
import sys
sys.path.append("/home/jan/Documents/ETH/Masterarbeit/climada_python")
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import geopandas as gpd
from climada.hazard import Hazard
from scipy import sparse
import copy as cp
from matplotlib import pyplot as plt
from climada.entity import Exposures, Entity, LitPop
from climada.entity import ImpactFuncSet, ImpactFunc
from climada.engine import Impact
import h5py
import hail_functions as fct
from sklearn.metrics import mean_squared_error
from scipy import optimize
import time
from scipy import stats
from scipy.stats import spearmanr
#%% Parameter
# If any value in force_new_hdf5_generation is True the script will ask for 
# user input wether to overwrite the hdf5 with the new data
force_new_hdf5_generation ={"haz_real": False, 
                            "haz_synth": False, 
                            "haz_dur": False,
                            "exp_infr": False, 
                            "exp_agr": False}
name_hdf5_file={"haz_real": "haz_real.hdf5", 
                "haz_synth": "haz_synth.hdf5", 
                "haz_dur": "haz_dur.hdf5",
                "exp_infr": "exp_switzerland.hdf5", 
                "exp_agr": "exp_agr.hdf5"}

# Optimization
plot_img = False
haz_type = "HL"
ev_list = ["12/07/2011", "13/07/2011"]#["01/07/2019", "02/07/2019", "18/08/2019", "06/08/2019", "30/06/2019", "15/06/2019"]#, "01/07/2019", "18/06/2019"]
imp_fun_infr_meshs = {"imp_id": 1, "L": 0.1, "x_0": 100, "k": 10}
imp_fun_infr_dur = {"imp_id": 8, "L": 0.01, "x_0": 120, "k": 1}


imp_fun_parameter = [imp_fun_infr_meshs,
                     imp_fun_infr_dur]
# Path to Data 
input_folder = "/home/jan/Documents/ETH/Masterarbeit/input"
results_folder = "~/Documents/ETH/Masterarbeit/results"
years =["2002", "2003", "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018","2019"]


#%% Hazard
haz_real = fct.load_haz(force_new_hdf5_generation, "haz_real", name_hdf5_file, input_folder, years)
haz_dur = fct.load_haz(force_new_hdf5_generation, "haz_dur", name_hdf5_file, input_folder, years)

haz_real.check()
haz_dur.check()

if plot_img:
    haz_real.plot_fraction(0)
    haz_real.plot_intensity(0)
    haz_real.plot_fraction(-1)
    haz_real.plot_intensity(-1)
    haz_real.plot_rp_intensity(return_periods=(1, 5, 10, 20))
    haz_dur.plot_fraction(0)
    haz_dur.plot_intensity(0)
    haz_dur.plot_fraction(-1)
    haz_dur.plot_intensity(-1)
    haz_dur.plot_rp_intensity(return_periods=(1, 5, 10, 20))
#%% Impact_function
# Set impact function (see tutorial climada_entity_ImpactFuncSet)
ifset_hail = ImpactFuncSet()
for imp_fun_dict in imp_fun_parameter:
    imp_fun = fct.create_impact_func(haz_type, 
                                 imp_fun_dict["imp_id"], 
                                 imp_fun_dict["L"], 
                                 imp_fun_dict["x_0"], 
                                 imp_fun_dict["k"])
    imp_fun.mdd[:] = 0.1
    ifset_hail.append(imp_fun)
if plot_img:
    ifset_hail.plot()

#%% Exposure
exp_infr_meshs = fct.load_exp_infr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real)
exp_infr_dur = exp_infr_meshs.copy()
exp_infr_dur["if_HL"] = 8 #change if_HL to match the corresponding imp_id
if plot_img:    
    exp_infr_meshs.plot_basemap()
    exp_infr_meshs.plot_hexbin()
    exp_infr_meshs.plot_scatter()
    exp_infr_meshs.plot_raster()


#%% Impact
imp_infr_meshs = Impact()
imp_infr_meshs.calc(exp_infr_meshs, ifset_hail, haz_real,save_mat=True)
# imp_infr.plot_raster_eai_exposure()
freq_curve_infr_meshs = imp_infr_meshs.calc_freq_curve()
if plot_img:
    freq_curve_infr_meshs.plot()
    imp_infr_meshs.plot_basemap_eai_exposure()
    imp_infr_meshs.plot_hexbin_eai_exposure()
    imp_infr_meshs.plot_scatter_eai_exposure()
    imp_infr_meshs.plot_raster_eai_exposure()

imp_infr_dur = Impact()
imp_infr_dur.calc(exp_infr_dur, ifset_hail, haz_dur, save_mat=True)
# imp_infr.plot_raster_eai_exposure()
freq_curve_infr_dur = imp_infr_dur.calc_freq_curve()
if plot_img:
    freq_curve_infr_dur.plot()
    imp_infr_dur.plot_basemap_eai_exposure()
    imp_infr_dur.plot_hexbin_eai_exposure()
    imp_infr_dur.plot_scatter_eai_exposure()
    imp_infr_dur.plot_raster_eai_exposure()


print("dmg infr meshs {} Mio CHF, dmg infr dur {} Mio CHF".format(imp_infr_meshs.aai_agg/1e6, imp_infr_dur.aai_agg/1e6))
ifset_hail.plot()
plt.show()
#aai_agg for % impact
if False:
    plt.show()
    imp_fun_list = np.arange(0, 0.005, 0.0001)
    dmg_for_imp_list = []
    for i in imp_fun_list:
        ifset_hail = ImpactFuncSet()
        imp_fun = fct.create_impact_func(haz_type, 
                                     1, 
                                     1, 
                                     1, 
                                     1)
        imp_fun.mdd[:] = i
        ifset_hail.append(imp_fun)
        
        imp_infr_meshs.calc(exp_infr_meshs, ifset_hail, haz_real,save_mat=False)
        dmg_for_imp_list.append(imp_infr_meshs.aai_agg/1e6)
        
    plt.plot(imp_fun_list, dmg_for_imp_list, "bo")
    plt.xlabel("% Affected Exposure")
    plt.ylabel("aai_agg in Millions")
    plt.title("MESHS on Infrastructure")
    plt.show()

#aai_agg for each Meshs size
if False:
    imp_fun_list = np.arange(1, 150, 1)
    dmg_for_imp_list = []
    for i in imp_fun_list:
        ifset_hail = ImpactFuncSet()
        imp_fun = fct.create_impact_func(haz_type, 
                                     1, 
                                     1, 
                                     1, 
                                     1)
        imp_fun.mdd[:] = 0
        imp_fun.mdd[i] = 1.0
        ifset_hail.append(imp_fun)
        
        imp_infr_meshs.calc(exp_infr_meshs, ifset_hail, haz_real,save_mat=True)
        dmg_for_imp_list.append(imp_infr_meshs.aai_agg/1e6)
    
    plt.plot(imp_fun_list, dmg_for_imp_list, "bo")
    plt.xlabel("Meshs size")
    plt.ylabel("aai_agg in Millions")
    plt.title("MESHS on Infrastructure")
plt.show()

#aai_agg for each Parameter
x_tresh_list = np.arange(20, 80, 30)
L = 1
label = []
for x_tresh in x_tresh_list:
    imp_fun_param_list = np.arange(x_tresh+1, 111, 30)
    dmg_for_imp_fun_param = []
    for i in imp_fun_param_list:
        y = fct.sigmoid2(np.arange(0, 150), L = L, x_0 = i, x_tresh = x_tresh)
        ifset_hail = ImpactFuncSet()
        imp_fun = fct.create_impact_func(haz_type, 1, 1, 1, 1, y = y)
        ifset_hail.append(imp_fun)
        plt.plot(y)
        label.append("x_tresh = {}; x_half = {}".format(x_tresh, i))
plt.title("Impfun Meshs on Infr, L = {}".format(L))
plt.xlabel("Intensity [mm]")
plt.ylabel("Impact [%]")
plt.legend(labels=label)
plt.show()




labels = []
for x_tresh in np.arange(20, 50, 5):
    L = 0.01
    x_tresh = x_tresh
    imp_fun_param_list = np.arange(x_tresh+1, 101, 1)
    dmg_for_imp_fun_param = []
    for i in imp_fun_param_list:
        y = fct.sigmoid2(np.arange(0, 150), L = L, x_0 = i, x_tresh = x_tresh)
        ifset_hail = ImpactFuncSet()
        imp_fun = fct.create_impact_func(haz_type, 1, 1, 1, 1, y = y)
        ifset_hail.append(imp_fun)
    
        imp_infr_meshs.calc(exp_infr_meshs, ifset_hail, haz_real,save_mat=False)
        dmg_for_imp_fun_param.append(imp_infr_meshs.aai_agg/1e6)
    labels.append("x_tresh = {}".format(x_tresh))
    plt.plot(imp_fun_param_list, dmg_for_imp_fun_param)
    plt.xlabel("Imp_fun Parameter x_0 (x_half)")
    plt.ylabel("aai_agg in Millions")
    plt.title("MESHS on Infrastructure (L = {})".format(L))
plt.legend(labels)
plt.show()
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("I'm done with the script")





#Secret test chamber pssst
if False:
    a=3
