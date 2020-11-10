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
load_exp_with_hdf5 = True
load_haz_with_hdf5 = True
load_resampling_data = False #False = real data, True = Synth data
optimize_type = "meshs" # "" = no optimization, "meshs", "dur"
plot_img = False
haz_type = "HL"
ev_list = ["12/07/2011", "13/07/2011"]#["01/07/2019", "02/07/2019", "18/08/2019", "06/08/2019", "30/06/2019", "15/06/2019"]#, "01/07/2019", "18/06/2019"]
start_day = 0 #min = 0
end_day = 183 #max = 183
imp_fun_infrastructure = {"imp_id": 1, "L": 0.08, "x_0": 100, "k": 5}
imp_fun_grape = {"imp_id": 2, "L": 0.80, "x_0": 48, "k": 23}
imp_fun_fruit = {"imp_id": 3, "L": 0.11, "x_0": 42, "k": 39.5}
imp_fun_agriculture = {"imp_id": 4, "L": 0.036, "x_0": 155, "k": 21.33}
imp_fun_dur_grape = {"imp_id": 5, "L": 0.8, "x_0": 20, "k": 1}
imp_fun_dur_fruit = {"imp_id": 6, "L": 1.0, "x_0": 20, "k": 1}
imp_fun_dur_agriculture = {"imp_id": 7, "L": 0.5, "x_0": 20, "k": 1}

imp_fun_parameter = [imp_fun_infrastructure,
                     imp_fun_grape,
                     imp_fun_fruit,
                     imp_fun_agriculture,
                     imp_fun_dur_grape, 
                     imp_fun_dur_fruit, 
                     imp_fun_dur_agriculture]
# Path to Data 
input_folder = "/home/jan/Documents/ETH/Masterarbeit/input"
results_folder = "~/Documents/ETH/Masterarbeit/results"
years =["2002", "2003", "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018","2019"]



#%% Hazard
haz_real = fct.load_haz(force_new_hdf5_generation, "haz_real", name_hdf5_file, input_folder, years)
haz_synth = fct.load_haz(force_new_hdf5_generation, "haz_synth", name_hdf5_file, input_folder, years)
haz_dur = fct.load_haz(force_new_hdf5_generation, "haz_dur", name_hdf5_file, input_folder, years)

haz_real.check()
haz_synth.check()

if plot_img:
    haz_real.plot_intensity(event = 0)
    haz_real.plot_fraction(event = 0)

#%% Impact_function
# Set impact function (see tutorial climada_entity_ImpactFuncSet)
ifset_hail = ImpactFuncSet()
for imp_fun_dict in imp_fun_parameter:
    imp_fun = fct.create_impact_func(haz_type, 
                                 imp_fun_dict["imp_id"], 
                                 imp_fun_dict["L"], 
                                 imp_fun_dict["x_0"], 
                                 imp_fun_dict["k"])
    ifset_hail.append(imp_fun)
ifset_hail.plot()

#%% Exposure

exp_infr = fct.load_exp_infr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real)
exp_meshs = fct.load_exp_agr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real)
exp_dur = exp_meshs.copy()
exp_dur["if_HL"] = exp_dur["if_HL"]+3
if plot_img:    
    exp_infr.plot_basemap()
    #This takes to long. Do over night!!!
    #exp_agr.plot_basemap() 

#%% Impact
imp_infr = Impact()
imp_infr.calc(exp_infr, ifset_hail, haz_real,save_mat=True)
# imp_infr.plot_raster_eai_exposure()
freq_curve_infr = imp_infr.calc_freq_curve()
freq_curve_infr.plot()
plt.show()

imp_agr = Impact()
imp_agr.calc(exp_meshs, ifset_hail, haz_real, save_mat = True)
freq_curve_agr = imp_agr.calc_freq_curve()
freq_curve_agr.plot()
plt.show()

imp_agr_dur = Impact()
imp_agr_dur.calc(exp_dur, ifset_hail, haz_dur, save_mat = True)
freq_curve_agr = imp_agr.calc_freq_curve()
freq_curve_agr.plot()
plt.show()

# for ev_name in ev_list:
#     imp_infr.plot_basemap_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])
#     imp_agr.plot_basemap_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])
# for ev_name in ev_list:
#     imp_infr.plot_hexbin_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_list[1])[0], ignore_zero = False)
#     imp_agr.plot_hexbin_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])
    
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("I'm done with the script")

#Secret test chamber pssst
if True:
    print("dmg infr {} Mio CHF, dmg agr {} Mio CHF".format(imp_infr.aai_agg/1e6, imp_agr.aai_agg/1e6))
    agr_meshs_yearly_imp = list(imp_agr.calc_impact_year_set(year_range = [2002, 2019]).values())
    agr_dur_yearly_imp = list(imp_agr_dur.calc_impact_year_set(year_range = [2002, 2019]).values())
    # plt.figure()
    # plt.bar(years, agr_dur_yearly_imp)
    # plt.show()
    # plt.figure()
    # plt.bar(years, agr_meshs_yearly_imp)
    dmg_from_sturmarchiev = [27.48, 46.14, 80.67, 76.80, 32.66, 62.47, 26.30, 110.60, 13.01, 34.53, 21.50, 74.77, 22.80, 19.84, 17.50, 35.80, 24.40, 33.30]
    
    norm_agr_meshs_yearly_imp = agr_meshs_yearly_imp/min(agr_meshs_yearly_imp)
    norm_agr_dur_yearly_imp = agr_dur_yearly_imp / min(agr_dur_yearly_imp)
    norm_dmg_from_sturmarchiev = [i / min(dmg_from_sturmarchiev) for i in dmg_from_sturmarchiev]
    
    #plot
    plt.figure()
    plt.plot(years, norm_agr_meshs_yearly_imp)
    # plt.plot(norm_agr_dur_yearly_imp)
    plt.plot(norm_dmg_from_sturmarchiev)
    plt.legend(["meshs", "sturmarchiev"])
    plt.show()

    print("pearson for agr with meshs (score, p_value) = {} ".format(stats.pearsonr(norm_dmg_from_sturmarchiev, norm_agr_meshs_yearly_imp)))
    print("pearson for agr with dur (score, p_value) = {} ".format(stats.pearsonr(norm_dmg_from_sturmarchiev, norm_agr_dur_yearly_imp)))
    
    coef, p_value = spearmanr(norm_dmg_from_sturmarchiev, norm_agr_meshs_yearly_imp)    
    print("spearman for agr with meshs (score, p_value) = ({}, {})".format(coef, p_value))
    coef, p_value = spearmanr(norm_dmg_from_sturmarchiev, norm_agr_dur_yearly_imp)    
    print("spearman for agr with dur (score, p_value) = ({}, {})".format(coef, p_value))
    #%% Optimization
    optimize_type = "" # "" = no optimization, "meshs", "dur"
    if optimize_type != "":
        num_fct = 3 #[1:3]
        bounds = num_fct*[(0.0,1),(1.0,200),(0.0,40)]
        init_parameter=[]
        if optimize_type == "meshs":
            parameter_optimize = imp_fun_parameter[1:1+num_fct]
            haz = haz_real
            exp = exp_meshs.copy()
        elif optimize_type == "dur":
            parameter_optimize = imp_fun_parameter[4:4+num_fct]
            haz = haz_dur
            exp = exp_dur.copy()
        for i in range(num_fct):
            init_parameter += [*parameter_optimize[i].values()][1:4]
            
        if num_fct == 1:
            exp["if_HL"] = parameter_optimize[0]["imp_id"]
        args = (parameter_optimize, exp, haz, haz_type, num_fct)
        # optimize_results = optimize.differential_evolution(func=fct.make_Y, bounds = bounds, args = args, workers = 3)
        # optimize_results = optimize.brute(func = fct.make_Y, ranges = bounds, args = args, Ns = 5, full_output=True)
        # optimize_results = optimize.minimize(fun = fct.make_Y, x0 = init_parameter,method="Powell", args = args, bounds = bounds)
        # test = fct.make_Y(init_parameter, args)
        # print(optimize_results)

# imp_agr.plot_raster_eai_exposure(raster_res = 0.008333333333325754)
    #%% WICHTIG: WIESO IST haz_hail.intensity_thresh = 10?????????????
