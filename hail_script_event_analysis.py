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
ev_list = ["26/05/2009", "23/07/2009", "12/07/2011", "13/07/2011"]#["01/07/2019", "02/07/2019", "18/08/2019", "06/08/2019", "30/06/2019", "15/06/2019"]#, "01/07/2019", "18/06/2019"]
imp_fun_infrastructure = {"imp_id": 1, "L": 0.1, "x_0": 100, "k": 10}
imp_fun_grape = {"imp_id": 2, "L": 0.8, "x_0": 35, "k": 10}
imp_fun_fruit = {"imp_id": 3, "L": 1.0, "x_0": 80, "k": 10}
imp_fun_agriculture = {"imp_id": 4, "L": 0.5, "x_0": 50, "k": 10}
imp_fun_dur_grape = {"imp_id": 5, "L": 1.37, "x_0": 38, "k": 42}
imp_fun_dur_fruit = {"imp_id": 6, "L": 1.37, "x_0": 38, "k": 42}
imp_fun_dur_agriculture = {"imp_id": 7, "L": 1.37, "x_0": 38, "k": 42}
# imp_fun_dur_grape = {"imp_id": 5, "L": 0.8, "x_0": 50, "k": 1}
# imp_fun_dur_fruit = {"imp_id": 6, "L": 1.0, "x_0": 50, "k": 1}
# imp_fun_dur_agriculture = {"imp_id": 7, "L": 0.5, "x_0": 50, "k": 1}

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
years_synth = ["1979", "1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001" ,"2002", "2003", "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018","2019"]


#%% Hazard
haz_real = fct.load_haz(force_new_hdf5_generation, "haz_real", name_hdf5_file, input_folder, years)
haz_dur = fct.load_haz(force_new_hdf5_generation, "haz_dur", name_hdf5_file, input_folder, years)

haz_real.check()

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
if plot_img:
    ifset_hail.plot()

#%% Exposure

exp_infr = fct.load_exp_infr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real)
exp_meshs = fct.load_exp_agr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real)
exp_dur = exp_meshs.copy()
exp_dur["if_HL"] = exp_dur["if_HL"]+3 #change if_HL to match the corresponding imp_id
if plot_img:    
    exp_infr.plot_basemap()
    #This takes to long. Do over night!!!
    #exp_agr.plot_basemap() 

#%% Impact
imp_infr = Impact()
imp_infr.calc(exp_infr, ifset_hail, haz_real,save_mat=True)
# imp_infr.plot_raster_eai_exposure()
freq_curve_infr = imp_infr.calc_freq_curve()
if plot_img:
    freq_curve_infr.plot()

imp_agr = Impact()
imp_agr.calc(exp_meshs, ifset_hail, haz_real, save_mat = True)
freq_curve_agr = imp_agr.calc_freq_curve()
if plot_img:
    freq_curve_agr.plot()

imp_agr_dur = Impact()
imp_agr_dur.calc(exp_dur, ifset_hail, haz_dur, save_mat = True)
freq_curve_agr_dur = imp_agr.calc_freq_curve()
if plot_img:
    freq_curve_agr_dur.plot()

if plot_img:
    for ev_name in ev_list:
        imp_infr.plot_basemap_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])
        imp_agr.plot_basemap_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])
        imp_agr_dur.plot_basemap_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])
        imp_infr.plot_hexbin_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])
        imp_agr.plot_hexbin_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])
        imp_agr_dur.plot_basemap_impact_exposure(event_id = haz_real.get_event_id(event_name=ev_name)[0])    
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("I'm done with the script")

def plot_event(name, ifset_hail, haz_real, haz_dur, exp_infr, exp_meshs, exp_dur, plot_img):
    print("Event Analysis for {}".format(name))
    ev_id = haz_real.get_event_id(event_name = name)
    meshs_intensity = haz_real.intensity[ev_id].todense().astype(int)
    meshs_intensity_no_0 = np.array(meshs_intensity[meshs_intensity!=0]).ravel()
    #remove outliers
    meshs_intensity_no_0 = np.delete(meshs_intensity_no_0, np.where(meshs_intensity_no_0 == 244))
    dur_intensity = haz_dur.intensity[ev_id].todense().astype(int)
    dur_intensity_no_0 = np.array(dur_intensity[dur_intensity!=0]).ravel()
    fig, axs = plt.subplots(1,2, sharey=False, tight_layout = False)
    fig.suptitle("Histogramm event {}".format(name))
    axs[0].hist(meshs_intensity_no_0, bins = 25)
    axs[1].hist(dur_intensity_no_0)
    axs[0].set(xlabel = "meshs [mm]", ylabel = "frequency")
    axs[1].set(xlabel = "duration [min]", ylabel = "frequency")
    axs[0].locator_params(axis="y", integer = True)
    axs[1].locator_params(axis="y", integer = True)
    fig.subplots_adjust(wspace = 0.35)
    plt.show()
    haz_real.plot_intensity(event = ev_id)
    plt.show()
    haz_real_ev = haz_real.select(event_names = [name])
    haz_dur_ev = haz_dur.select(event_names = [name])
    imp_agr_real_ev = Impact()
    imp_agr_dur_ev = Impact()
    imp_infr_real_ev = Impact()
    imp_agr_real_ev.calc(exp_meshs, ifset_hail, haz_real_ev, save_mat = True)
    imp_agr_dur_ev.calc(exp_dur, ifset_hail, haz_dur_ev, save_mat = True)
    imp_infr_real_ev.calc(exp_infr, ifset_hail, haz_real_ev, save_mat = True)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Meshs on agr at event {}: at_event: {} mio; aai_agg: {} mio; eai_exp: {} mio". format(name, imp_agr_real_ev.at_event/1e6,imp_agr_real_ev.aai_agg/1e6, imp_agr_real_ev.eai_exp/1e6))
    print("Duration on agr at event {}: at_event: {} mio; aai_agg: {} mio; eai_exp: {} mio". format(name, imp_agr_dur_ev.at_event/1e6, imp_agr_dur_ev.aai_agg/1e6, imp_agr_dur_ev.eai_exp/1e6))
    print("Meshs on infr at event {}: at_event: {} mio; aai_agg: {} mio; eai_exp: {} mio". format(name, imp_infr_real_ev.at_event/1e6, imp_infr_real_ev.aai_agg/1e6, imp_infr_real_ev.eai_exp/1e6))
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    if plot_img:
        imp_agr_real_ev.plot_basemap_impact_exposure()
        imp_agr_real_ev.plot_hexbin_impact_exposure()
        imp_agr_dur_ev.plot_basemap_impact_exposure()
        imp_agr_dur_ev.plot_hexbin_impact_exposure()    
for name in ev_list:
    plot_event(name,ifset_hail, haz_real, haz_dur, exp_infr, exp_meshs, exp_dur, plot_img)
#Secret test chamber pssst
if False:
    a=3
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    