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
plot_img = False
haz_type = "HL"
ev_list = ["12/07/2011", "13/07/2011"]#["01/07/2019", "02/07/2019", "18/08/2019", "06/08/2019", "30/06/2019", "15/06/2019"]#, "01/07/2019", "18/06/2019"]
start_day = 0 #min = 0
end_day = 183 #max = 183
imp_fun_infrastructure = {"imp_id": 1, "max_y": 0.1, "middle_x": 100, "width": 100}
imp_fun_grape = {"imp_id": 2, "max_y": 0.8, "middle_x": 35, "width": 50}
imp_fun_fruit = {"imp_id": 3, "max_y": 1.0, "middle_x": 80, "width": 150}
imp_fun_agriculture = {"imp_id": 4, "max_y": 0.5, "middle_x": 50, "width": 100}
imp_fun_parameter = [imp_fun_infrastructure, imp_fun_grape, imp_fun_fruit, imp_fun_agriculture]
# Path to Data 
input_folder = "/home/jan/Documents/ETH/Masterarbeit/input"
results_folder = "~/Documents/ETH/Masterarbeit/results"
years =["2018"]#["2002", "2003", "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018","2019"]



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
                                 imp_fun_dict["max_y"], 
                                 imp_fun_dict["middle_x"], 
                                 imp_fun_dict["width"])
    ifset_hail.append(imp_fun)


#%% Exposure

exp_infr = fct.load_exp_infr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real)
exp_agr = fct.load_exp_agr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real)
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
imp_agr.calc(exp_agr, ifset_hail, haz_real, save_mat = True)
freq_curve_agr = imp_agr.calc_freq_curve()
freq_curve_agr.plot()
plt.show()

imp_agr_dur = Impact()
imp_agr_dur.calc(exp_agr, ifset_hail, haz_dur, save_mat = True)
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

# imp_agr.plot_raster_eai_exposure(raster_res = 0.008333333333325754)