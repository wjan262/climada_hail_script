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
from climada.entity.tag import Tag

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
plot_img = True
haz_type = "HL"
ev_list = ["12/07/2011", "13/07/2011"]#["01/07/2019", "02/07/2019", "18/08/2019", "06/08/2019", "30/06/2019", "15/06/2019"]#, "01/07/2019", "18/06/2019"]
imp_fun_infrastructure = {"imp_id": 1, "L": 0.1, "x_0": 100, "k": 10}
imp_fun_grape = {"imp_id": 2, "L": 0.8, "x_0": 35, "k": 10}
imp_fun_fruit = {"imp_id": 3, "L": 1.0, "x_0": 80, "k": 10}
imp_fun_agriculture = {"imp_id": 4, "L": 0.5, "x_0": 50, "k": 10}



imp_fun_parameter = [imp_fun_infrastructure,
                     imp_fun_grape,
                     imp_fun_fruit,
                     imp_fun_agriculture,]
# Path to Data 
input_folder = "/home/jan/Documents/ETH/Masterarbeit/input"
results_folder = "~/Documents/ETH/Masterarbeit/results"
years_synth = ["1979", "1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001" ,"2002", "2003", "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018","2019"]


#%% Hazard
haz_synth = fct.load_haz(force_new_hdf5_generation, "haz_synth", name_hdf5_file, input_folder, years_synth)

haz_synth.check()

if plot_img:
    haz_synth.plot_fraction(0)
    haz_synth.plot_intensity(0)
    haz_synth.plot_fraction(-1)
    haz_synth.plot_intensity(-1)
    haz_synth.plot_rp_intensity(return_periods=(1, 5, 10, 20))

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

exp_synth_infr_meshs = fct.load_exp_infr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_synth)
exp_synth_agr_meshs = fct.load_exp_agr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_synth)
if plot_img:    
    exp_synth_infr_meshs.plot_basemap()
    exp_synth_infr_meshs.plot_hexbin()
    exp_synth_infr_meshs.plot_scatter()
    exp_synth_infr_meshs.plot_raster()
    
    exp_synth_agr_meshs.tag = Tag(file_name = "exp_agr", description="Exposure_description")
    exp_synth_agr_meshs.plot_basemap()
    exp_synth_agr_meshs.plot_hexbin()
    exp_synth_agr_meshs.plot_scatter()
    exp_synth_agr_meshs.plot_raster(raster_res = 0.001)

#%% Impact
imp_synth_infr_meshs = Impact()
imp_synth_infr_meshs.calc(exp_synth_infr_meshs, ifset_hail, haz_synth,save_mat=True)
freq_curve_synth_infr_meshs = imp_synth_infr_meshs.calc_freq_curve()
if plot_img:
    freq_curve_synth_infr_meshs.plot()
    imp_synth_infr_meshs.plot_basemap_eai_exposure()
    imp_synth_infr_meshs.plot_hexbin_eai_exposure()
    imp_synth_infr_meshs.plot_scatter_eai_exposure()
    imp_synth_infr_meshs.plot_raster_eai_exposure(raster_res = 0.001)

imp_synth_agr_meshs = Impact()
imp_synth_agr_meshs.calc(exp_synth_agr_meshs, ifset_hail, haz_synth, save_mat=True)
freq_curve_synth_agr_meshs = imp_synth_agr_meshs.calc_freq_curve()
if plot_img:
    freq_curve_synth_agr_meshs.plot()
    imp_synth_agr_meshs.plot_basemap_eai_exposure()
    imp_synth_agr_meshs.plot_hexbin_eai_exposure()
    imp_synth_agr_meshs.plot_scatter_eai_exposure()
    imp_synth_agr_meshs.plot_raster_eai_exposure(raster_res = 0.001)
    
print("dmg synth_infr_meshs {} Mio CHF, dmg synth_agr_meshs {} Mio CHF"
      .format(imp_synth_infr_meshs.aai_agg/1e6, imp_synth_agr_meshs.aai_agg/1e6))


print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("I'm done with the script")

#Secret test chamber pssst
if False:
    a=3