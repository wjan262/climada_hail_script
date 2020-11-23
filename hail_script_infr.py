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
imp_fun_infrastructure = {"imp_id": 1, "L": 0.1, "x_0": 100, "k": 10}



imp_fun_parameter = [imp_fun_infrastructure,]
# Path to Data 
input_folder = "/home/jan/Documents/ETH/Masterarbeit/input"
results_folder = "~/Documents/ETH/Masterarbeit/results"
years =["2002", "2003", "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018","2019"]
years_synth = ["1979", "1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001" ,"2002", "2003", "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018","2019"]


#%% Hazard
haz_real = fct.load_haz(force_new_hdf5_generation, "haz_real", name_hdf5_file, input_folder, years)
haz_synth = fct.load_haz(force_new_hdf5_generation, "haz_synth", name_hdf5_file, input_folder, years_synth)
haz_dur = fct.load_haz(force_new_hdf5_generation, "haz_dur", name_hdf5_file, input_folder, years)

haz_real.check()
haz_synth.check()
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

exp_infr = fct.load_exp_infr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real)

if plot_img:    
    exp_infr.plot_basemap()
    exp_infr.plot_hexbin()
    exp_infr.plot_scatter()
    exp_infr.plot_raster()


#%% Impact
imp_infr = Impact()
imp_infr.calc(exp_infr, ifset_hail, haz_real,save_mat=True)
# imp_infr.plot_raster_eai_exposure()
freq_curve_infr = imp_infr.calc_freq_curve()
if plot_img:
    freq_curve_infr.plot()
    plt.show()

print("dmg infr {} Mio CHF".format(imp_infr.aai_agg/1e6))


print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("I'm done with the script")

#Secret test chamber pssst
if False:
    print("dmg infr {} Mio CHF, dmg agr_meshs {} Mio CHF, dmg agr_dur {} Mio CHF".format(imp_infr.aai_agg/1e6, imp_agr.aai_agg/1e6, imp_agr_dur.aai_agg/1e6))
    agr_meshs_yearly_imp = list(imp_agr.calc_impact_year_set(year_range = [2002, 2019]).values())
    agr_dur_yearly_imp = list(imp_agr_dur.calc_impact_year_set(year_range = [2002, 2019]).values())
    # plt.figure()
    # plt.bar(years, agr_dur_yearly_imp)
    # plt.show()
    # plt.figure()
    # plt.bar(years, agr_meshs_yearly_imp)
    dmg_from_sturmarchiv = [27.48, 46.14, 80.67, 76.80, 32.66, 62.47, 26.30, 110.60, 13.01, 34.53, 21.50, 71.77, 22.80, 19.84, 17.50, 35.80, 24.40, 33.30]
    dmg_from_sturmarchiv = [i*1e6 for i in dmg_from_sturmarchiv]
    norm_agr_meshs_yearly_imp = np.divide(agr_meshs_yearly_imp, min(agr_meshs_yearly_imp))
    norm_agr_dur_yearly_imp = np.divide(agr_dur_yearly_imp, min(agr_dur_yearly_imp))
    norm_dmg_from_sturmarchiv = np.divide(dmg_from_sturmarchiv, min(dmg_from_sturmarchiv)) #[i / min(dmg_from_sturmarchiv) for i in dmg_from_sturmarchiv]
    
    #plot
    plt.figure()
    plt.plot(years, agr_meshs_yearly_imp)
    plt.plot(agr_dur_yearly_imp)
    plt.plot(dmg_from_sturmarchiv, linewidth = 3)
    plt.legend(["meshs", "dur", "sturmarchiv"])
    plt.xticks(rotation = 45)
    plt.show()
