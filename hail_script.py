#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:54:40 2020

@author: jan
"""


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
# Parameter
load_exp_with_h5py = True
load_haz_with_hdf5 = True
plot_img = False
haz_type = "HL"
ev_list = []#["01/07/2019", "02/07/2019", "18/08/2019", "06/08/2019", "30/06/2019", "15/06/2019"]#, "01/07/2019", "18/06/2019"]
start_day = 0 #min = 0
end_day = 183 #max = 183
imp_fun_infrastructure = {"imp_id": 1, "max_y": 0.1, "middle_x": 100, "width": 100}
imp_fun_grape = {"imp_id": 2, "max_y": 0.8, "middle_x": 35, "width": 50}
imp_fun_fruit = {"imp_id": 3, "max_y": 1.0, "middle_x": 80, "width": 150}
imp_fun_agriculture = {"imp_id": 4, "max_y": 0.5, "middle_x": 50, "width": 100}

imp_fun_parameter = [imp_fun_infrastructure, imp_fun_grape, imp_fun_fruit, imp_fun_agriculture]


# Path to Data 
#Todo Make path absolute (input und result folder)
input_folder = "/home/jan/Documents/ETH/Masterarbeit/input"
results_folder = "~/Documents/ETH/Masterarbeit/results"
years = ["2002", "2003", "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018","2019"]
# path_poh = "hail_log_data/BZC_X1d66_V2_2019.nc"
# path_meshs = "hail_log_data/MZC_X1d66_V2_2019.nc"

def create_impact_func(haz_type, imp_id, max_y, middle_x, width):
    name = {1: "Hail on infrastructure", 2: "Hail on grape production", 3: "Hail on fruit production", 4: "Hail on agriculture (without fruits and grape production"}
    imp_fun= ImpactFunc() 
    imp_fun.haz_type = haz_type
    imp_fun.id = imp_id
    imp_fun.name = name[imp_id]
    imp_fun.intensity_unit = 'mm'
    imp_fun.intensity = np.linspace(0, 244, num=245)
    imp_fun.mdd = mdd_function(max_y, middle_x, width)
    imp_fun.paa = np.linspace(1, 1, num=245)
    imp_fun.check()
    return imp_fun

def get_hail_data(years,
                  input_folder,
                  start_time=0,
                  end_time=183,
                  chunk_size = 100000):
    """

    Parameters
    ----------
    years : list of int
        List containing years to import radar data from.
    input_folder : str
        Path to folder containing radar data files
    start_time : int, optional
        first day read from yearly data. The default is 0 which corresponds to '01/04/2019'.
    end_time : int, optional
        last day read from yearly data. The default is 183 which corresponds to '30/09/2019'.
    chunk_size : int, optional
        chunk size to load data using xarray. The default is 100000.

    Returns
    -------
    fraction : scipy.sparse.csr.csr_matrix
        scipy sparse matrix containing the fraction for each event (a day is an event).
    intensity : scipy.sparse.csr.csr_matrix
        scipy sparse matrix containing the intensity for each event (a day is an event).
    event_name : list of str
        name for each event (date as string '01/04/2019').
    lat : numpy.ndarray
        array containing latitude for each point.
    lon : numpy.ndarray
        array containing longitude for each point.

    """
    fraction = None
    intensity = None
    lat = None
    lon = None
    event_name = []
    date = np.ndarray(0)
    for year in years:
        xr_poh = xr.open_dataset(
            input_folder + "/" + "BZC_X1d66_V2_" + year + ".nc",
            chunks = chunk_size)
        xr_meshs = xr.open_dataset(
             input_folder + "/" + "MZC_X1d66_V2_" + year + ".nc",
             chunks = chunk_size)
        if lat is None:
            lat = np.reshape(xr_poh.sel(time = xr_poh.time[0]).coords["lat"].values, -1, order="F")
            lon = np.reshape(xr_poh.sel(time = xr_poh.time[0]).coords["lon"].values, -1, order="F")

        for i in range(start_time, end_time):
            df_poh = xr_poh.sel(time = xr_poh.time[i]).to_dataframe() #get the data by day
            df_poh["BZC"] = np.where(df_poh["BZC"] >= 80, 1, 0) #this is a test
            df_meshs = xr_meshs.sel(time = xr_meshs.time[i]).to_dataframe()
            csr_poh = sparse.csr_matrix(df_poh["BZC"].fillna(value=0))
            csr_meshs = sparse.csr_matrix(df_meshs["MZC"].fillna(value=0))
            event_name.append(df_poh.time[0].strftime("%d/%m/%Y"))
            date = np.append(date, df_poh.time[0].toordinal())
            if fraction is None: #first iteration        
                fraction = csr_poh #0-100%
                intensity = csr_meshs #20-244mm
            else: #all the following iteration get append.
                fraction = sparse.vstack([fraction, csr_poh])
                intensity = sparse.vstack([intensity, csr_meshs])
        xr_poh.close()
        xr_meshs.close()
    return fraction, intensity, event_name, lat, lon, date

def mdd_function(max_y=0.1, middle_x=90, width=100, plot_y = False):
    """
    

    Parameters
    ----------
    max_y : float, optional
        max value for mdd. The default is 0.1.
    middle_x : int, optional
        specifies x location of the function. The default is 90.
    width : TYPE, optional
        specifies the extent in x direciton of the function. The default is 100.
    plot_y : bool, optional
        plot y. The default is False.

    Returns
    -------
    y : numpy.ndarray
        for impact function mdd.

    """
    if width/2 > middle_x:
        print("Changed middle_x form {} to width/2 {}".format(middle_x, width/2))
        middle_x = int(width/2)
    y = np.zeros(245)
    x = np.linspace(-6, 6, num=width)
    start_sig = int(middle_x - width/2)
    end_sig = start_sig + width
    y[start_sig:end_sig] = 1/(1+np.exp(-x))
    y[end_sig:] = 1.
    y = y * max_y
    y[0:20]=0 #values under 20 do not exist in MESHS
    if plot_y:
        plt.plot(y)
    return y

if load_haz_with_hdf5:
    haz_hail = Hazard()
    haz_hail.read_hdf5(input_folder + "/haz_hail.hdf5")
else:
    haz_hail = Hazard(haz_type)
    haz_hail.units = "mm"
    haz_hail.fraction, haz_hail.intensity, haz_hail.event_name, lat, lon, haz_hail.date = get_hail_data(
        years = years,  
        input_folder = input_folder, 
        start_time=start_day,  
        end_time=end_day)
    haz_hail.intensity_thres = 20
    #set coordinates
    haz_hail.centroids.set_lat_lon(lat, lon)
    haz_hail.event_id = np.arange(haz_hail.intensity.shape[0], dtype = int) + 1
    #set frequency for all events to 1
    haz_hail.frequency = np.ones(haz_hail.intensity.shape[0])/len(years)
haz_hail.check()

if plot_img:
    haz_hail.plot_intensity(event = 0)
    haz_hail.plot_fraction(event = 0)

# Set impact function (see tutorial climada_entity_ImpactFuncSet)
ifset_hail = ImpactFuncSet()
for imp_fun_dict in imp_fun_parameter:
    imp_fun = create_impact_func(haz_type, 
                                 imp_fun_dict["imp_id"], 
                                 imp_fun_dict["max_y"], 
                                 imp_fun_dict["middle_x"], 
                                 imp_fun_dict["width"])
    ifset_hail.append(imp_fun)

# if_hail = ImpactFunc() 
# if_hail.haz_type = haz_type
# if_hail.id = 1
# if_hail.name = 'LS Linear function'
# if_hail.intensity_unit = 'mm'
# if_hail.intensity = np.linspace(0, 244, num=245)
# if_hail.mdd = mdd_function(max_y = 0.1, middle_x = 90, width = 100)
# if_hail.paa = np.linspace(0, 1, num=245)
# if_hail.check()
# if plot_img:
#     if_hail.plot()
# ifset_hail = ImpactFuncSet()
# ifset_hail.append(if_hail)




# Set exposure: (see tutorial climada_entity_LitPop)
if load_exp_with_h5py:
    # LitPop Exposure
    exp_hail = LitPop()
    exp_hail.read_hdf5(input_folder +"/exp_switzerland.hdf5")
    exp_hail.check()
    #Agrar Exposure    
    exp_agr = Exposures()
    exp_agr.read_hdf5(input_folder + "/exp_agr.hdf5")
    exp_agr.check()
else: #be carefull, this step will take ages when you do both at once
    # LitPop Exposure
    exp_hail = LitPop()
    exp_hail.set_country('Switzerland', reference_year = 2019)
    exp_hail.set_geometry_points()
    exp_hail = exp_hail.rename(columns = {'if_': 'if_HL'})
    exp_hail = Exposures(exp_hail)
    exp_hail.set_lat_lon()
    exp_hail.check()
    exp_hail.assign_centroids(haz_hail, method = "NN", distance ="haversine", threshold = 2)
    exp_hail.write_hdf5(input_folder + "/exp_switzerland.hdf5")
    # Agrar Exposure
    exp_agr = Exposures()
    exp_agr.read_hdf5(input_folder + "/exp_hail_agr.hdf5")
    # exp_agr.loc[exp_agr["region_id"]==201, "if_"]= int(3)
    # exp_agr.loc[exp_agr["region_id"]==202, "if_"]= int(2)
    # exp_agr.loc[exp_agr["region_id"]==221, "if_"]= int(4)
    # exp_agr = exp_agr.rename(columns = {'if_': 'if_HL'})

    exp_agr.check()
    exp_agr.assign_centroids(haz_hail, method = "NN", distance = "haversine", threshold = 2)
    exp_agr.write_hdf5(input_folder + "/exp_agr.hdf5")


if plot_img:    
    exp_hail.plot_basemap()
    #This takes to long. Do over night!!!
    #exp_agr.plot_basemap() 

imp_hail = Impact()
imp_hail.calc(exp_hail, ifset_hail, haz_hail,save_mat=True)
# imp_hail.plot_raster_eai_exposure()

for ev_name in ev_list:
    imp_hail.plot_basemap_impact_exposure(event_id = haz_hail.get_event_id(event_name=ev_name)[0])
#     # plt.show()

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("I'm done with the script")

#Secret test chamber pssst
if True:
    imp_agr = Impact()
    imp_agr.calc(exp_agr, ifset_hail, haz_hail, save_mat = True)
    print("dmg litpop {} Mio CHF, dmg agr {} Mio CHF".format(imp_hail.aai_agg/1e6, imp_agr.aai_agg/1e6))
    for ev_name in ev_list:
        imp_agr.plot_basemap_impact_exposure(event_id = haz_hail.get_event_id(event_name=ev_name)[0])

# imp_agr.plot_raster_eai_exposure(raster_res = 0.008333333333325754)