#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:15:08 2020

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
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error
from scipy import optimize

#%% Functions

def generate_haz(name_haz, name_hdf5_file, input_folder, years):
    """
    

    Parameters
    ----------
    name_haz : str
        name of hazard ["haz_real", "haz_synth", "haz_dur"].
    name_hdf5_file : dict
        names of corresponding hdf5 files.
    input_folder : str
        Path to input folder.
    years : list of str
        years for which data will be loaded.

    Returns
    -------
    haz : climada.hazard.base.Hazard
        CLIMADA hazard.

    """
    # here it gets a bit ugly
    

    if name_haz == "haz_real":
        haz = Hazard("HL")
        haz.units = "mm"
        haz.fraction, haz.intensity, duration, haz.event_name, \
                lat, lon, haz.date = get_hail_data(
                years = years,  
                input_folder = input_folder)
        haz.intensity_thres = 20
        #set coordinates
        haz.centroids.set_lat_lon(lat, lon)
        haz.event_id = np.arange(haz.intensity.shape[0], dtype = int) + 1
        #set frequency
        haz.frequency = np.ones(haz.intensity.shape[0])/len(years)
    elif name_haz == "haz_synth":
        haz = Hazard("HL")
        haz.units = "mm"
        haz.fraction, haz.intensity, haz.event_name, \
                lat, lon, haz.date = get_hail_data_synth(
                years = years,  
                input_folder = input_folder)
        haz.intensity_thres = 20
        #set coordinates
        haz.centroids.set_lat_lon(lat, lon)
        haz.event_id = np.arange(haz.intensity.shape[0], dtype = int) + 1
        #set frequency
        haz.frequency = np.ones(haz.intensity.shape[0])/len(years)
    else:
        haz = Hazard("HL")
        haz.units = "min"
        haz.fraction, placeholder, haz.intensity, haz.event_name, \
                lat, lon, haz.date = get_hail_data(
                years = years,  
                input_folder = input_folder)
        haz.intensity_thres = 0
        #set coordinates
        haz.centroids.set_lat_lon(lat, lon)
        haz.event_id = np.arange(haz.intensity.shape[0], dtype = int) + 1
        #set frequency 
        haz.frequency = np.ones(haz.intensity.shape[0])/len(years)
    return haz

def load_haz(force_new_hdf5_generation, name_haz, name_hdf5_file, input_folder, years):
    """
    Parameters
    ----------
    force_new_hdf5_generation : dict of bool
        Dict containing which hdf5 should be new generated.
    name_haz : str
        name of hazard ["haz_real", "haz_synth", "haz_dur"]. 
    name_hdf5_file : dict
        names of corresponding hdf5 files.
    input_folder : str
        Path to input folder.
    years : list of str
        years for which data will be loaded.

    Returns
    -------
    haz : climada.hazard.base.Hazard
        CLIMADA hazard.

    """
    my_file = Path(input_folder + "/" + name_hdf5_file[name_haz])
    if force_new_hdf5_generation[name_haz]:
        haz = generate_haz(name_haz, name_hdf5_file, input_folder, years)
    elif my_file.exists():
        haz = Hazard()
        haz.read_hdf5(input_folder + "/" + name_hdf5_file[name_haz])
    else:
        print("{} does not exist and will be generated. Be patient".format(my_file))
        haz = generate_haz(name_haz, name_hdf5_file, input_folder, years)
    return haz

def create_impact_func(haz_type, imp_id, L, x_0, k):
    """
    Parameters
    ----------
    haz_type : str
        .
    imp_id : TYPE
        DESCRIPTION.
    max_y : TYPE
        DESCRIPTION.
    middle_x : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.

    Returns
    -------
    imp_fun : TYPE
        DESCRIPTION.

    """
    name = {1: "Hail (Meshs) on infrastructure",
            2: "Hail (Meshs) on grape production",
            3: "Hail (Meshs) on fruit production",
            4: "Hail (Meshs) on agriculture (without fruits and grape production",
            5: "Hail (duration) on grape production", 
            6: "Hail (duration) on fruit production",
            7: "Hail (duration) on agriculture (without fruits and grape production",}
    imp_fun= ImpactFunc() 
    imp_fun.haz_type = haz_type
    imp_fun.id = imp_id
    imp_fun.name = name[imp_id]
    if imp_id <=4:
        imp_fun.intensity_unit = 'mm'
        num = 245
    else:
        imp_fun.intensity_unit = "min"
        num = 120
    x = np.arange(num)
    imp_fun.intensity = np.linspace(0, num, num=num)
    imp_fun.mdd = sigmoid(x, L, x_0, k)

    imp_fun.paa = np.linspace(1, 1, num=num)
    imp_fun.check()
    return imp_fun

def get_hail_data_synth(years,
                  input_folder,
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
        print(year)
        xr_poh = xr.open_dataset(
            input_folder + "/S-2/" + "Synth_BZC_" + str(year) + ".nc",
            chunks = chunk_size)
        xr_meshs = xr.open_dataset(
             input_folder + "/S-2/" + "Synth_MZC_" + str(year) + ".nc",
             chunks = chunk_size)
        if lat is None:
            lat = np.reshape(xr_poh.sel(time = xr_poh.time[0]).coords["lat"].values, -1, order="F")
            lon = np.reshape(xr_poh.sel(time = xr_poh.time[0]).coords["lon"].values, -1, order="F")

        for i in range(xr_poh.time.size):
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

def get_hail_data(years,
                  input_folder,
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
        print(year)
        xr_poh = xr.open_dataset(
            input_folder + "/" + "BZC_X1d66_V2_" + year + ".nc",
            chunks = chunk_size)
        xr_meshs = xr.open_dataset(
             input_folder + "/" + "MZC_X1d66_V2_" + year + ".nc",
             chunks = chunk_size)
        xr_dur = xr.open_dataset(
            input_folder + "/hailgrids_duration/BZC/" + "BZC_DURd66_V2_" + year + ".nc", 
            chunks = chunk_size)
        if lat is None:
            lat = np.reshape(xr_poh.sel(time = xr_poh.time[0]).coords["lat"].values, -1, order="F")
            lon = np.reshape(xr_poh.sel(time = xr_poh.time[0]).coords["lon"].values, -1, order="F")

        for i in range(xr_poh.time.size):
            df_poh = xr_poh.sel(time = xr_poh.time[i]).to_dataframe() #get the data by day
            df_poh["BZC"] = np.where(df_poh["BZC"] >= 80, 1, 0) #this is a test
            df_meshs = xr_meshs.sel(time = xr_meshs.time[i]).to_dataframe()
            df_dur = xr_dur.sel(time = xr_dur.time[i]).to_dataframe()
            csr_poh = sparse.csr_matrix(df_poh["BZC"].fillna(value=0))
            csr_meshs = sparse.csr_matrix(df_meshs["MZC"].fillna(value=0))
            csr_dur = sparse.csr_matrix(df_dur["BZC80_dur"].fillna(value=0)*20)
            event_name.append(df_poh.time[0].strftime("%d/%m/%Y"))
            date = np.append(date, int(df_poh.time[0].toordinal()))
            if fraction is None: #first iteration        
                fraction = csr_poh #0-100%
                intensity = csr_meshs #20-244mm
                duration = csr_dur
            else: #all the following iteration get append.
                fraction = sparse.vstack([fraction, csr_poh])
                intensity = sparse.vstack([intensity, csr_meshs])
                duration = sparse.vstack([duration, csr_dur])
        xr_poh.close()
        xr_meshs.close()
    return fraction, intensity, duration,  event_name, lat, lon, date

def mdd_function_sigmoid(imp_id, max_y=0.1, start_sig = 0, width=100, plot_y = False):
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
    y = np.zeros(245)
    x = np.linspace(-6, 6, num=width)
    end_sig = width + start_sig
    if start_sig < 0:
        x = x[abs(start_sig):]
        start_sig = 0
    if width + start_sig > len(y):
        x = x[0:len(y)]
    y[start_sig:end_sig] = 1/(1+np.exp(-x))
    y[end_sig:] = 1.
    if y[0]>0: #move function in y direction so that f(x=0)=0
        y = y - y[0]
    y = y * (1/y[-1]) * max_y #stretch function in y direction so that f(x=max) = max_y
    if imp_id <= 4:
        y[0:20]=0 #values under 20 do not exist in MESHS
    if plot_y:
        plt.plot(y)
    return y

def sigmoid(x, L, x_0, k):
    x_min = x.min()
    y = np.zeros(len(x))
    f_x = np.zeros(len(x))
    i=0
    for i2 in x:
        f_x = max(i2-x_min,0)/(x_0-x_min)
        y[i] = L*((f_x**k)/(1+f_x**k))
        i+=1
    return y
        
def RMSF(X, Y):
    sol = 0
    for i in range(len(X)):
        sol += np.log(Y[i]/X[i])**2
    sol = np.exp((sol/len(X))**(1/2))
    return sol

def make_Y(parameter, *args): # *args = imp_fun_parameter, exp, agr, haz_type
    # a = time.perf_counter()
    parameter_optimize, exp, haz, haz_type, num_fct = args
    ifset_hail = ImpactFuncSet()
    if num_fct ==1:
        parameter_optimize[0]["L"] = parameter[0]
        parameter_optimize[0]["x_0"] = parameter[1]
        parameter_optimize[0]["k"] = parameter[2]
    else:
        parameter_optimize[0]["L"] = parameter[0]
        parameter_optimize[0]["x_0"] = parameter[1]
        parameter_optimize[0]["k"] = parameter[2]
        parameter_optimize[1]["L"] = parameter[3]
        parameter_optimize[1]["x_0"] = parameter[4]
        parameter_optimize[1]["k"] = parameter[5]
        parameter_optimize[2]["L"] = parameter[6]
        parameter_optimize[2]["x_0"] = parameter[7]
        parameter_optimize[2]["k"] = parameter[8]
    # b = time.perf_counter()
    # print("time to write parameter_optimize: ", b-a)
    for imp_fun_dict in parameter_optimize:
        imp_fun = fct.create_impact_func(haz_type, 
                             imp_fun_dict["imp_id"], 
                             imp_fun_dict["L"], 
                             imp_fun_dict["x_0"], 
                             imp_fun_dict["k"])
        ifset_hail.append(imp_fun)
    c  = time.perf_counter()
    # print("time to make imp_fun: ", c-b)
    imp = Impact()
    # imp.calc(self = imp, exposures = exp, impact_funcs = ifset_hail, hazard = haz, save_mat = True)
    imp.calc(exp, ifset_hail, haz, save_mat = True)
    d = time.perf_counter()
    print("time to calc impact: ", d-c)
    Y = list(imp.calc_impact_year_set(year_range = [2002, 2019]).values())
    for count, y in enumerate(Y):
        if y==0:
            Y[count] = 1
            
    Y_norm = np.divide(Y, min(Y))
    O_norm = [2.1122213681783246,3.5465026902382784, 6.200614911606457, 5.90315142198309, 2.5103766333589546, 4.801691006917755, 2.02152190622598, 8.501152959262106, 1.0, 2.6541122213681785, 1.6525749423520368, 5.747117601844734, 1.7524980784012298, 1.5249807840122982, 1.345119139123751, 2.751729438893159, 1.8754803996925442,2.55956956187548]
    
    # res = mean_squared_error(Y_norm, O_norm)**0.5
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Params {}".format(parameter_optimize))
    print("The sum of the new Impact is: {}".format(sum(Y)))
    coef, p_value = spearmanr(O_norm, Y_norm)    
    print("spearman for agr  (score, p_value) = ({}, {})".format(coef, p_value))
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # e= time.perf_counter()
    # print("time to get result: ", e-d)
    return coef*-1
    
# Set exposure: (see tutorial climada_entity_LitPop)
def load_exp_infr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real):
    file = Path(input_folder + "/" + name_hdf5_file["exp_infr"])
    if force_new_hdf5_generation["exp_infr"] or not file.exists(): #be carefull, this step will take ages when you do both at once
        # LitPop Exposure
        print("generating new exp_infr")
        exp_infr = LitPop()
        exp_infr.set_country('Switzerland', reference_year = 2019)
        exp_infr.set_geometry_points()
        exp_infr = exp_infr.rename(columns = {'if_': 'if_HL'})
        exp_infr = Exposures(exp_infr)
        exp_infr.set_lat_lon()
        exp_infr.check()
        exp_infr.assign_centroids(haz_real, method = "NN", distance ="haversine", threshold = 2)
        exp_infr.write_hdf5(input_folder + "/exp_switzerland.hdf5")
    else:
        # LitPop Exposure
        exp_infr= LitPop()
        exp_infr.read_hdf5(input_folder +"/exp_switzerland.hdf5")
        exp_infr.check()
    return exp_infr

def load_exp_agr(force_new_hdf5_generation, name_hdf5_file, input_folder, haz_real):
    file1 = Path(input_folder + "/" + name_hdf5_file["exp_agr"])
    file2 = Path(input_folder + "/" + "exp_agr_no_centr.hdf5")
    if not file2.exists() and not file1.exists():
        print("Please use import_agrar_exposure to create the hdf5 file!" + 
              " and move it to the input folder")
        sys.exit()
    elif force_new_hdf5_generation["exp_agr"]: #be carefull, this step will take ages when you do both at once
        if not file2.exists():
                    print("Please use import_agrar_exposure to create the hdf5 file!" + 
                          " and move it to the input folder")
                    sys.exit()
        exp_agr = Exposures()
        exp_agr.read_hdf5(input_folder + "/exp_agr_no_centr.hdf5")
    
        exp_agr.check()
        exp_agr.assign_centroids(haz_real, method = "NN", distance = "haversine", threshold = 2)
        exp_agr.check()
        exp_agr.write_hdf5(input_folder + "/exp_agr.hdf5")
    
    else:
        #Agrar Exposure    
        exp_agr = Exposures()
        exp_agr.read_hdf5(input_folder + "/exp_agr.hdf5")
        exp_agr.check()
    return exp_agr