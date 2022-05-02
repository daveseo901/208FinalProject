import numpy as np
import pandas as pd
from datetime import datetime
import re

def dp_histogram_2d(data, a, b, epsilon, k):
    bin_lims = np.arange(a, b + (b - a) / k, (b - a) / k)
    data = np.clip(data, a, b)
    sens_hist = np.histogram2d(data.T[0], data.T[1], bins=(bin_lims, bin_lims))[0]
    p = 1 - np.exp(-epsilon / (2 * (k ** 2)))
    noise = np.random.geometric(p, size=sens_hist.shape) - np.random.geometric(p, size=sens_hist.shape)
    return (np.clip(sens_hist + noise, 0, len(data)), bin_lims)

def choose_ind_2d(probs):
    ind = np.random.choice(np.arange(probs.size), p=probs.ravel())
    return np.unravel_index(ind, probs.shape)

def dp_synthetic_data_2d(data, a, b, epsilon, k, m):
    dp_hist, bin_lims = dp_histogram_2d(data, a, b, epsilon, k)
    dp_hist /= np.sum(dp_hist)
    inds = [choose_ind_2d(dp_hist) for i in range(m)]
    step = (b - a) / k
    vals = [[np.random.uniform(a + ix * step, a + (ix + 1) * step), np.random.uniform(a + iy * step, a + (iy + 1) * step)] for (ix, iy) in inds]
    return vals

def dp_histogram(data, a, b, epsilon, k):
    bin_lims = np.arange(a, b + (b - a) / k, (b - a) / k)
    data = np.clip(data, a, b)
    sens_hist = np.histogram(data, bins=bin_lims)[0]
    print(sens_hist)
    p = 1 - np.exp(-epsilon / (2 * (k ** 2)))
    noise = np.random.geometric(p, size=len(sens_hist)) - np.random.geometric(p, size=len(sens_hist))
    return (np.array(np.clip(sens_hist + noise, 0, len(data)), dtype="float"), bin_lims)

def choose_ind(probs):
    return np.random.choice(np.arange(probs.size), p=probs)

def dp_synthetic_data(data, a, b, epsilon, k, m):
    dp_hist, bin_lims = dp_histogram(data, a, b, epsilon, k)
    dp_hist /= np.sum(dp_hist)
    inds = [choose_ind(dp_hist) for i in range(m)]
    step = (b - a) / k
    vals = [np.random.uniform(a + ix * step, a + (ix + 1) * step) for ix in inds]
    return vals

def dp_synthetic_data_discrete(data, a, b, epsilon, k, m):
    dp_hist, bin_lims = dp_histogram(data, a, b, epsilon, k)
    dp_hist /= np.sum(dp_hist)
    inds = [choose_ind(dp_hist) for i in range(m)]
    step = (b - a) / k
    vals = [a + ix for ix in inds]
    return vals

def date_to_int(date):
    return pd.to_datetime(date).timetuple().tm_yday

def coords_to_list(coords: str):
    return np.array(re.split(', ', coords.replace("(", "").replace(")", "")), dtype="float")

def count_laplace(data, column, value, epsilon, b):
    true_count = len(data[data[column] == value])
    return np.clip(true_count + np.random.laplace(0, 1 / epsilon), 0, b)

def count_gaussian(data, column, value, epsilon, delta, b):
    true_count = len(data[data[column] == value])
    sigma = (1 / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
    return np.clip(true_count + np.random.normal(0, sigma), 0, b)

def count_synthetic(data, column, value, epsilon, a, b, k, m):
    if column == "date":
        selected = list(map(date_to_int, data["date"]))
        synthetic_data = dp_synthetic_data_discrete(selected, a, b, epsilon, k, m)
        return synthetic_data.count(date_to_int(value))
    #synthetic_data = dp_synthetic_data(selected, a, b, epsilon, k, m)
    return synthetic_data.count(value)
    
def mean_loc_exact(data):
    lat = []
    long = []
    for lat_long in data["location"]:
        coords = coords_to_list(lat_long)
        lat.append(coords[0])
        long.append(coords[1])
    true_lat_mean = np.mean(lat)
    true_long_mean = np.mean(long)
    return (true_lat_mean, true_long_mean)
    
def mean_loc_laplace(data, epsilon, a, b):
    n = len(data["location"])
    param = (2 * (b - a)) / (n * epsilon)
    lat = []
    long = []
    for lat_long in data["location"]:
        coords = coords_to_list(lat_long)
        lat.append(coords[0])
        long.append(coords[1])
    true_lat_mean = np.mean(lat)
    true_long_mean = np.mean(long)
    return (np.clip(true_lat_mean + np.random.laplace(param), a, b), np.clip(true_long_mean + np.random.laplace(param), a, b))
    
def mean_loc_gaussian(data, epsilon, delta, a, b):
    n = len(data["location"])
    param = (2 * (b - a)) / (n * epsilon) * np.sqrt(2 * np.log(1.25 / delta))
    lat = []
    long = []
    for lat_long in data["location"]:
        coords = coords_to_list(lat_long)
        lat.append(coords[0])
        long.append(coords[1])
    true_lat_mean = np.mean(lat)
    true_long_mean = np.mean(long)
    return (np.clip(true_lat_mean + np.random.laplace(param), a, b), np.clip(true_long_mean + np.random.laplace(param), a, b))

def mean_loc_laplace_exper(data, epsilon, a, b, true_mean):
    n = len(data["location"])
    param = (2 * (b - a)) / (n * epsilon)
    true_lat_mean = true_mean[0]
    true_long_mean = true_mean[1]
    return (np.clip(true_lat_mean + np.random.laplace(param), a, b), np.clip(true_long_mean + np.random.laplace(param), a, b))
    
def mean_loc_gaussian_exper(data, epsilon, delta, a, b, true_mean):
    n = len(data["location"])
    param = (2 * (b - a)) / (n * epsilon) * np.sqrt(2 * np.log(1.25 / delta))
    true_lat_mean = true_mean[0]
    true_long_mean = true_mean[1]
    return (np.clip(true_lat_mean + np.random.laplace(param), a, b), np.clip(true_long_mean + np.random.laplace(param), a, b))