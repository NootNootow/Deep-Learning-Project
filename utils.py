import pandas as pd
import numpy as np
import os
import geoio

def process_malawi():
    lsms_dir = os.path.join(COUNTRIES_DIR, 'malawi_2016', 'LSMS')
    consumption_file = 'IHS4 Consumption Aggregate.csv'
    consumption_ph_col = 'rexpagg' # per household
    hhsize_col = 'hhsize' # people in household

    geovariables_file = 'HouseholdGeovariables_csv/HouseholdGeovariablesIHS4.csv'
    lat_col = 'lat_modified'
    lon_col = 'lon_modified'

    # purchasing power parity for malawi in 2016 (https://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=MW)
    ppp = 215.182
    
    for file in [consumption_file, geovariables_file]:
        assert os.path.isfile(os.path.join(lsms_dir, file)), print(f'Could not find {file}')
    
    df = pd.read_csv(os.path.join(lsms_dir, consumption_file))
    df['cons_ph'] = df[consumption_ph_col]
    df['pph'] = df[hhsize_col]
    df['cons_ph'] = df['cons_ph'] / ppp / 365
    df = df[['case_id', 'cons_ph', 'pph']]

    df_geo = pd.read_csv(os.path.join(lsms_dir, geovariables_file))
    df_cords = df_geo[['case_id', 'HHID', lat_col, lon_col]]
    df_cords.rename(columns={lat_col: 'cluster_lat', lon_col: 'cluster_lon'}, inplace=True)
    df_combined = pd.merge(df, df_cords, on='case_id')
    df_combined.drop(['case_id', 'HHID'], axis=1, inplace=True)
    df_combined.dropna(inplace=True) # can't use na values
    
    df_clusters = df_combined.groupby(['cluster_lat', 'cluster_lon']).sum().reset_index()
    df_clusters['cons_pc'] = df_clusters['cons_ph'] / df_clusters['pph'] # divides total cluster income by people
    df_clusters['country'] = 'mw'
    return df_clusters[['country', 'cluster_lat', 'cluster_lon', 'cons_pc']]

def process_ethiopia():
    lsms_dir = os.path.join(COUNTRIES_DIR, 'ethiopia_2015', 'LSMS')
    consumption_file = 'Consumption Aggregate/cons_agg_w3.csv'
    consumption_pc_col = 'total_cons_ann' # per capita
    hhsize_col = 'hh_size' # people in household

    geovariables_file = 'Geovariables/ETH_HouseholdGeovars_y3.csv'
    lat_col = 'lat_dd_mod'
    lon_col = 'lon_dd_mod'

    # purchasing power parity for ethiopia in 2015 (https://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=ET)
    ppp = 7.882
    
    for file in [consumption_file, geovariables_file]:
        assert os.path.isfile(os.path.join(lsms_dir, file)), print(f'Could not find {file}')
    
    df = pd.read_csv(os.path.join(lsms_dir, consumption_file))
    df['cons_ph'] = df[consumption_pc_col] * df[hhsize_col]
    df['pph'] = df[hhsize_col]
    df['cons_ph'] = df['cons_ph'] / ppp / 365
    df = df[['household_id2', 'cons_ph', 'pph']]

    df_geo = pd.read_csv(os.path.join(lsms_dir, geovariables_file))
    df_cords = df_geo[['household_id2', lat_col, lon_col]]
    df_cords.rename(columns={lat_col: 'cluster_lat', lon_col: 'cluster_lon'}, inplace=True)
    df_combined = pd.merge(df, df_cords, on='household_id2')
    df_combined.drop(['household_id2'], axis=1, inplace=True)
    df_combined.dropna(inplace=True) # can't use na values
    
    df_clusters = df_combined.groupby(['cluster_lat', 'cluster_lon']).sum().reset_index()
    df_clusters['cons_pc'] = df_clusters['cons_ph'] / df_clusters['pph'] # divides total cluster income by people
    df_clusters['country'] = 'eth'
    return df_clusters[['country', 'cluster_lat', 'cluster_lon', 'cons_pc']]

def process_nigeria():
    lsms_dir = os.path.join(COUNTRIES_DIR, 'nigeria_2015', 'LSMS')
    consumption_file = 'cons_agg_wave3_visit1.csv'
    consumption_pc_col = 'totcons' # per capita
    hhsize_col = 'hhsize' # people in household

    geovariables_file = 'nga_householdgeovars_y3.csv'
    lat_col = 'LAT_DD_MOD'
    lon_col = 'LON_DD_MOD'

    # purchasing power parity for nigeria in 2015 (https://data.worldbank.org/indicator/PA.NUS.PRVT.PP?locations=NG)
    ppp = 95.255
    
    for file in [consumption_file, geovariables_file]:
        assert os.path.isfile(os.path.join(lsms_dir, file)), print(f'Could not find {file}')
    
    df = pd.read_csv(os.path.join(lsms_dir, consumption_file))
    df['cons_ph'] = df[consumption_pc_col] * df[hhsize_col]
    df['pph'] = df[hhsize_col]
    df['cons_ph'] = df['cons_ph'] / ppp / 365
    df = df[['hhid', 'cons_ph', 'pph']]

    df_geo = pd.read_csv(os.path.join(lsms_dir, geovariables_file))
    df_cords = df_geo[['hhid', lat_col, lon_col]]
    df_cords.rename(columns={lat_col: 'cluster_lat', lon_col: 'cluster_lon'}, inplace=True)
    df_combined = pd.merge(df, df_cords, on='hhid')
    df_combined.drop(['hhid'], axis=1, inplace=True)
    df_combined.dropna(inplace=True) # can't use na values
    
    df_clusters = df_combined.groupby(['cluster_lat', 'cluster_lon']).sum().reset_index()
    df_clusters['cons_pc'] = df_clusters['cons_ph'] / df_clusters['pph'] # divides total cluster income by people
    df_clusters['country'] = 'ng'
    return df_clusters[['country', 'cluster_lat', 'cluster_lon', 'cons_pc']]