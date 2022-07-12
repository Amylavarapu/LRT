import sys
!{sys.executable} -m pip install PyAthena
from pyathena import connect
conn=connect(s3_staging_dir= ,
            region_name=)

from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import logging

''' This Code Pulls the product problem adverse events for a specific product code in a given time frame '''
def procode_lrt_df(pro_code, lower_date: str = 'YYYY-MM-DD', upper_date: str = 'YYYY-MM-DD'):
    df = pd.read_sql(f'''SELECT pma_pmn_number, date_received, date_of_event, date_report, mdr_report_key, adverse_event_flag, product_problems
FROM "fda-open-database"."event", 
    UNNEST(device) as t(devices)
WHERE devices.device_report_product_code = '%s';'''%(pro_code),conn)
    #df[['date_received','date_of_event','date_report']] = df[['date_received','date_of_event','date_report']].apply(pd.to_datetime,format = '%Y%m%d', errors='ignore')
    df['date_received'] = pd.to_datetime(df['date_received'], errors='ignore',format = '%Y%m%d')
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='ignore',format = '%Y%m%d')
    df['date_report'] = pd.to_datetime(df['date_report'], errors='ignore',format = '%Y%m%d')
    df = df.sort_values(by='date_of_event')
#    df = df[['product_problems','pma_pmn_number']].loc[(df['date_of_event'] > lower_date) & (df['date_of_event'] < upper_date)].reset_index()
    df = df[['product_problems','pma_pmn_number']].loc[(df['date_received'] > lower_date) & (df['date_received'] < upper_date)].reset_index()
    df = df.rename(columns={'product_problems':'AEID'}).drop(['index'], axis = 1)
    df['AEID'] = df['AEID'].str.replace(r'[][]', '', regex=True)
    df['AEID'] = df['AEID'].str.split(',')
    df = df.explode('AEID', ignore_index = True)
    df = df.replace(r"^ +| +$", r"", regex=True)        
    df = data[['AEID','pma_pmn_number']]
    df = df.groupby(['AEID','pma_pmn_number']).size().unstack().fillna(0).reset_index()
    return df
    
''' This Code Pulls the patient problem adverse events for a specific product code in a given time frame '''
def patient_problem_df(pro_code, lower_date: str = 'YYYY-MM-DD', upper_date: str = 'YYYY-MM-DD'):
    df = pd.read_sql(f'''SELECT pma_pmn_number, date_received, date_of_event, date_report, mdr_report_key, adverse_event_flag, patients.patient_problems
FROM "fda-open-database"."event", 
    UNNEST(device) as t(devices),
    UNNEST(patient) as t(patients)
WHERE devices.device_report_product_code = '%s';'''%(pro_code),conn)
    #df[['date_received','date_of_event','date_report']] = df[['date_received','date_of_event','date_report']].apply(pd.to_datetime,format = '%Y%m%d', errors='ignore')
    df['date_received'] = pd.to_datetime(df['date_received'], errors='ignore',format = '%Y%m%d')
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='ignore',format = '%Y%m%d')
    df['date_report'] = pd.to_datetime(df['date_report'], errors='ignore',format = '%Y%m%d')
    df = df.sort_values(by='date_of_event')
    df = df[['patient_problems','pma_pmn_number']].loc[(df['date_of_event'] > lower_date) & (df['date_of_event'] < upper_date)].reset_index()
#    df = df[['patient_problems','pma_pmn_number']].loc[(df['date_received'] > lower_date) & (df['date_received'] < upper_date)].reset_index()
    df = df.rename(columns={'patient_problems':'AEID'}).drop(['index'], axis = 1)
    df['AEID'] = df['AEID'].str.replace(r'[][]', '', regex=True)
    df['AEID'] = df['AEID'].str.split(',')
    df = df.explode('AEID', ignore_index = True)
    df = df.replace(r"^ +| +$", r"", regex=True)
    df = data[['AEID','pma_pmn_number']]
    df = df.groupby(['AEID','pma_pmn_number']).size().unstack().fillna(0).reset_index()
    return df

def flogLRv(x, fnidot, fndotj, fndotdot):
    logLRvr = x*(np.log(x)-np.log(fnidot))
    diff = fndotj - x
    diff[diff == 0] = 0.0001
    logLRvr = logLRvr+diff*(np.log(diff)-np.log(fndotdot-fnidot))
    logLRvr = logLRvr-fndotj*(np.log(fndotj)-np.log(fndotdot))
    logLRvr[logLRvr.isna() ] = 0
    pj=x/fnidot
    qj=(fndotj-x)/(fndotdot-fnidot)
    logLRvr[ pj<qj ] = 0
    return logLRvr

def fpvalue(x, fsim):
    return np.mean(fsim >= x)
   
 def calculate_lrt(
    data: pd.DataFrame,
    row_var: str='imdrf_term',
    col_var: str='brand_name',
    directory: str='./lrt/',
    num_sim: int=10000,
    signal_only: bool=True
):
    """Provides LRT calculation.
    Based on: https://www.kaggle.com/howardyao/d.eming-conference-lrt-presentation/data
    Args:
    ----
    data : pd.DataFrame, should have at least two columns, where
        each row is a separate event. 
        If no `row_var`, `col_var` provided, those columns should 
        be `level_3_term` and `brand_name`.
    row_var : str, default `level_3_term`
        LRT row variable–Adverse events (AE)
    col_var : str, default `brand_name`
        LRT column variable
    directory : str, default `./lrt/
        Directory to store files for later processing
    num_sim: int=1000, number of Monte Carlo simulations
    signal_only: bool=True, return all results or signals only
        
    Output:
    ------
    pd.DataFrame, columns: aeid (row_var), ntrt, nidot, ndotj, ndotdot, rr, std_rr, 
        low_rr, upp_rr, log_lr, pvector, pobs, t_alpha, is_signal, pvalue, col_var
    """
    logging.info(f'=== Received {len(data)} records for LRT calculation. Starting...')

    logging.info(f'=== Calculating number of cases reported for all {row_var} and {col_var} variables')
    #nijs = data.rename(columns={row_var:'AEID'}) # cause that's how R code expects this column to be called
    nijs = data
    
    logging.info(f'=== Calculating marginal total for {col_var} column variable')
    col_vars = nijs.columns[1:]
    #col_vars = list(nijs.columns)[1:]
    ndotjs = []
    for col in col_vars:
        ndotjs.append((col, nijs[col].sum()))
    ndotjs = pd.DataFrame(ndotjs, columns=[col_var, 'ndotj']).sort_values('ndotj', ascending=False)

    logging.info(f'=== Calculating marginal total for {row_var} row variable')
    row_vars = nijs[row_var]
    #row_vars = list(nijs[row_var])
    nidots = []
    for row in row_vars:
        nidots.append((row, nijs[nijs[row_var] == row].sum(axis=1).values[0]))
    nidots = pd.DataFrame(nidots, columns=[row_var, 'nidot']).sort_values('nidot', ascending=False)
    
    logging.info('=== Calculating grand total')
    ndotdot = ndotjs['ndotj'].sum() #+ nidots['nidot'].sum()
    logging.info(f'=== Total events: {ndotdot}')

    logging.info(f'=== Calculating LRT for all {len(col_vars)} variables. Will take awhile...')
    result = pd.DataFrame()
    for col in tqdm(col_vars):
        temp = nijs[[row_var, col]]
        temp = temp.rename(columns={col:'ntrt'})
        temp = temp.merge(nidots, on=row_var)
        ndotj = ndotjs[ndotjs[col_var] == col]['ndotj'].values[0]
        temp['ndotj'] = ndotj
        temp['ndotdot'] = ndotdot
        temp['k_number'] = col

        # calculating LRT
        # RR and 95 CI
        np.seterr(divide='ignore')
        temp['RR'] = (temp.ntrt) * ((temp.ndotdot - temp.nidot)/(temp.nidot*(temp.ndotj-temp.ntrt)))
        temp['stdRR'] = 1/temp.ntrt - 1/temp.nidot + 1/(temp.ndotj-temp.ntrt) - 1/(temp.ndotdot-temp.nidot)
        temp.stdRR = np.sqrt(temp.stdRR)
        temp['lowRR'] = np.log(temp.RR)-1.96*(temp.stdRR)
        temp.lowRR = np.exp(temp.lowRR)
        temp['uppRR'] = np.log(temp.RR)+1.96*(temp.stdRR)
        temp.uppRR = np.exp(temp.uppRR)

        # logLR from input data
        temp['logLR'] = flogLRv(temp.ntrt, temp.nidot, temp.ndotj[0], temp.ndotdot[0])
        temp = temp.sort_values('logLR', ascending=False)

        # MC to compute threshold under H0
        temp['Pvector'] = temp.nidot/temp.ndotdot
        temp['Pobs'] = temp.ntrt/temp.ndotj
        sim_data = np.random.multinomial(temp.ndotj[0], size=num_sim, pvals=temp.Pvector.values).T
        np.place(sim_data,sim_data==0,1)
        # obtain logLR from simulated null data
        #sim_logLR = np.apply_along_axis(
             #flogLRv, 0, sim_data, fnidot=temp.nidot, fndotj=temp.ndotj[0], fndotdot=temp.ndotdot[0])

        # iterate over columns
        fnidot=temp.nidot, 
        fndotj=temp.ndotj[0], 
        fndotdot=temp.ndotdot[0]
        num_cols = sim_data.shape[1]
        col_len = sim_data.shape[0]
        sim_logLR = np.empty((num_cols, col_len)) # Transposed share
        for i in range(num_cols):
            x = sim_data[:,[i]].reshape(col_len)                  
            logLRvr = x*(np.log(x)-np.log(fnidot))
            diff = fndotj - x
            diff[diff == 0] = 0.0001
            logLRvr = logLRvr+diff*(np.log(diff)-np.log(fndotdot-fnidot))
            logLRvr = logLRvr-fndotj*(np.log(fndotj)-np.log(fndotdot))
            logLRvr[np.isnan(logLRvr) ] = 0
            pj=x/fnidot
            qj=(fndotj-x)/(fndotdot-fnidot)
            logLRvr[ pj<qj ] = 0
            sim_logLR[i] = logLRvr
        sim_logLR = sim_logLR.T

                
        # get MLR from 1000 simulated null data 
        sim_maxlogLR = np.apply_along_axis(np.amax, 0, sim_logLR)

        temp['T_alpha'] = np.quantile(sim_maxlogLR, 0.95)
        temp['is_signal'] = temp.logLR > temp.T_alpha

        #if signal_only:
         #   temp = temp[temp['is_signal'] == True]

        temp['pvalue'] = [fpvalue(x, sim_maxlogLR) for x in temp.logLR]

        result = result.append(temp)

    logging.info(f'=== Finished LRT calculation. Resulting table contains {len(result)} rows.')

    return result
