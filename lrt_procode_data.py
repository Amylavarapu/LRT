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
                        '''
            Args:
            ----- 
            pro_code: str, the product code where we want to generate product problem lrt alerts for
            lower_date: str,
            upper_date: str,
            
            ----
            Output: pd.dataframe, 'AEID', pma_pmn_number
            
            '''
    pro_code = pro_code.upper()
    df = pd.read_sql(f'''SELECT pma_pmn_number, date_received, date_of_event, date_report, mdr_report_key, adverse_event_flag, product_problems, devices.device_report_product_code
FROM "fda-open-database"."event", 
    UNNEST(device) as t(devices)
WHERE devices.device_report_product_code = '{pro_code}' AND date_received >= '{lower_date.replace("-","")}' AND date_received <= '{upper_date.replace('-','')}';''',conn)
    #df[['date_received','date_of_event','date_report']] = df[['date_received','date_of_event','date_report']].apply(pd.to_datetime,format = '%Y%m%d', errors='ignore')
    df['date_received'] = pd.to_datetime(df['date_received'], errors='ignore',format = '%Y%m%d')
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='ignore',format = '%Y%m%d')
    df['date_report'] = pd.to_datetime(df['date_report'], errors='ignore',format = '%Y%m%d')
    df = df.sort_values(by='date_received')
    df = df[['product_problems','pma_pmn_number']]#.loc[(df['date_received'] > lower_date) & (df['date_received'] < upper_date)].reset_index()
    df = df.rename(columns={'product_problems':'AEID'})#.drop(['index'], axis = 1)
    df['AEID'] = df['AEID'].str.replace(r'[][]', '', regex=True)
    df['AEID'] = df['AEID'].str.split(',')
    df = df.explode('AEID', ignore_index = True)
    df = df.replace(r"^ +| +$", r"", regex=True)
    return df
  
  
  ''' This Code Pulls the patient problem adverse events for a specific product code in a given time frame '''




def patient_problem_df(pro_code, lower_date: str = 'YYYY-MM-DD', upper_date: str = 'YYYY-MM-DD'):
                                '''
            Args:
            ----- 
            pro_code: str, the product code where we want to generate patient problem lrt alerts for
            lower_date: str,
            upper_date: str,
            
            ----
            Output: pd.dataframe, 'AEID', pma_pmn_number
            
            '''
    pro_code = pro_code.upper()
    df = pd.read_sql(f'''SELECT pma_pmn_number, date_received, date_of_event, date_report, mdr_report_key, adverse_event_flag, patients.patient_problems
FROM "fda-open-database"."event", 
    UNNEST(device) as t(devices),
    UNNEST(patient) as t(patients)
WHERE devices.device_report_product_code = '{pro_code}' AND date_received >= '{lower_date.replace("-","")}' AND date_received <= '{upper_date.replace('-','')}';''',conn)
    #df[['date_received','date_of_event','date_report']] = df[['date_received','date_of_event','date_report']].apply(pd.to_datetime,format = '%Y%m%d', errors='ignore')
    df['date_received'] = pd.to_datetime(df['date_received'], errors='ignore',format = '%Y%m%d')
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='ignore',format = '%Y%m%d')
    df['date_report'] = pd.to_datetime(df['date_report'], errors='ignore',format = '%Y%m%d')
    df = df.sort_values(by='date_of_event')
    df = df[['patient_problems','pma_pmn_number']]#.loc[(df['date_of_event'] >= lower_date) & (df['date_of_event'] <= upper_date)].reset_index()
#   df = df[['patient_problems','pma_pmn_number']].loc[(df['date_received'] > lower_date) & (df['date_received'] < upper_date)].reset_index()
    df = df.rename(columns={'patient_problems':'AEID'})#.drop(['index'], axis = 1)
    df['AEID'] = df['AEID'].str.replace(r'[][]', '', regex=True)
    df['AEID'] = df['AEID'].str.split(',')
    df = df.explode('AEID', ignore_index = True)
    df = df.replace(r"^ +| +$", r"", regex=True)
    return df
