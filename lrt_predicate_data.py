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

''' Predicate LRT Functions '''




predicate_df = pd.read_sql(f'''SELECT * FROM "anvesh_db"."source_target_predicates"; ''',conn)




def sub_tree(k_number):
'''
Args

----
k_number: str, the k number for the device that we want to pull predicate devices from

----
Output: list, list containing the input k number and the predicate devices

'''
    k_list = []
    k_list.append(k_number)
    for n in k_list:
        preds = predicate_df.loc[predicate_df.k_number == n].predicates.unique()
        for value in preds:
            if value in k_list:
                pass
            else:
                k_list.append(value)
    #if len(k_list) == 1:
        #return predicate_df.loc[predicate_df['predicates'] == k_number]
    #else:
        #return k_list
    return k_list

def generation_count_dictionary(dataframe: pd.Dataframe = predicate_df,k_number):
            '''
            Args:
            
            ----
            dataframe: default = predicate, this is the source target dataframe
            k_number: the k_number that we want to explore it's predicates
            ----
            Output: 
            dictionary which will provde generation count and distance between input device and preidcates
            ''''
    predicate_edge_list = dataframe.loc[dataframe['k_number'].isin(sub_tree(k_number))].to_records(index=False)
    predicate_edge_list = list(predicate_edge_list)
    predicate_edge_list2 = dataframe.loc[dataframe['predicates'].isin(sub_tree(k_number))].to_records(index=False)
    predicate_edge_list2 = list(predicate_edge_list2)
    final_list = predicate_edge_list + predicate_edge_list2
    G = nx.Graph()
    G.add_edges_from(final_list)
    test = dict(nx.all_pairs_shortest_path_length(G))
    return test


def gen_lrt_df(k_number, lower_date: str = 'YYYY-MM-DD', upper_date: str = 'YYYY-MM-DD'):
            '''
            Args:
            ----- 
            k_nunmber: str, the device whose predicates we want to generate product problem lrt alerts for
            lower_date: str,
            upper_date: str,
            
            ----
            Output: pd.dataframe, 'AEID', pma_pmn_number
            
            '''
    df = pd.read_sql(f'''SELECT pma_pmn_number, date_received, date_of_event, date_report, mdr_report_key, adverse_event_flag, product_problems FROM "fda-open-database"."event" WHERE pma_pmn_number IN{tuple(sub_tree(k_number))} AND date_received >= '{lower_date.replace("-","")}' AND date_received <= '{upper_date.replace('-','')}';''',conn)
    #df[['date_received','date_of_event','date_report']] = df[['date_received','date_of_event','date_report']].apply(pd.to_datetime,format = '%Y%m%d', errors='ignore')
    df['date_received'] = pd.to_datetime(df['date_received'], errors='ignore',format = '%Y%m%d')
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='ignore',format = '%Y%m%d')
    df['date_report'] = pd.to_datetime(df['date_report'], errors='ignore',format = '%Y%m%d')
    df = df.sort_values(by='date_of_event')
    df = df[['product_problems','pma_pmn_number']]#.loc[(df['date_report'] >= lower_date) & (df['date_report'] <= upper_date)].reset_index()
    #df = df[['product_problems','pma_pmn_number']].loc[(df['date_of_event'] > lower_date) & (df['date_of_event'] < upper_date)].reset_index()
    df = df.rename(columns={'product_problems':'AEID'})#.drop(['index'], axis = 1)
    df['AEID'] = df['AEID'].str.replace(r'[][]', '', regex=True)
    df['AEID'] = df['AEID'].str.split(',')
    df = df.explode('AEID', ignore_index = True)
    df = df.replace(r"^ +| +$", r"", regex=True)
    return df
  
def patient_problem_preds(k_number, lower_date: str = 'YYYY-MM-DD', upper_date: str = 'YYYY-MM-DD'):
                '''
            Args:
            ----- 
            k_nunmber: str, the device whose predicates we want to generate patient problem lrt alerts for
            lower_date: str,
            upper_date: str,
            
            ----
            Output: pd.dataframe, 'AEID', pma_pmn_number
            
            '''
    df = pd.read_sql(f'''SELECT pma_pmn_number, date_received, date_of_event, date_report, mdr_report_key, adverse_event_flag, patients.patient_problems
FROM "fda-open-database"."event", 
    UNNEST(device) as t(devices),
    UNNEST(patient) as t(patients) WHERE pma_pmn_number IN{tuple(sub_tree(k_number))} AND date_received >= '{lower_date.replace("-","")}' AND date_received <= '{upper_date.replace('-','')}';''',conn)
    #df[['date_received','date_of_event','date_report']] = df[['date_received','date_of_event','date_report']].apply(pd.to_datetime,format = '%Y%m%d', errors='ignore')
    df['date_received'] = pd.to_datetime(df['date_received'], errors='ignore',format = '%Y%m%d')
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='ignore',format = '%Y%m%d')
    df['date_report'] = pd.to_datetime(df['date_report'], errors='ignore',format = '%Y%m%d')
    df = df.sort_values(by='date_of_event')
    df = df[['patient_problems','pma_pmn_number']]#.loc[(df['date_report'] >= lower_date) & (df['date_report'] <= upper_date)].reset_index()    
    #df = df[['patient_problems','pma_pmn_number']].loc[(df['date_of_event'] > lower_date) & (df['date_of_event'] < upper_date)].reset_index()
    df = df.rename(columns={'patient_problems':'AEID'})#.drop(['index'], axis = 1)
    df['AEID'] = df['AEID'].str.replace(r'[][]', '', regex=True)
    df['AEID'] = df['AEID'].str.split(',')
    df = df.explode('AEID', ignore_index = True)
    df = df.replace(r"^ +| +$", r"", regex=True)
    return df
