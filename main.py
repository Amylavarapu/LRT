import lrt_functions
import lrt_predicate_data
import procode_data

def preds_lrt(k_number, lower_date: str = 'YYYY-MM-DD', upper_date: str = 'YYYY-MM-DD'):
    df = lrt_predicate_data.gen_lrt_df(k_number,lower_date,upper_date)
    lrt_df =  lrt_functions.calculate_lrt(df,'AEID','pma_pmn_number','./S-11')
    return lrt_df

def calculate_predicate_lrt(k_number, lower_date, upper_date):
    '''
    Args:
    ----
    k_number: str, The device that we want to pull predicate information for 
    lower_date: str, The lower date of the time window we want to explore.
    upper_date: str, The upper date of the time window.
    
    ---
    
    Output: pd.Dataframe,
    'AEID', 'k_number', 'is_signal', 'pvalue', 'problem type', 'Device Generation'
    
    '''
    k_number = k_number.upper()
    df = preds_lrt(k_number, lower_date, upper_date)
    df['problem type'] = 'Product Problem'
    df2 = lrt_functions.calculate_lrt(lrt_predicate_data.patient_problem_preds(k_number,lower_date ,upper_date),'AEID','pma_pmn_number','./S-11')
    df2['problem type'] = 'Patient Problem'
    df = df.append(df2, ignore_index=True)
    df = df[['AEID','k_number','is_signal','pvalue','problem type']]        
    final_dict = lrt_predicate_data.generation_count_dictionary(predicate_df, k_number)
    df['Device Generation'] = df.apply(lambda row: final_dict[k_number][str(row['k_number'])], axis=1)
    return df 
  
  
  def calculate_procode_lrt(pro_code, lower_date, upper_date):
      '''
    Args:
    ----
    pro_code: str, The 3 letter product code that we want to explorer
    lower_date: str, The lower date of the time window we want to explore.
    upper_date: str, The upper date of the time window.
    
    ---
    
    Output: pd.Dataframe,
    'AEID', 'k_number', 'is_signal', 'pvalue', 'problem type', 'product code'
    
    '''
    pro_code = pro_code.upper()
    df = procode_data.procode_lrt_df(pro_code, lower_date, upper_date)
    df = lrt_functions.calculate_lrt(df, 'AEID', 'pma_pmn_number', './S-11')
    df['problem type'] = 'Product Problem'
    df2 = lrt_functions.calculate_lrt(procode_data.patient_problem_df(pro_code,lower_date ,upper_date),'AEID','pma_pmn_number','./S-11')
    df2['problem type'] = 'Patient Problem'
    df = df.append(df2, ignore_index=True)
    df = df[['AEID','k_number','is_signal','pvalue','problem type']]
    df['product code'] = pro_code
    return df 
