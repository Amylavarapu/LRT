import lrt_functions
import lrt_predicate_data
import procode_data

def preds_lrt(k_number, lower_date: str = 'YYYY-MM-DD', upper_date: str = 'YYYY-MM-DD'):
    df = lrt_predicate_data.gen_lrt_df(k_number,lower_date,upper_date)
    lrt_df =  lrt_functions.calculate_lrt(df,'AEID','pma_pmn_number','./S-11')
    return lrt_df

def calculate_predicate_lrt(k_number, lower_date, upper_date):
    k_number = k_number.upper()
    df = lrt_predicate_data.preds_lrt(k_number, lower_date, upper_date)
    df['problem type'] = 'Product Problem'
    df2 = calculate_lrt(lrt_predicate_data.patient_problem_preds(k_number,lower_date ,upper_date),'AEID','pma_pmn_number','./S-11')
    df2['problem type'] = 'Patient Problem'
    df = df.append(df2, ignore_index=True)
    df = df[['AEID','k_number','is_signal','pvalue','problem type']]        
    final_dict = lrt_predicate_data.generation_count_dictionary(predicate_df, k_number)
    df['Device Generation'] = df.apply(lambda row: final_dict[k_number][str(row['k_number'])], axis=1)
    return df 
  
  
  def calculate_procode_lrt(pro_code, lower_date, upper_date):
    pro_code = pro_code.upper()
    df = procode_data.procode_lrt_df(pro_code, lower_date, upper_date)
    df = calculate_lrt(df, 'AEID', 'pma_pmn_number', './S-11')
    df['problem type'] = 'Product Problem'
    df2 = calculate_lrt(procode_data.patient_problem_df(pro_code,lower_date ,upper_date),'AEID','pma_pmn_number','./S-11')
    df2['problem type'] = 'Patient Problem'
    df = df.append(df2, ignore_index=True)
    return df 
