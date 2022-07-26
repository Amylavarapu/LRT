# LRT
## LRT for any Dataframe
The lrt fucntion can take any dataframe as an input that is in the following Schema.
|Feature| Description|
|----|----|
|AEID| The code for the adverse event or keyword|
|Device| This can be the brand name or submission number|
The output follows this Schema

|Feature| Description|
|----|----|
|aeid (row_var)| The code for the adverse event or keyword|
|ntrt ||
|nidot ||
|ndotj|| 
|ndotdot|| 
|rr ||
|std_rr|| 
|low_rr || 
|upp_rr|| 
|log_lr|| 
|pvector|| 
|pobs|| 
|t_alpha||
|is_signal||
|pvalue||
|Device (col_var)|This can be the brand name or submission number|

## LRT Using Predicates
calculate_predicate_lrt takes 3 inputs. The k number whose predicates we want to generate LRT alerts for, the lower date of our time window and the upper date of our time window. The output is a datagram with the following schema.

|Feature| Description|
|----|----|
|AEID| This the code for the Adverse event |
|k_number ||
|is_signal ||
|pvalue|| 
|problem type|| 
|Device Generation||

## LRT using Product Code
calculate_procode_lrt takes 3 inputs. The product code we want to generate LRT alerts for, the lower date of our time window and the upper date of our time window. The output is a datagram with the following schema.

|Feature| Description|
|----|----|
|AEID| This the code for the Adverse event |
|k_number ||
|is_signal ||
|pvalue|| 
|problem type|| 
|Product Code||
