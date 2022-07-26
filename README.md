# LRT
The LRT Function has two methods 

## LRT Using Predicates

calculate_predicate_lrt takes 3 inputs. The k number whose predicates we want to generate LRT alerts for, the lower date of our time window and the upper date of our time window. The output is a datagram with the following schema.


|'AEID'| 'k_number'| 'is_signal'| 'pvalue' | 'problem type' | 'Device Generation'|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |

