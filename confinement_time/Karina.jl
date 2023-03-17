include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/pre_defined_variables.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/extending_functions.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/General_preprocessing.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/H_mode_confinement.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/Scaling.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/Compare.jl")

# Databases

## DB2P8, DB5

DB2P8_karina = H_mode_data("/Users/joe/.data/Multi_Machine_Fusion_Data_ITER_21_03_01/DB52P3_ed_!S.csv", :DB2P8)
SELDB5_karina =  H_mode_data("/Users/joe/.data/Multi_Machine_Fusion_Data_ITER_21_03_01/DB52P3_ed_!S.csv", :SELDB5) 


for (field, field_name) in Iterators.zip([:DB2P8, :SELDB5], [:DB2P8, :DB5])
    params = unique(vcat([:id], regression_dimensionless, regression_parameters, IPB98_interesting_parameters, parameters_dimensionless))
    CSV.write("/Users/joe/Project/PhD/Intern/EP_Masters_Students_2022/2022_2023/confinement_time/data/$(field_name).csv", getfield(eval(field*"_karina"), :modified)[!, params])
end

## Reintroduced points

DB2P8_SELDB5_comp_karina = Compare(DB2P8, SELDB5)
ids = DataFrame(:id => DB2P8_SELDB5_comp_karina.unique_df2.id)
CSV.write("/Users/joe/Project/PhD/Intern/EP_Masters_Students_2022/2022_2023/confinement_time/data/new_point_ids.csv", ids)