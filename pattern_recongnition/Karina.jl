using Pkg
Pkg.activate("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src")

using ScikitLearn,
    PyCall,
    DataFrames,
    CSV,
    DataFramesMeta,
    Statistics,
    GLM,
    Random,
    StatsBase,
    LinearAlgebra,
    StatsPlots,
    Measures,
    Combinatorics,
    .Threads,
    ThreadTools, 
    Distributions,
    ColorSchemes,
    Dates,
    SparseArrays,
    Graphs,
    KernelDensity,
    Clustering,
    MultivariateStats,
    Statistics,
    LinearAlgebra,
    LinRegOutliers,
    MLJ,
    MLJScikitLearnInterface,
    MLJFlux,
    PrettyPrint,
    ProgressBars,
    OrderedCollections

@sk_import linear_model: Ridge

abstract type workflow end

include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/pre_defined_variables.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/extending_functions.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/General_preprocessing.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/H_mode_confinement.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/Scaling.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/Compare.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/Deviation.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/Secondry.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/Plotting.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/Power_regression.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/influential_points.jl")
include("/Users/joe/Project/Coding/J_Major_Radius_22_06_08/src/optimising.jl")

DB2P8S = H_mode_data("/Users/joe/.data/Multi_Machine_Fusion_Data_ITER_21_03_01/DB52P3_ed.csv", :DB2P8)
SELDB5S =  H_mode_data("/Users/joe/.data/Multi_Machine_Fusion_Data_ITER_21_03_01/DB52P3_ed.csv", :SELDB5) 

DB2P8_SELDB5_compS = Compare(DB2P8S, SELDB5S)

DB2P8_SELDB5_introS = deviation(DB2P8_SELDB5_compS, [:common_dataset, :unique_df2])

abstract type deviation_optimized end

####################################################################################################
# structs
####################################################################################################

mutable struct non_optimized_data <: deviation_optimized
    original::DataFrame
    deviation::DataFrame
end

struct non_optimized_deviation <: deviation_optimized
    log_space::non_optimized_data
    original_space::non_optimized_data
end

mutable struct optimized_data_ <: deviation_optimized
    original::DataFrame
    deviation::DataFrame
end

struct optimized_deviation_ <: deviation_optimized
    log_space::optimized_data_
    original_space::optimized_data_
end

struct complete_space_ <: deviation_optimized
    ELMy_ELMy::optimized_deviation_
    ITER_ELMy::optimized_deviation_
    ELMy_ITER::optimized_deviation_
    ITER_ITER::optimized_deviation_
end

####################################################################################################
# functions
####################################################################################################

function original_plus_dev(D::deviation_optimized)
    dev = @subset D.deviation @byrow in(:deviation, ["dev"])
    return reduce(vcat, [D.original, dev], cols=:union)
end

function original_plus_no_dev(D::deviation_optimized)
    dev = @subset D.deviation @byrow in(:deviation, ["no_dev"])
    return reduce(vcat, [D.original, dev], cols=:union)
end

function cardinality_metadata(D::deviation_optimized; name::Symbol=:cardinality)
    no_of_points = [DataFrame() for _ in 1:2]
    df = D.deviation

    for (n, dev) in enumerate(["dev", "no_dev"])
        DF = (@subset df @byrow in(:deviation, [dev]))
        gbdf, key = grouping(DF, :TOK)

        for id in key
            ℓ = size(gbdf[id], 1)
            df_new = DataFrame(:TOK => id[1], name*:_*dev => ℓ)
            no_of_points[n] = vcat(no_of_points[n], df_new)
        end
        DF = DataFrame(:TOK => String7("Total"), name*:_*dev => sum(no_of_points[n][!, name*:_*dev]))
        no_of_points[n] = vcat(no_of_points[n], DF)
    end
    return outerjoin(no_of_points..., on=:TOK) |> sort
end

function correlation_metadata(data::DataFrame; kwargs...)
    parameters = names(data)
    ℓ = length(parameters)

    reduced = data |> Array
    data = StatsBase.standardize(UnitRangeTransform, reduced, dims=1)

    correlation = cor(data |> Array)
    df = DataFrame(round.(correlation; kwargs...), parameters)
    insertcols!(df, 1, :_ => parameters)
    return df
end

function correlation_metadata(D::deviation_optimized, dev::String="baseline"; parameters::Vector=regression_predictor, kwargs...)

    if in(dev, ["baseline", "ordinary"])
        data = D.ordinary
    elseif in(dev, ["dev"])
        data = original_plus_dev(D)
    elseif in(dev, ["no_dev"])
        data = original_plus_no_dev(D)
    else
        error("dev must be either 'dev', 'no_dev' or 'baseline'")
    end

    reduced = data[!, parameters] |> Array
    data = StatsBase.standardize(UnitRangeTransform, reduced, dims=1)

    correlation = cor(data)

    df = DataFrame(round.(correlation; kwargs...), parameters)
    insertcols!(df, 1, :_ => parameters)
    return df
end

function correlation_metadata(D::deviation_optimized; parameters::Vector=regression_predictor, kwargs...)

    ℓ = length(parameters)

    reduced_1 = original_plus_dev(D)[!, parameters] |> dropmissing |> Array 
    data_1 = StatsBase.standardize(UnitRangeTransform, reduced_1, dims=1)

    reduced_2 = original_plus_no_dev(D)[!, parameters] |> dropmissing |> Array
    data_2 = StatsBase.standardize(UnitRangeTransform, reduced_2, dims=1)

    C1 = cor(data_1)
    C2 = cor(data_2)

    correlation = [i < j ? C1[i, j] : C2[i, j] for i in 1:ℓ, j in 1:ℓ]

    df = DataFrame(round.(correlation; kwargs...), parameters)
    insertcols!(df, 1, :_ => parameters)
    return df
end

begin
    opt!dev_all = Dict()
    IDS_all = Dict()
    for df in [:DB2P8_SELDB5_introS]
        data = eval(df)
        sorted_by_αR = sort(data.powerset_regressions, :αR)
        translated = sorted_by_αR.αR .- data.original_regression.αR
        IDS_all[df] = sorted_by_αR[findall(i -> i < 0, translated), :id_added]  
        IDS_all_no_dev = setdiff(sorted_by_αR.id_added, IDS_all[df])
        
        DCreintroLog = deepcopy(data.reintroduced)
        @transform!(DCreintroLog, @byrow :deviation = in(:id, IDS_all[df]) ? "dev" : "no_dev")
        log_sp = non_optimized_data(data.original, DCreintroLog)

        DCreintroOri = deepcopy(data.comparison.df2.csv)
        @transform!(DCreintroOri, @byrow :deviation = in(:id, IDS_all[df]) ? "dev" : in(:id, IDS_all_no_dev) ? "no_dev" : "baseline")
        original_sp = non_optimized_data((@subset DCreintroOri @byrow in(:deviation, ["baseline"])), (@subset DCreintroOri @byrow in(:deviation, ["dev", "no_dev"])))

        opt!dev_all[df] = non_optimized_deviation(log_sp, original_sp)
        
        # dir = joinpath(data_dir[1], "deviation/naive_all/$(directory(data))")
        # mkpath(dir)
        # CSV.write(joinpath(dir, "optimum_deviation.csv"), opt!dev_all[df].log_space.deviation)
        # CSV.write(joinpath(dir, "original_space_optimum_deviation.csv"), vcat(opt!dev_all[df].original_space.deviation))
    end
end
opt!dev_all[:DB2P8_SELDB5_introS]

# cummulating datapoints which individually reduce αR

begin
    dev_results = Dict()
    no_dev_results = Dict()

    for df in [:DB2P8_SELDB5_introS]
        data = eval(df)
        ids = IDS_all[df]
        ℓ = length(ids)

        dev_results[df] = DataFrame()
        no_dev_results[df] = DataFrame()

        dev = deepcopy(data.original)
        no_dev = deepcopy(vcat(data.original, data.reintroduced))

        for ID in ProgressBar(ids)
            append!(dev, filter(:id => ==(ID), no_dev))
            filter!(:id => !=(ID), no_dev)

            append!(dev_results[df], ols(dev, IPB98()) |> write_table)
            append!(no_dev_results[df], ols(no_dev, IPB98()) |> write_table)
        end
    end
end

# The 1st N points (starting from those that individually reduce αR the most) until the subset minimizes αR

begin
    opt!dev_min = Dict()
    IDS_min = Dict()

    for df in [:DB2P8_SELDB5_introS]
        data = eval(df)
        ℓ = findmin(dev_results[df].αR)[2]
        IDS_min[df] = IDS_all[df][1:ℓ]
        IDS_min_no_dev = IDS_all[df][ℓ+1:end]

        DCreintroLog = deepcopy(data.reintroduced)
        @transform!(DCreintroLog, @byrow :deviation = in(:id, IDS_min[df]) ? "dev" : "no_dev")
        log_sp = non_optimized_data(data.original, DCreintroLog)

        DCreintroOri = deepcopy(data.comparison.df2.csv)
        @transform!(DCreintroOri, @byrow :deviation = in(:id, IDS_min[df]) ? "dev" : in(:id, IDS_min_no_dev) ? "no_dev" : "baseline")
        original_sp = non_optimized_data((@subset DCreintroOri @byrow in(:deviation, ["baseline"])), (@subset DCreintroOri @byrow in(:deviation, ["dev", "no_dev"])))

        opt!dev_min[df] = non_optimized_deviation(log_sp, original_sp)

        # dir = joinpath(data_dir[1], "deviation/naive_min/$(directory(data))")
        # mkpath(dir)
        # CSV.write(joinpath(dir, "optimum_deviation.csv"), opt!dev_all[df].log_space.deviation)
        # CSV.write(joinpath(dir, "original_space_optimum_deviation.csv"), vcat(opt!dev_all[df].original_space.deviation))
    end
end

# Find the subset which gets αR no dev back to the original

begin
    opt!dev_N0 = Dict()
    IDS_N0 = Dict()

    for df in [:DB2P8_SELDB5_introS]
        data = eval(df)
        ℓ = findmin(abs.(no_dev_results[df].αR .- data.original_regression[1, :αR]))[2]
        IDS_N0[df] = IDS_all[df][1:ℓ]
        IDS_N0_no_dev = IDS_all[df][ℓ+1:end]

        DCreintroLog = deepcopy(data.reintroduced)
        @transform!(DCreintroLog, @byrow :deviation = in(:id, IDS_N0[df]) ? "dev" : "no_dev")
        log_sp = non_optimized_data(data.original, DCreintroLog)

        DCreintroOri = deepcopy(data.comparison.df2.csv)
        @transform!(DCreintroOri, @byrow :deviation = in(:id, IDS_N0[df]) ? "dev" : in(:id, IDS_N0_no_dev) ? "no_dev" : "baseline")
        original_sp = non_optimized_data((@subset DCreintroOri @byrow in(:deviation, ["baseline"])), (@subset DCreintroOri @byrow in(:deviation, ["dev", "no_dev"])))

        opt!dev_N0[df] = non_optimized_deviation(log_sp, original_sp)

        # dir = joinpath(data_dir[1], "deviation/naive_N0/$(directory(data))")
        # mkpath(dir)
        # CSV.write(joinpath(dir, "optimum_deviation.csv"), opt!dev_all[df].log_space.deviation)
        # CSV.write(joinpath(dir, "original_space_optimum_deviation.csv"), vcat(opt!dev_all[df].original_space.deviation))
    end
end
begin
    # P = plot()
    DF = dev_results[:DB2P8_SELDB5_introS]
    DF.x .= 1:N
    plot(DF.x, DF.αR, label=false, lc=:black, lw=2, xaxis=("no. of points", 0:500:3500), yaxis=("αR", 1:0.2:2.2), grid=false)
    # N = size(DF, 1)

    # scatter!((i, DF.αR[i]), label=false)
    # P
end

DB2P8_SELDB5_introS.deviation_plots[:TIME]