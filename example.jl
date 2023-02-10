push!(LOAD_PATH, ".")
using MSM


using DelimitedFiles 
using DataFrames
using CSV
using Dates
using TimeSeries
using BenchmarkTools
using NLSolversBase
using Statistics
using StatsBase
using Distributions
using PlotlyJS
kbar = 5
b = 6.0
m0 = 1.6
γₖ = 0.5
σ = 2.0/sqrt(252)
T = 2000
returns = MSM.simulate(b,m0,γₖ,σ,kbar,T);
param = MSM.estimate(returns,kbar,1)
writedlm("output.txt", returns)

returns = CSV.File("data.txt") |> DataFrame

param = MSM.estimate(returns.Value,kbar,1)

starting = [3.0,
1.23286040068112,
0.11,
0.193608607674193,]
MSM.gridsearch(returns,kbar,1)

γ = Matrix{Float64}(undef,2,kbar); 
MSM.get_gammas!(
    γ,
    γₖ,
    b,
    kbar,
)

MSM.transition_mat(
    γₖ,
    b,
    kbar,
    2^kbar)
g_m = ones(2^kbar)
MSM.gofm(
g_m,
m0,
kbar)