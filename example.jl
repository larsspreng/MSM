#push!(LOAD_PATH, ".")
#using MSM
using BenchmarkTools
using LinearAlgebra 
using Bits
using LoopVectorization
using Optim
using Statistics
using FiniteDifferences
using StatsBase
using Distributions
include("MSM_Estimation.jl")
include("MSM_Prediction.jl")
include("MSM_Simulate.jl")
kbar = 5
b = 1.0
m0 = 1.6
γₖ = 0.5
σ = 2.0/sqrt(252)
T = 2000
scale = 1
data = simulate(b,m0,γₖ,σ,kbar,T);
param = MSM.estimate(returns,kbar,scale)

