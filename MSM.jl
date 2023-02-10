"""
    Julia module to estimate the Markov-Switching Multifractal model of 
    
    Calvet, L. E. and Fisher, A. J. (2004). How to Forecast Long-Run Volatility: Regime Switching
    and the Estimation of Multifractal Processes. Journal of Financial Econometrics, 2(1):49–83.

    Written by Lars Spreng.
"""
module MSM

using MKL
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

end