push!(LOAD_PATH, ".")
using MSM

kbar = 5
b = 1.0
m0 = 1.6
γₖ = 0.5
σ = 2.0/sqrt(252)
T = 2000
scale = 1
data = MSM.simulate(b,m0,γₖ,σ,kbar,T);
param = MSM.estimate(data,kbar,scale)

