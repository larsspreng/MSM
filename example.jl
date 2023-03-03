push!(LOAD_PATH, ".")
using MSM

kbar = 5
b = 1.1
m0 = 1.6
γ₀ = 0.5
σ = 2.0/sqrt(252)
T = 2000
scale = 1
data = MSM.simulate(b,m0,γ₀,σ,kbar,T);
ψ = MSM.estimate(data,kbar,scale)

