# Markov-Switching Multifractal Model

This Julia module implements the Markov-Switching Multifractal model of Calvet, L. E. and Fisher, A. J. (2004). How to Forecast Long-Run Volatility: Regime Switching
and the Estimation of Multifractal Processes. _Journal of Financial Econometrics_, 2(1):49-83.

The code can be used to run the module on a time-series of returns, as well as on the residuals of an ARMA(1,1) process. 

## Example
Load the module
```julia
using MSM
```
Simulate a return series by setting the parameters of the process. The number of states is given by $\bar{k}$
``` julia
kbar = 5 
```
The parameter controling the binomial distribution from which the states are drawn is $m_0 \in (0,2]$
``` julia
m0 = 1.6
```
The state transition probabilities are given by $\gamma_k = 1 - (1 - \gamma_0)^b$, where $\gamma_0 \in (0,1)$ and $b \in (1,\infty)$
``` julia
b = 1.1 
γ₀ = 0.5
```
Finally, $\sigma>0$ is a positive constant in the volatility process 
``` julia
σ = 2.0/sqrt(252)
```
The number of time periods is given by
``` julia
T = 2000
```
To simulate the process, we can use the function
``` julia
data = MSM.simulate(b,m0,γ₀,σ,kbar,T)
```
and to estimate the parameters
``` julia
ψ = MSM.estimate(data,kbar,scale)
```