
"""
    arma_volatility(input,data,n,kbar;h=0,window="fixed",)

    Function to compute volatility of ARMA(1,1) residuals of return series:
    (a) in-sample volatility if h=0 and window = "fixed"
    (b) volatility forecasts if h>0 either
        - using a rolling window by setting window = "rolling"
        - using a fixed window by setting window = "fixed"
"""
function arma_volatility(input,data,n,kbar;h=0,window="fixed",)

    T = size(data,1);
    c = input[1] |> copy
    β = input[2] |> copy
    ϕ = input[3] |> copy
    b = input[4] |> copy
    m0 = input[5] |> copy
    γₖ = input[6] |> copy
    σ = input[7]/sqrt(252/n)

    μ = zeros(T)
    kbar2 = 2^kbar;   
    
    arma(μ,c,β,ϕ,data)
    ret = data .- μ

    A = transition_mat(γₖ,b,kbar,kbar2); # Compute state transition matrix
    g_m = ones(kbar2) 
    gofm(g_m,m0,kbar); # Compute all possible volatility states 
    wt = Matrix{Float64}(undef,kbar2,T);
    get_weights!(wt,data,g_m,σ); # Obtain weights

    pi_mat = Matrix{Float64}(undef,kbar2,T+1);     
    pi_mat[:,1] .= (1/kbar2);
     # Compute model implied volatility
    if window == "rolling"
        filterstates!(pi_mat,A,wt);
        return sqrt(σ^2*dot_tturbo(A*pi_mat[:,end],g_m.^2)); # Predict volatility
    elseif window == "fixed" 
        vol = Vector{Float64}(undef,T)
        filterstates!(vol,pi_mat,g_m,A,wt,σ,h);
        return vol
    end 
end

"""
    arma_returns(input,data,n,kbar;h=0,window="fixed",)

    Function to compute volatility of return series:
    (a) in-sample returns if h=0 and window = "fixed"
    (b) return forecasts if h>0 either
        - using a rolling window by setting window = "rolling"
        - using a fixed window by setting window = "fixed"
"""
function arma_returns(input,data,n,kbar;h=0,window="fixed",)

    T = size(data,1);
    c = input[1] |> copy
    β = input[2] |> copy
    ϕ = input[3] |> copy
    b = input[4] |> copy
    m0 = input[5] |> copy
    γₖ = input[6] |> copy
    σ = input[7]/sqrt(252/n)

    μ = zeros(T)
    #kbar2 = 2^kbar;   
    if h == 0
        arma(μ,c,β,ϕ,data)
        return data .- μ
    elseif h > 0
        if window == "rolling"
            arma(μ,c,β,ϕ,data)
            #ret = data .- μ
            return c + β*data[end] + ϕ*(data[end] - μ[end]);  # Predict returns
        elseif window == "fixed"  
            arma(μ,c,β,ϕ,data)
            return μ
        end 
    end
end

"""
    volatility(input,data,n,kbar;h=0,window="fixed",)

Function to:
    (a) compute in-sample volatility if h=0 and window = "fixed"
    (b) compute volatility forecasts if h>0 either
        - using a rolling window by setting window = "rolling"
        - using a fixed window by setting window = "fixed"
"""
function volatility(input,data,n,kbar;h=0,window="fixed",)
    
    b = input[1] |> copy
    m0 = input[2] |> copy
    γₖ = input[3] |> copy
    σ = input[4]/sqrt(252/n)
    kbar2 = 2^kbar;
    T = size(data,1);                       

    A = transition_mat(γₖ,b,kbar,kbar2); # Compute state transition matrix
    g_m = ones(kbar2) 
    gofm(g_m,m0,kbar); # Compute all possible volatility states 
    wt = Matrix{Float64}(undef,kbar2,T);
    get_weights!(wt,data,g_m,σ); # Obtain weights

    pi_mat = Matrix{Float64}(undef,kbar2,T+1);     
    pi_mat[:,1] .= (1/kbar2);
     # Compute model implied volatility
    if window == "rolling"
        filterstates!(pi_mat,A,wt);
        return sqrt(σ^2*dot_tturbo(A*pi_mat[:,end],g_m.^2)); # Predict volatility
    elseif window == "fixed" 
        vol = Vector{Float64}(undef,T)
        filterstates!(vol,pi_mat,g_m,A,wt,σ,h)
        return vol
    end 
end

"""
    filterstates!(
    vol,
    pi_mat,
    g_m,
    A,
    wt,
    σ,
    h
)

Function to compute states
"""
function filterstates!(
    vol,
    pi_mat,
    g_m,
    A,
    wt,
    σ,
    h
)
    piA = similar(wt[:,1])
    @views @inbounds for t in axes(wt,2)          
        mul!(piA,A,pi_mat[:,t]);
        pi_mat[:,t+1] .= wt[:,t].*piA; 
        ft = sum_tturbo(pi_mat[:,t+1]);
        if ft == 0                    
            pi_mat[1,t+1] = 1.0;   
        else
            pi_mat[:,t+1] ./= ft; 
        end
        if h == 0
            vol[t] = sqrt(σ^2*dot_tturbo(g_m.^2,pi_mat[:,t+1]))
        elseif h > 0
            vol[t] = sqrt(σ^2*dot_tturbo(A^h*pi_mat[:,t],g_m.^2))
        end
    end
end

""" 
    arma(
    μ::AbstractVector{N},
    c::N,
    β::N,
    ϕ::N,
    data,
    h::Int64,
) where {N <: Real}

Function to compute ARMA(1,1) process for mean of returns.
"""
function arma(
    μ::AbstractVector{N},
    c::N,
    β::N,
    ϕ::N,
    data,
    h::Int64,
) where {N <: Real}    

    T = length(data)
    @views @inbounds for t ∈ 2:T+h
        if t == 2
            μ[t-1] = c 
        end
        μ[t] = c + β*data[t-1] + ϕ*(data[t-1] - μ[t-1]); 
    end
end

"""
    filterstates!(
    pi_mat,
    A,
    wt,
)

Function to compute transition matrices without computing volatility
"""
function filterstates!(
    pi_mat,
    A,
    wt,
)
    piA = similar(wt[:,1])
    @views @inbounds for t in axes(wt,2)          
        mul!(piA,A,pi_mat[:,t]);
        pi_mat[:,t+1] .= wt[:,t].*piA; 
        ft = sum_tturbo(pi_mat[:,t+1]);
        if ft == 0                    
            pi_mat[1,t+1] = 1.0;   
        else
            pi_mat[:,t+1] ./= ft; 
        end
    end
end