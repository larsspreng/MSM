
using MKL
using DelimitedFiles 
using DataFrames
using CSV
using Dates
using TimeSeries
using LinearAlgebra 
using BenchmarkTools
using Bits
using LoopVectorization
data = vec(Array(DataFrame(CSV.File("data.csv"))));

kbar=3;

function estimate(
    msm::MSMStructure,

)
    @btime likelihood(input,kbar,data)
end


function likelihood(
    input,
    kbar::Int64,
    data,
)
    
    σ = input[4]/sqrt(252);
    kbar2 = 2^kbar;
    T = size(data,1);                       

    A = transition_mat(input,kbar,kbar2); # Compute state transition matrix
    g_m = ones(kbar2) #Vector{Float64}(undef,kbar2)
    gofm(g_m,input,kbar); # Compute all possible states 
    wt = Matrix{Float64}(undef,kbar2,T);
    get_weights!(wt,data,g_m,σ);
    
    pi_mat = Matrix{Float64}(undef,kbar2,T+1);     
    pi_mat[:,1] .= (1/kbar2);
    ll = get_loglik!(pi_mat,A,wt)
    return -ll
end

function get_loglik!(
    pi_mat,
    A,
    wt,
)
    ll = 0.0;
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
        
        ll += log(dot_tturbo(wt[:,t],piA));
    end
    return ll
end

function get_weights!(
    wt,
    data,
    g_m,
    σ,
)
    pa = (2.0*pi)^-0.5;
    @views @inbounds @fastmath for i in axes(wt,2)
        for j in axes(wt,1)
            wt[j,i] = pa*exp(- 0.5*( (data[i]/(σ*g_m[j]))^2 ))/(σ*g_m[j]) + 1e-16;
        end
    end
end

function transition_mat(input,kbar::Int64,kbar2::Int64)
    A = zeros((kbar2),(kbar2));
    transmat_template(A,kbar);
    
    gamma = Matrix{Float64}(undef,2,kbar);                          
    get_gammas!(gamma,input,kbar,) 

    prob = ones(kbar2);    
    get_probs!(prob,gamma,kbar)
    get_transmat!(A,prob,kbar,kbar2)
    return A
end

function transmat_template(A::Matrix{Float64},kbar::Int64)  
    @views @inbounds for i in 0:(2^kbar-1)       
        for j = i:(2^kbar-1)-i  
            A[i+1,j+1] = xor(i,j);
        end
    end
    return A
end

function get_gammas!(
    γ,
    input,
    kbar,
)
    
    @views @inbounds for i in axes(γ,2)
        if i == 1
            γ[1,1] = 1.0-(1.0-input[3])^(1.0/(input[1]^(kbar-1)));
            γ[2,1] = (1.0-(1.0-input[3])^(1.0/(input[1]^(kbar-1))))*0.5;
        else
            γ[1,i] = 1.0-(1.0-(1.0-γ[1,1])^(input[1]^(i-1)))*0.5;
            γ[2,i] = (1.0-(1.0-γ[1,1])^(input[1]^(i-1)))*0.5;
        end
    end
    γ[1,1] *= (-0.5);
    γ[1,1] += 1.0
end

function get_probs!(
    prob,
    gamma,
    kbar,
)
    @views @inbounds @fastmath for i in eachindex(prob)   
        for m = 1:kbar  
            prob[i] *= gamma[(bit(i-1,m)+1),kbar+1-m];
        end
    end
end

function get_transmat!(
    A,
    prob,
    kbar,
    kbar2,
)
    @views @inbounds for i in 0:2^(kbar-1)-1 
        for j in i:(2^(kbar-1)-1)  
            A[kbar2-i,j+1] = prob[kbar2-Int(A[i+1,j+1])]; 
            A[kbar2-j,i+1] = A[kbar2-i,j+1];
            A[j+1,kbar2-i] = A[kbar2-i,j+1];
            A[i+1,kbar2-j] = A[kbar2-i,j+1];    
            A[i+1,j+1] = prob[Int(A[i+1,j+1]+1)];
            A[j+1,i+1] = A[i+1,j+1];
            A[kbar2-j,kbar2-i] = A[i+1,j+1];
            A[kbar2-i,kbar2-j] = A[i+1,j+1];
        end
    end 
end

function gofm(g_m,input,kbar)
    @views @inbounds for i in eachindex(g_m)
        for j in 0:(kbar-1)       
            if i-1 == 2^j   
                g_m[i] *= (2.0 - input[2]);
            else 
                g_m[i] *= input[2];
            end
        end
        g_m[i] ^= 0.5;
    end
end

function sum_tturbo(x::AbstractArray{T}) where {T <: Real} 
    s = zero(T)
    @tturbo for j ∈ axes(x,2)
            for i ∈ axes(x,1)
            s += x[i,j]
        end
    end
    s
end

function dot_tturbo(a::AbstractArray{T}, b::AbstractArray{T}) where {T <: Real}
    s = zero(T)
    @tturbo for i ∈ eachindex(a,b)
        s += a[i] * b[i]
    end
    s
end