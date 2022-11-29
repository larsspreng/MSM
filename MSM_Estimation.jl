
using MKL
using DelimitedFiles 
using DataFrames
using CSV
using Dates
using TimeSeries
using LinearAlgebra 
using BenchmarkTools
using Bits
data = vec(Array(DataFrame(CSV.File("data.csv"))));

kbar=3;

function estimate(
    msm::MSMStructure,

)
    
end


function likelihood(
    input,
    kbar::Int64,
    data,
)
    
    sigma = input[4]/sqrt(252);
    kbar2 = 2^kbar;
    kbar1 = kbar + 1;
    T = size(data,1);                       

    A = transition_mat(input,kbar,kbar1,kbar2); # Compute state transition matrix
    g_m = Vector{Float64}(undef,kbar2)
    gofm(g_m,input,kbar,kbar2); # Compute all possible states 
    wt = Matrix{Float64}(undef,kbar2,T);
    @btime get_weights!(wt,data,g_m,sigma,T,kbar2);
    
    pi_mat = Matrix{Float64}(undef,kbar2,T+1);     
    pi_mat[:,1] .= (1/kbar2);
    @btime get_loglik!(pi_mat,A,wt,T)
    
end

function get_loglik!(
    pi_mat,
    A,
    wt,
    T,
)
    ll = 0.0;
    @views for t=2:T+1          
        piA = A*pi_mat[:,t-1];
        C = wt[:,t-1].*piA; 
        ft = simplesum(C);
        if ft == 0                    
            pi_mat[1,t] = 1.0;   
        else
            pi_mat[:,t] = C./ft; 
        end
        
        ll += log(wt[:,t-1]*piA);
    end
end

function get_weights!(
    wt,
    data,
    g_m,
    sigma,
    T,
    kbar2
)
    pa = (2*pi)^-0.5;
    @views @inbounds for i in 1:T 
        for j in 1:kbar2
            wt[j,i] = pa*exp(- 0.5*( (data[i]/(sigma*g_m[j]))^2 ))/(sigma*g_m[j]) + 1e-16;
        end
    end
end

function transmat_template(A,kbar::Int64)  
    @views @inbounds for i in 0:(2^kbar-1)       
        for j = i:(2^kbar-1)-i  
            A[i+1,j+1] = xor(i,j);
        end
    end
    return A
end

function transition_mat(input,kbar,kbar1,kbar2)
    A = zeros((2^kbar),(2^kbar));
    transmat_template(A,kbar);
    b = input[1];
    gamma_kbar = input[3];
    
    gamma = Matrix{Float64}(undef,2,kbar);                          
    get_gammas!(gamma,gamma_kbar,b,kbar,) 
    prob = ones(kbar2);    
    get_probs!(prob,gamma,kbar,kbar1)
    get_transmat!(A,prob,kbar,kbar2,)
    return A
end

function get_gammas!(
    gamma,
    gamma_kbar,
    b,
    kbar,
)
    gamma[1,1] = 1-(1-gamma_kbar)^(1/(b^(kbar-1)));
    gamma[2,1] = (1-(1-gamma_kbar)^(1/(b^(kbar-1))))*0.5;
    @views for i in 2:kbar
        gamma[1,i] = 1-(1-(1-gamma[1,1])^(b^(i-1)))*0.5;
        gamma[2,i] = (1-(1-gamma[1,1])^(b^(i-1)))*0.5;
    end
    gamma[1,1] *= (-0.5);
    gamma[1,1] += 1.0
end

function get_probs!(
    prob,
    gamma,
    kbar,
    kbar1
)
    @views for i=0:2^kbar-1   
        for m = 1:kbar  
            prob[i+1] *= gamma[(bit(i,m)+1),kbar1-m];
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

function gofm(g_m,input,kbar,kbar2)
    m0 = input[2]; 
    m1 = 2.0-m0;
    g_m1 = collect(0:(kbar2-1));
    @views for i in 1:(kbar2)
        g=1.0;
        for j in 0:(kbar-1)       
            if g_m1[i] == 2^j   
                g *= m1;
            else 
                g *= m0;
            end
        end
        g_m[i] = sqrt(g);
    end
end

function simplesum(x::AbstractArray)   
    out = 0.0
    @views @inbounds for i in axes(x,1)
        for j in axes(x,2)
            out += x[i,j]
        end
    end
    return out
end