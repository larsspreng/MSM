

function volatility(input,data,kbar;h=0,window="fixed")
    
    b = input[1] |> copy
    m0 = input[2] |> copy
    γₖ = input[3] |> copy
    σ = input[4]/sqrt(252)

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
        return pi_mat[:,end]*A^h; # Predict volatility
    elseif window == "fixed" 
        vol = Vector{Float64}(undef,T)
        filterstates!(vol,pi_mat,g_m,A,wt,σ,h);
        return vol
    end 
end

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
            vol[t] = σ*dot_tturbo(g_m,pi_mat[:,t+1])
        elseif h > 0
            vol[t] = σ*dot_tturbo(A^h*pi_mat[:,t],g_m)
        end
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
