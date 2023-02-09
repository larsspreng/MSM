function estimate(
    data,
    kbar::Int64,
    n::Int64
)
    θ₀ = gridsearch(data,kbar,n)

    
    lower = [1.0, 1.0, 0.001, 0.0001]; 
    upper = [50.0, 1.99, 0.99999, 5.0];
    make_closures(data,kbar,n) = θ₀ -> likelihood(θ₀,data,kbar,n);
    nll = make_closures(data,kbar,n)   
    #df = TwiceDifferentiable(nll, θ₀; autodiff=:forward);
    #dfc = TwiceDifferentiableConstraints(lower, upper)
    #@btime  result = optimize(df,dfc,θ₀,IPNewton(), Optim.Options(g_tol=1.0e-6)); 
    df = OnceDifferentiable(nll, θ₀; autodiff=:forward);
    result = optimize(df,lower,upper,θ₀,Fminbox(LBFGS()), Optim.Options(g_tol=1.0e-5,iterations=500));

    return Optim.minimizer(result)
end

function estimate(
    data,
    kbar::Int64,
    n::Int64,
    θ₀::Vector{Float64}
)
    if (θ₀[1] < 1)
        error("b must be greater than 1")
    end
    if (θ₀[2] < 1) || (θ₀[2] > 1.99)
        error("m0 must be between (1,1.99]")
    end
    if (θ₀[3] < 0.00001) || (θ₀[3] > 0.99999)
        error("gamma_k be between [0,1]")
    end
    if (θ₀[4] < 0.00001)
        error("sigma must be a positive (non-zero) number")
    end
    if (θ₀[4] > 1)
        println("Sigma value is very large - consider using smaller value")
    end

    make_closures(data,kbar,n) = θ₀ -> likelihood(θ₀,data,kbar,n);
    nll = make_closures(data,kbar,n)
    lower = [1.0, 1.0, 0.001, 0.0001]; 
    upper = [50.0, 1.99, 0.99999, 5.0];   
    #df = TwiceDifferentiable(nll, θ₀; autodiff=:forward);
    #dfc = TwiceDifferentiableConstraints(lx, ux)
    #result = optimize(df,dfc,θ₀,IPNewton()); 
    df = OnceDifferentiable(nll, θ₀; autodiff=:forward);
    result = optimize(df,lower,upper,θ₀,Fminbox(LBFGS()), Optim.Options(g_tol=1.0e-5,iterations=500));

    return Optim.minimizer(result)
end

function gridsearch(data,kbar::Int64,n::Int64)
    
    index = 1;
    σ = std(data)*sqrt(252/n);
    output = Vector{Vector{Float64}}(undef,84)
    ll = Vector{Float64}(undef,84)
    @views @inbounds for b in [1.5, 3.0, 6.0, 20.0]        
            for γₖ in [0.1, 0.5, 0.9]
                for m0 in 1.2:0.1:1.8
                    input = [b,m0,γₖ,σ];
                    tmp = likelihood(input,data,kbar,n)
                    output[index] = input;
                    ll[index] = tmp;
                    index=index+1;
                end
            end
        end
    return output[findmin(ll)[2]]
end

function likelihood(
    input::AbstractVector{N},
    data,
    kbar::Int64,
    n::Int64
) where {N <: Real} 

    b = input[1] |> copy
    m0 = input[2] |> copy
    γₖ = input[3] |> copy
    σ = input[4]/sqrt(252/n)

    kbar2 = 2^kbar;
    T = size(data,1);                       

    A = transition_mat(γₖ,b,kbar,kbar2); # Compute state transition matrix
    g_m = ones(N,kbar2) #Vector{Float64}(undef,kbar2)
    gofm(g_m,m0,kbar); # Compute all possible states 
    wt = Matrix{N}(undef,kbar2,T);
    get_weights!(wt,data,g_m,σ);
    
    pi_mat = Matrix{N}(undef,kbar2,T+1);     
    pi_mat[:,1] .= (1/kbar2);
    ll = get_loglik!(pi_mat,A,wt)
    return -ll
end

function get_loglik!(
    pi_mat::AbstractMatrix{N},
    A::AbstractMatrix{N},
    wt::AbstractMatrix{N},
) where {N <: Real} 

    ll = zero(N);
    piA = zeros(N,size(wt,1));
    @views @inbounds for t in axes(wt,2)          
        mul!(piA,A,pi_mat[:,t]);
        pi_mat[:,t+1] .= wt[:,t].*piA; 
        ft = sum(pi_mat[:,t+1]);
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
    wt::AbstractMatrix{N},
    data,
    g_m::AbstractVector{N},
    σ::N,
) where {N <: Real} 

    pa = (2.0*pi)^-0.5;
    @views @inbounds for i in axes(wt,2)
        for j in axes(wt,1)
            wt[j,i] = pa*exp(- 0.5*( (data[i]/(σ*g_m[j]))^2 ))/(σ*g_m[j]) + 1e-16;
        end
    end
end

function transition_mat(
    γₖ::N,
    b::N,
    kbar::Int64,
    kbar2::Int64
) where {N <: Real} 

    A = zeros(N,kbar2,kbar2);
    transmat_template(A,kbar);
    
    γ = Matrix{N}(undef,2,kbar);                          
    get_gammas!(γ,γₖ,b,kbar) 

    prob = ones(N,kbar2);    
    get_probs!(prob,γ,kbar)
    get_transmat!(A,prob,kbar,kbar2)
    return A
end

function transmat_template(
    A::AbstractMatrix{N},
    kbar::Int64
) where {N <: Real}  
    @views @inbounds for i in 0:(2^kbar-1)       
        for j = i:(2^kbar-1)-i  
            A[i+1,j+1] = xor(i,j);
        end
    end
    return A
end

function get_gammas!(
    γ::AbstractMatrix{N},
    γₖ::N,
    b::N,
    kbar::Int64,
) where {N <: Real} 
    
    @views @inbounds for i in axes(γ,2)
        if i == 1
            γ[1,1] = 1.0-(1.0-γₖ)^(1.0/(b^(kbar-1)));
            γ[2,1] = (1.0-(1.0-γₖ)^(1.0/(b^(kbar-1))))*0.5;
        else
            γ[1,i] = 1.0-(1.0-(1.0-γ[1,1])^(b^(i-1)))*0.5;
            γ[2,i] = (1.0-(1.0-γ[1,1])^(b^(i-1)))*0.5;
        end
    end
    γ[1,1] *= (-0.5);
    γ[1,1] += 1.0
end

function get_probs!(
    prob::AbstractVector{N},
    γ::AbstractMatrix{N},
    kbar::Int64,
) where {N <: Real} 

    @views @inbounds @fastmath for i in eachindex(prob)   
        for m = 1:kbar  
            prob[i] *= γ[(bit(i-1,m)+1),kbar+1-m];
        end
    end
end

function get_transmat!(
    A::AbstractMatrix{N},
    prob::AbstractVector{N},
    kbar::Int64,
    kbar2::Int64,
) where {N <: Real} 
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

function gofm(
    g_m::AbstractVector{N},
    m0::N,
    kbar::Int64
) where {N <: Real} 
    @views @inbounds for i in eachindex(g_m)
       g = 1.0
        for j in 0:(kbar-1)       
            if ((i-1) & (2^j)) != 0   
                g *= (2.0 - m0);
            else 
                g *= m0;
            end
        end
        g_m[i] = sqrt(g);
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
    @inbounds @fastmath for i ∈ eachindex(a,b)
        s += a[i] * b[i]
    end
    s
end

function arma_estimate(
    data,
    kbar::Int64,
    n::Int64
)
    θ₀ = arma_gridsearch(data,kbar,n)

    lower = [-50.0, -1.0, -1.0, 1.0, 1.0, 0.001, 0.0001]; 
    upper = [50.0, 1.0, 1.0, 50.0, 1.99, 0.99999, 5.0];
    θ₀[:] .= max.(θ₀, lower.+0.001)
    θ₀[:] .= min.(θ₀, upper.-0.001)

    make_closures(data,kbar,n) = θ₀ -> arma_likelihood(θ₀,data,kbar,n);
    nll = make_closures(data,kbar,n)   
    df = TwiceDifferentiable(nll, θ₀; autodiff=:forward);
    dfc = TwiceDifferentiableConstraints(lower, upper)
    result = optimize(df,dfc,θ₀,IPNewton(), Optim.Options(g_tol=1.0e-6)); 
    # df = OnceDifferentiable(nll, θ₀; autodiff=:forward);
    # result = optimize(df,lower,upper,θ₀,Fminbox(LBFGS()), Optim.Options(g_tol=1.0e-5,iterations=500));

    return Optim.minimizer(result)
end

function arma_estimate(
    data,
    kbar::Int64,
    n::Int64,
    θ₀::Vector{Float64}
)

    lower = [-50.0, -1.0, -1.0, 1.0, 1.0, 0.001, 0.0001]; 
    upper = [50.0, 1.0, 1.0, 50.0, 1.99, 0.99999, 5.0];
    θ₀[:] .= max.(θ₀, lower.+0.001)
    θ₀[:] .= min.(θ₀, upper.-0.001)
    if (abs(θ₀[1]) > 1)
        println("AR constant is very large - consider using smaller value")
    end
    if (abs(θ₀[2]) > 1)
        error("AR coefficient must be smaller than 1")
    end
    if (abs(θ₀[3]) > 1)
        println("MA coefficient is very large - consider smaller value")
    end
    if (θ₀[4] < 1)
        error("b is "*string(θ₀[4])*" but must be greater than 1")
    end
    if (θ₀[5] < 1) || (θ₀[5] > 1.99)
        error("m0 is "*string(θ₀[5])*" but must be between (1,1.99]")
    end
    if (θ₀[6] < 0.00001) || (θ₀[6] > 0.99999)
        error("gamma_k is "*string(θ₀[6])*" but must be between [0,1]")
    end
    if (θ₀[7] < 0.00001)
        error("sigma is "*string(θ₀[7])*" but must be a positive (non-zero) number")
    end
    if (θ₀[7] > 1)
        println("Sigma value is very large - consider using smaller value")
    end

    make_closures(data,kbar,n) = θ₀ -> arma_likelihood(θ₀,data,kbar,n);
    nll = make_closures(data,kbar,n)   
    df = TwiceDifferentiable(nll, θ₀; autodiff=:forward);
    dfc = TwiceDifferentiableConstraints(lower, upper)
    result = optimize(df,dfc,θ₀,IPNewton(), Optim.Options(g_tol=1.0e-6)); 
    # df = OnceDifferentiable(nll, θ₀; autodiff=:forward);
    # result = optimize(df,lower,upper,θ₀,Fminbox(LBFGS()));

    return Optim.minimizer(result)
end

function arma_gridsearch(data,kbar::Int64,n::Int64)
    
    index = 1;
    σ = std(data)*sqrt(252/n);
    output = Vector{Vector{Float64}}(undef,84)
    β = data[1:end-1] \ data[2:end]
    ll = Vector{Float64}(undef,84)
    @views @inbounds for b in [1.5, 3.0, 6.0, 20.0]        
                    for γₖ in [0.1, 0.5, 0.9]
                        for m0 in 1.2:0.1:1.8
                            input = [0.0,β,0.0,b,m0,γₖ,σ];
                            tmp = arma_likelihood(input,data,kbar,n)
                            output[index] = input;
                            ll[index] = tmp;
                            index=index+1;
                        end
                    end
                end
    return output[findmin(ll)[2]]
end

function arma_likelihood(
    input::AbstractVector{N},
    data,
    kbar::Int64,
    n::Int64
) where {N <: Real} 

    T = size(data,1);
    c = input[1] |> copy
    β = input[2] |> copy
    ϕ = input[3] |> copy
    b = input[4] |> copy
    m0 = input[5] |> copy
    γₖ = input[6] |> copy
    σ = input[7]/sqrt(252/n)

    μ = zeros(N,T)
    kbar2 = 2^kbar;   
    
    arma(μ,c,β,ϕ,data)
    ret = data .- μ

    A = transition_mat(γₖ,b,kbar,kbar2); # Compute state transition matrix
    g_m = ones(N,kbar2) #Vector{Float64}(undef,kbar2)
    gofm(g_m,m0,kbar); # Compute all possible states 
    wt = Matrix{N}(undef,kbar2,T);
    get_weights!(wt,ret,g_m,σ);
    
    pi_mat = Matrix{N}(undef,kbar2,T+1);     
    pi_mat[:,1] .= (1/kbar2);
    ll = get_loglik!(pi_mat,A,wt)
    
    return -ll
end


function arma(
    μ::AbstractVector{N},
    c::N,
    β::N,
    ϕ::N,
    data,
) where {N <: Real}    
    @views @inbounds for t ∈ 2:length(data)
        if t == 2
            μ[t-1] = c 
        end
        μ[t] = c + β*data[t-1] + ϕ*(data[t-1] - μ[t-1]); 
    end
end