
function simulate(
    b::N,
    m0::N,
    γₖ::N,
    σ::N,
    kbar::Int64,
    T::Int64
) where {N <: Real} 

    m1 = 2.0 - m0
    γ = Vector{Float64}(undef,kbar)
    M = Matrix{Float64}(undef,kbar,T)
    simulategammas!(γ,γₖ,b,kbar)
    simulatestates!(M,γ)
    data = ones(N,T)
    states = (M[:,1].==1).*m1 .+ (M[:,1].==0).*m0
    simulatedata!(data,states,M,m0,m1,σ,)
    return data
end

function simulategammas!(
    γ::AbstractVector{N},
    γₖ::N,
    b::N,
    kbar::Int64,
) where {N <: Real}
    @views @inbounds for i ∈ eachindex(γ)
        if i == 1
            γ[1] = 1-(1-γₖ)^(1/(b^(kbar-1)))
        else
            γ[i] = 1-(1-γ[1])^(b^(i-1))
        end
    end
end

function simulatestates!(
    M::AbstractMatrix{N},
    γ::AbstractVector{N}
) where {N <: Real}
    @views @inbounds for j ∈ axes(M,1)
        for t ∈ axes(M,2)
            M[j,t] = rand([0,1]) < γ[j] #rand([Binomial(1, γ[j])])
        end
    end
end

function simulatedata!(
    data::AbstractVector{N},
    states::AbstractVector{N},
    M::AbstractMatrix{N},
    m0::N,
    m1::N,
    σ::N,
) where {N <: Real}
    @views @inbounds for t ∈ axes(M,2)
        for i ∈ axes(M,1)
            if t == 1
                data[1] *= states[i]
            else
                if M[i,t] == 1
                    states[i] = rand([m0,m1])
                end
                data[t] *= states[i]
            end
        end
        data[t] ^= (1/2)
        data[t] *= σ*rand(Normal(0,1))
    end
end