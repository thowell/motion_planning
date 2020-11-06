"""
    sample mean
"""
function sample_mean(X, β)
    N = length(X)
    n = length(X[1])
    μ = β * sum(X) ./ N
    return μ
end

"""
    sample covariance
"""
function sample_covariance(X, β, γ)
    N = length(X)
    μ = sample_mean(X, β)
    P = γ * sum([(X[i] - μ) * (X[i] - μ)' for i = 1:N]) ./ N
    return P
end

"""
    deterministic sampling
"""
function resample(X, α::T, β::T, γ::T) where T
    n = length(X[1])
    μ = sample_mean(X, β)
    P = sample_covariance(X, β, γ)
    cols = sqrt(P)
    Xs = [μ + s * α * cols[:,i] for s in [-1.0, 1.0] for i = 1:n]
    return Xs
end

function resample_vec(X, n, N, k, α, β, γ)
    _X = [X[(i - 1) * n .+ (1:n)] for i = 1:N]
    Xs = resample(_X, α, β, γ)
    return Xs[k]
end

function resample(μ, P, α::T) where T
    n = length(μ)
    cols = sqrt(P)
    Xs = [μ + s * α * cols[:,i] for s in [-1.0, 1.0] for i = 1:n]
    return Xs
end
