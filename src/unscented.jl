"""
    sample mean
"""
function sample_mean(X)
    N = length(X)
    n = length(X[1])
    μ = sum(X) ./ N
    return μ
end

"""
    sample covariance
"""
function sample_covariance(X, β)
    N = length(X)
    μ = sample_mean(X)
    P = sum([(X[i] - μ) * (X[i] - μ)' for i = 1:N]) ./ (2.0 * β^2.0)
    return P
end

"""
    deterministic sampling
"""
function resample(X, β)
    n = length(X[1])
    μ = sample_mean(X)
    P = sample_covariance(X, β)
    cols = sqrt(P)
    Xs = [μ + s * β * cols[:,i] for s in [-1.0, 1.0] for i = 1:n]
    return Xs
end

function resample_vec(X, n, N, β)
    _X = [X[(i - 1) * n .+ (1:n)] for i = 1:N]
    return vcat(resample(_X, β)...)
end

function resample_vec(X, n, N, β, k)
    _X = [X[(i - 1) * n .+ (1:n)] for i = 1:N]
    Xs = resample(_X, β)
    return Xs[k]
end

function resample(μ, P, β::T) where T
    n = length(μ)
    cols = sqrt(P)
    Xs = [μ + s * β * cols[:,i] for s in [-1.0, 1.0] for i = 1:n]
    return Xs
end
