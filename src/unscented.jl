"""
    sample mean
"""
function sample_mean(x)
    N = length(x)
    n = length(x[1])
    μ = sum(x) ./ N
    return μ
end

"""
    sample covariance
"""
function sample_covariance(x, β)
    N = length(x)
    μ = sample_mean(x)
    P = sum([(x[i] - μ) * (x[i] - μ)' for i = 1:N]) ./ (2.0 * β^2.0)
    return P
end

"""
    deterministic sampling
"""
function resample(x, β)
    n = length(x[1])
    μ = sample_mean(x)
    P = sample_covariance(x, β)
    cols = Array(sqrt(P))
    xs = [μ + s * β * cols[:,i] for s in [-1.0, 1.0] for i = 1:n]
    return xs
end

function resample_vec(x, n, N, β)
    _x = [x[(i - 1) * n .+ (1:n)] for i = 1:N]
    return vcat(resample(_x, β)...)
end

function resample_vec(x, n, N, β, k)
    _x = [x[(i - 1) * n .+ (1:n)] for i = 1:N]
    xs = resample(_x, β)
    return xs[k]
end

function resample(μ, P, β::T) where T
    n = length(μ)
    cols = Array(sqrt(P))
    xs = [μ + s * β * cols[:,i] for s in [-1.0, 1.0] for i = 1:n]
    return xs
end
