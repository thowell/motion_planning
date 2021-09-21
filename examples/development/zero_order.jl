using LinearAlgebra
using ForwardDiff
using Plots
using Distributions

# Bundled Gradients through Contact via Randomized Smoothing

function gd(x0; iters=1000, verbose=true, visualize=true)
    x = copy(x0)
    α = 0.001

    for i = 1:iters
        grad = ∇f(x)
        x -= α * grad
    end

    if verbose
        println("grad. norm: ", norm(∇f(x)))
    end

    if visualize
        x_range = range(-1.0, stop = 1.0, length = 1000)
        plt = plot(x_range, f.(x_range), label="", color=:black, width=2.0)
        plt = scatter!([x0,], [f(x0),], color=:red, label="")
        plt = scatter!([x,], [f(x),], color=:green, label="")
        display(plt)
    end

    return x
end

function gd(x0; iters=1000, verbose=true, visualize=true,
    mode=:gradient,
    μ=0.0, σ=[0.001], σ_schedule=1000, N=100,
    xmin=-1.0, xmax=1.0)
    x = copy(x0)
    α = 0.001

    for s in σ
        dist = Normal(μ, s)
        for i = 1:iters
            if mode == :gradient
                grad = ∇f(x)
            elseif mode == :smooth_gradient
                grad = sum([∇f(x + rand(dist, 1)[1]) for j = 1:N]) / N
            elseif mode == :zero_order_gradient
                fj = []
                for j = 1:N
                    ϵ = rand(dist, 1)[1]
                    # push!(fj, (f(x + ϵ) - f(x)) / (1.0 * ϵ))
                    push!(fj, (f(x + ϵ) - f(x - ϵ)) / (2.0 * ϵ))
                    grad = sum(fj) / N
                end
            else
                nothing
            end
            x -= α * grad
        end
    end

    if verbose
        println("grad. norm: ", norm(∇f(x)))
    end

    if visualize
        x_range = range(xmin, stop = xmax, length = 1000)
        plt = plot(x_range, f.(x_range), label="", color=:black, width=2.0)
        for s in σ
            dist = Normal(μ, s)
            if mode != :gradient
                plt = plot!(x_range, [sum([f(xt + rand(dist, 1)[1]) for j = 1:1000]) / 1000  for xt in x_range],
                    label="σ=$s", width=1.0, legend=:top)
            end
        end
        plt = scatter!([x0,], [f(x0),], color=:red, label="")
        plt = scatter!([x,], [f(x),], color=:green, label="")
        display(plt)
    end

    return x
end

# example 1
f(x) = x^2.0 + 0.1 * sin(20.0 * x)
∇f(x) = 2.0 * x + 2.0 * cos(20.0 * x)

# initial point
x0 = -1.0

# gradient descent
x_sol = gd(x0, iters=1000,
           σ=[0.5, 0.25, 0.1, 0.01, 0.001],
           mode=:zero_order_gradient)

# example 2
function f(x)
    if x >= 0.0
        return 1.0
    else
        return 0.0
    end
end
∇f(x) = 0.0

# initial point
x0 = 1.0

# gradient descent
x_sol = gd(x0, iters=10000,
           σ=[1.0, 0.5, 0.1],#, 0.25, 0.1, 0.01, 0.001],#, 0.075, 0.05, 0.01],
           N = 100,
           mode=:zero_order_gradient,
           xmin=-2.0, xmax=2.0)

# example 3
function f(x)
    if x >= 0.0
        return -1.0 + x
    else
        return 1.0 - x
    end
end
function ∇f(x)
    if x >= 0.0
        return 1.0
    else
        return -1.0
    end
end

x0 = 2.0
x_sol = gd(x0, iters=1000,
           σ=[1.0, 0.5, 0.1],#, 0.25, 0.1, 0.01, 0.001],#, 0.075, 0.05, 0.01],
           N = 100,
           mode=:zero_order_gradient,
           xmin=-2.0, xmax=2.0)
