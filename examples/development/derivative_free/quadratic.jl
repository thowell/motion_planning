using Plots
using ForwardDiff
using LinearAlgebra
using Random
using Statistics

# https://openai.com/blog/evolution-strategies/

sol = [0.5; 0.1]
f(w) = -sum((w - sol).^2.0)
f(a, b) = f([a; b])

n = 1000
x_limits = range(-2.0, stop = 2.0, length = n)
y_limits = range(-2.0, stop = 2.0, length = n)

npop = 50
sigma = 0.1
alpha = 0.001

w = Random.randn(2)

plt = Plots.contour(x_limits, y_limits, f)
plt = plot!((w[1], w[2]), markershape = :circle, color = :cyan, outline = :cyan)

for i = 1:300
    N = Random.randn(npop, 2)
    R = zeros(npop)
    i % 10 == 0 && (plt = Plots.contour(x_limits, y_limits, f))

    for j = 1:npop
        w_try = w + sigma * N[j, :]
        R[j] = f(w_try)

        if i % 10 == 0
            plt = plot!((w_try[1], w_try[2]),
                markershape = :circle, color = :cyan, outline = :cyan,
                label = "")
        end
    end
    i % 10 == 0 && display(plt)

    A = (R .- mean(R)) ./ std(R)
    w = w + alpha / (npop * sigma) * N' * A

    # plt = plot!((w[1], w[2]),
    #     markershape = :circle, color = :cyan, outline = :cyan,
    #     label = "")
end
