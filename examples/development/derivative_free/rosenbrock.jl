using Plots
using ForwardDiff
using LinearAlgebra

a = 1.0
b = 100.0

f(x) = (a - x[1])^2.0 + b * (x[2] - x[1]^2.0)^2.0
f(x, y) = f([x; y])
∇f(x) = ForwardDiff.gradient(f, x)
∇²f(x) = ForwardDiff.hessian(f, x)

n = 1000
x_limits = range(-1.0, stop = 1.0, length = n)
y_limits = range(-1.0, stop = 1.0, length = n)

x = zeros(2)

function solve(x;
        max_iter = 100,
        visualize = true)
    visualize && (plt = Plots.contour(x_limits, y_limits, f))
    visualize && (plt = plot!((x[1], x[2]), markershape = :circle, color = :cyan, outline = :cyan))

    for i = 1:max_iter
        grad = ∇f(x)
        println("x:  $x")
        println("grad norm: $(norm(grad))")
        norm(grad) < 1.0e-3 && break

        hess = ∇²f(x)
        # Δ = hess \ grad
        Δ = grad

        α = 1.0
        iter = 1
        while norm(∇f(x - α * Δ)) > norm(grad) && iter < 100
            α *= 0.5
            iter += 1
        end

        x .-= α * Δ

        if i % 10 == 0 && visualize
            plt = plot!((x[1], x[2]),
                markershape = :circle, color = :cyan, outline = :cyan,
                legend = false)
            display(plt)
        end
    end
    print("iters: $i")
    return x
end

solve([0.0, -1.0], max_iter = 1000)
