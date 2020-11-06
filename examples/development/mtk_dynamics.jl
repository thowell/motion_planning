using BenchmarkTools, ModelingToolkit

include(joinpath(pwd(), "src/models/car.jl"))

x = zeros(model.n)
u = zeros(model.m)
w = zeros(model.d)
h = 0.0
t = 0
q = zeros(model.nq)

@benchmark M_func($model, $q)
@benchmark C_func($model, $q, $q)
@benchmark fd($model, $x, $x, $u, $w, $h, $t)

n = 15
A = Diagonal(@SVector ones(n))

L(q) = transpose(q) * A * q

f(q) = ForwardDiff.gradient(L, q)
g(q) = 2.0 * A * q
x = rand(n)
f(x) == g(x)
@benchmark f($x)
@benchmark g($x)

# generate fast functions
@variables x_sym[1:n]

F = g(x_sym);
fast = eval(ModelingToolkit.build_function(F, x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[1])
fast_sparsity = ModelingToolkit.sparsejacobian(F, x_sym)
∇fast! = eval(ModelingToolkit.build_function(fast_sparsity, x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇fast = similar(fast_sparsity, Float64)

y = ones(n)
@benchmark fast!($y, $x)
@benchmark fast($x)

l = L(x_sym);
fast = eval(ModelingToolkit.build_function([l], x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[1])
fast_sparsity = ModelingToolkit.sparsejacobian([l], x_sym)
dfast = eval(ModelingToolkit.build_function(fast_sparsity, x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[1])

l2 = dfast(x_sym)

fast2 = eval(ModelingToolkit.build_function(simplify.(l2), x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[1])
fast2_sparsity = ModelingToolkit.sparsejacobian(simplify.(l2), x_sym)
∇fast2! = eval(ModelingToolkit.build_function(fast2_sparsity, x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇fast2 = similar(fast2_sparsity, Float64)

# pendulum

L(q, q̇) = 0.5 * 1.0 * q̇^2.0 - 1.0 * 9.81 * 1.0 * cos(q)
