include_model("double_integrator")

using Distributions

T = 10
h = 1.0
u_dist = Distributions.MvNormal(zeros(model.m),
	Diagonal(1.0e-1 * ones(model.m)))
u_hist = rand(u_dist, T-1)

x1 = [-1.0; 0.0]
x_hist = [x1]

for t = 1:T-1
	w = zeros(model.d)
	x⁺ = propagate_dynamics(model, x_hist[end], u_hist[t], w, h, t;
			solver = :levenberg_marquardt,
			tol_r = 1.0e-8, tol_d = 1.0e-6)
	push!(x_hist, x⁺)
end

plot(hcat(x_hist...)')
plot(hcat(u_hist...)')

# LQR
A, B = get_dynamics(model)
Q = @SMatrix [1.0 0.0; 0.0 0.0]
R = @SMatrix [1.0]

function objective(x, u)
	T = length(x)
	J = 0.0
	for t = 1:T
		J += x[t]' * Q * x[t]
		t < T & continue
		J += u[t]' * R * u[t]
	end
	return J
end

K, P = tvlqr([A for t = 1:T-1], [B for t = 1:T-1],
	[Q for t = 1:T], [R for t = 1:T-1])

K_vec = [vec(K[t]) for t = 1:T-1]

plot(hcat(K_vec...)')
K_inf = copy(K[1])

x1 = [-1.0; 0.0]
x_hist = [x1]
u_hist = []

for t = 1:T-1
	w = zeros(model.d)
	push!(u_hist, -K[t] * x_hist[end])
	# push!(u_hist, -K_inf * x_hist[end])

	x⁺ = propagate_dynamics(model, x_hist[end], u_hist[end], w, h, t;
			solver = :levenberg_marquardt,
			tol_r = 1.0e-8, tol_d = 1.0e-6)

	push!(x_hist, x⁺)
end

plot(hcat(x_hist...)')
plot(hcat(u_hist...)')

J = objective(x_hist, u_hist) # 2.297419

# policy gradient w/ Adam
α = 0.001
β1 = 0.9
β2 = 0.999
ϵ = 10^(-8)

θ = randn(model.n)
