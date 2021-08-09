n = 2

mass = 1.0
gravity = 9.81
mass_matrix = Diagonal(mass * ones(n))
bias = [0.0; gravity * mass]
contact_jacobian = Diagonal(ones(n))
ϕ(q) = q[2]

h = 0.01

a = 500.0
softplus(x) = 1 / a * log(1.0 + exp(a * x))
softminus(x) = x - softplus(x)

plot(softplus)
plot(softminus)

function dynamics(x, u, t)
	q = x[1:n]
	v = x[n .+ (1:n)]

	k_impact = 500.0
	b_impact = 10.0
	# λ = [0.0; -k_impact * min(0.0, ϕ(q))]
	λ = [0.0; -k_impact * softminus(ϕ(q)) + max(0.0, - b_impact * v[2])]

	v⁺ = v + h * (mass_matrix \ (-bias + transpose(contact_jacobian) * λ))
	q⁺ = q + h * v⁺

	return [q⁺; v⁺]
end


q1 = [0.0; 1.0]
v1 = [0.0; 0.0]
x1 = [q1; v1]

x_hist = [x1]
T = 1000

for t = 1:T-1
	push!(x_hist, dynamics(x_hist[end], nothing, t))
	println(ϕ(x_hist[end][1:2]))
end

plot(hcat(x_hist...)[1:2, :]', )
