struct DynamicsData{T}
	m
	ip::InteriorPoint
    z_subset_init::Vector{T}
	θ_params
	h::T
	diff_idx::Int
	x_view::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
	δx_view::SubArray{T,2,Matrix{T},Tuple{Vector{Int},Vector{Int}},false}
	δu_view::SubArray{T,2,Matrix{T},Tuple{Vector{Int},Vector{Int}},false}
	δx_cache::Vector{Matrix{T}}
	δu_cache::Vector{Matrix{T}}
	z_cache::Vector{Vector{T}}
end

function dynamics_data(m, h, T,
        r!, rz!, rθ!, rz, rθ;
        idx_ineq = collect(1:0),
        idx_soc = Vector{Int}[],
        z_subset_init = ones(size(rz)[1] - m.dim.q),
		θ_params = [],
		diff_idx = -1,
		opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = false))

    nz = size(rz)[1]
    nθ = size(rθ)[2]
	z = zeros(nz)
	θ = zeros(nθ)

	ip = interior_point(z, θ,
		idx_ineq = idx_ineq,
        idx_soc = idx_soc,
		r! = r!,
		rz! = rz!,
		rθ! = rθ!,
		rz = rz,
		rθ = rθ,
		opts = opts)

	ip.opts.diff_sol = false

	x_view = view(ip.z, collect(1:m.dim.q))
	δx_view = view(ip.δz, collect(1:m.dim.q), collect(1:2 * m.dim.q))
	δu_view = view(ip.δz, collect(1:m.dim.q), collect(2 * m.dim.q .+ (1:m.dim.u)))
	δx_cache = [zeros(m.dim.q, 2 * m.dim.q) for t = 1:T-1]
	δu_cache = [zeros(m.dim.q, m.dim.u) for t = 1:T-1]

	z_cache = [zero(ip.z) for t = 1:T-1]

	DynamicsData(m, ip, z_subset_init, θ_params, h,
		diff_idx, x_view, δx_view, δu_view, δx_cache, δu_cache,
		z_cache)
end

function f!(d::DynamicsData, q0, q1, u1, mode = :dynamics)
	ip = d.ip
	h = d.h

	ip.z .= copy([q1; d.z_subset_init])
	ip.θ .= copy([q0; q1; u1; h; d.θ_params])

	status = interior_point_solve!(ip)

	!status && (@warn "dynamics failure (res norm: $(norm(ip.r, Inf))) \n
		               z = $(ip.z), \n
					   θ = $(ip.θ)")
end

function f(d::DynamicsData, q0, q1, u1, t)
	f!(d, q0, q1, u1, :dynamics)
	differentiate_solution!(d.ip, z = (d.diff_idx == -1 ? d.ip.z : d.ip.z_cache[d.diff_idx]))
	d.δx_cache[t] .= d.δx_view
	d.δu_cache[t] .= d.δu_view
	d.z_cache[t] .= d.ip.z
	return d.x_view
end

function fx1(d::DynamicsData, q0, q1, u1, t)
	return d.δx_cache[t]
end

function fu1(d::DynamicsData, q0, q1, u1, t)
	return d.δu_cache[t]
end

struct ImplicitDynamics{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
	dynamics::DynamicsData
	cache::Dict
end

function ImplicitDynamics{I,T}(n, m, d, dynamics; cache=Dict()) where {I,T}
	ImplicitDynamics{I,T}(n, m, d, dynamics, cache)
end

function fd(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.m.dim.q
	q0 = view(x, 1:nq)
	q1 = view(x, nq .+ (1:nq))

	q2 = f(model.dynamics, q0, q1, u, t)

	return [q1; q2]
end

function fdx(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.m.dim.q
	q0 = view(x, 1:nq)
	q1 = view(x, nq .+ (1:nq))
	dq2dx1 = fx1(model.dynamics, q0, q1, u, t)

	return [zeros(nq, nq) I; dq2dx1]
end

function fdu(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.m.dim.q
	q0 = view(x, 1:nq)
	q1 = view(x, nq .+ (1:nq))
	dq2du1 = fu1(model.dynamics, q0, q1, u, t)
	return [zeros(nq, model.m); dq2du1]
end
