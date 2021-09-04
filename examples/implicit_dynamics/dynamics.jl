struct DynamicsData{T}
	m
	ip_dyn::InteriorPoint
	ip_jac::InteriorPoint
    z_subset_init::Vector{T}
	θ_params
	h::T
	diff_idx::Int
end

typeof(view(data.ip_dyn.z, collect(1:data.m.dim.q)))
typeof(view(data.ip_dyn.δz, collect(1:data.m.dim.q), collect(1:data.m.dim.q)))

function dynamics_data(m, h,
        r!, rz!, rθ!, rz, rθ;
        idx_ineq = collect(1:0),
        idx_soc = Vector{Int}[],
        z_subset_init = ones(size(rz)[1] - m.dim.q),
		θ_params = [],
		diff_idx = -1,
		dyn_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = false),
		jac_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-2,
						κ_init = 0.1,
						diff_sol = true))
    nz = size(rz)[1]
    nθ = size(rθ)[2]
	z = zeros(nz)
	θ = zeros(nθ)

	ip_dyn = interior_point(z, θ,
		idx_ineq = idx_ineq,
        idx_soc = idx_soc,
		r! = r!,
		rz! = rz!,
		rθ! = rθ!,
		rz = rz,
		rθ = rθ,
		opts = dyn_opts)

	ip_dyn.opts.diff_sol = false

	ip_jac = interior_point(z, θ,
		idx_ineq = idx_ineq,
        idx_soc = idx_soc,
		r! = r!,
		rz! = rz!,
		rθ! = rθ!,
		rz = rz,
		rθ = rθ,
		opts = jac_opts)

	ip_jac.opts.diff_sol = true

	DynamicsData(m, ip_dyn, ip_jac, z_subset_init, θ_params, h, diff_idx)
end

function f!(d::DynamicsData, q0, q1, u1, mode = :dynamics)
	ip = (mode == :dynamics ? d.ip_dyn : d.ip_jac)
	h = d.h

	ip.z .= copy([q1; d.z_subset_init])
	ip.θ .= copy([q0; q1; u1; h; d.θ_params])

	status = interior_point_solve!(ip)

	!status && (@warn "dynamics failure (res norm: $(norm(ip.r, Inf))) \n
		               z = $(ip.z), \n
					   θ = $(ip.θ)")
end

function f(d::DynamicsData, q0, q1, u1)
	f!(d, q0, q1, u1, :dynamics)
	return copy(d.ip_dyn.z[1:d.m.dim.q])
end

# function fq0(d::DynamicsData, q0, q1, u1)
# 	f!(d, q0, q1, u1, :jacobian)
# 	return copy(d.ip_jac.δz[1:d.m.dim.q, 1:d.m.dim.q])
# end
#
# function fq1(d::DynamicsData, q0, q1, u1)
# 	f!(d, q0, q1, u1, :jacobian)
# 	return copy(d.ip_jac.δz[1:d.m.dim.q, d.m.dim.q .+ (1:d.m.dim.q)])
# end

function fx1(d::DynamicsData, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.m.dim.q, 1:(2 * d.m.dim.q)])
end

function fu1(d::DynamicsData, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.m.dim.q, 2 * d.m.dim.q .+ (1:d.m.dim.u)])
end

struct ImplicitDynamics{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
	dynamics::DynamicsData
end

function fd(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.m.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]

	q2 = f(model.dynamics, q0, q1, u)

	return [q1; q2]
end

function fdx(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.m.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2dx1 = fx1(model.dynamics, q0, q1, u)

	return [zeros(nq, nq) I; dq2dx1]
end

function fdu(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.m.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2du1 = fu1(model.dynamics, q0, q1, u)
	return [zeros(nq, model.m); dq2du1]
end

# test
# data = dynamics_data(model, h,
# 	dyn_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-4, κ_init = 0.1),
# 	jac_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-2, κ_init = 0.1))

# f(data, q0, q1, zeros(m))
# fq0(data, q0, q1, zeros(m))
# fq1(data, q0, q1, zeros(m))
# fx1(data, q0, q1, zeros(m))
# fu1(data, q0, q1, zeros(m))

# model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)
# fd(model_implicit, [q0; q1], zeros(model_implicit.m), zeros(model_implicit.d), h, 1)
# fdx(model_implicit, [q0; q1], zeros(model_implicit.m), zeros(model_implicit.d), h, 1)
# fdu(model_implicit, [q0; q1], zeros(model_implicit.m), zeros(model_implicit.d), h, 1)
