struct Sample
	β
	#TODO: adaptive sampling
end

function sample_params(b, T)
	β = [b for t = 1:T]
	return Sample(β)
end

struct SampleDynamics <: Constraints
	n
	ineq
	w0
end

function sample_dynamics_constraints(prob, N, M)
	T = prob.nom.T

	# sample mean
	n = prob.mean.model.n * T

	# resample
	for t = 1:T-1
		n += prob.mean.model.n * N
	end

	ineq = (1:0)
	w0 = [zeros(p.model.n) for p in prob.sample]

	return SampleDynamics(n, ineq, w0)
end

function constraints!(c, Z, con::SampleDynamics,
	prob::DPOProblems, idx::DPOIndices,
	N, D,
	dist,
	sample)

	T = prob.nom.T
	h = prob.nom.h
	M = N + D

	con_shift = 0

	# sample mean
	for t = 1:T

		# mean
		μ = view(Z, idx.mean[prob.mean.idx.x[t]])

		# samples
		x = [view(Z, idx.sample[i][prob.sample[i].idx.x[t]]) for i = 1:N]

		# sample mean
		c[con_shift .+ (1:prob.mean.model.n)] = μ - sample_mean(x)
		con_shift += prob.mean.model.n
	end

	# resample
	n_resample = N * prob.mean.model.n

	for t = 1:T-1
		xt = view(Z, idx.xt[t])
		ut = view(Z, idx.ut[t])
		μ = view(Z, idx.mean[prob.mean.idx.x[t]])
		ν = view(Z, idx.mean[prob.mean.idx.u[t]])

		# samples
		xt⁺ = view(Z, idx.xt[t + 1])

		c[con_shift .+ (1:n_resample)] = sample_dynamics(prob.mean.model,
			xt, ut, μ, ν, dist.w, h, t, sample.β)[1] - xt⁺
		con_shift += n_resample
	end

    return nothing
end

function constraints_jacobian!(∇c, Z, con::SampleDynamics,
	prob::DPOProblems, idx::DPOIndices,
	N, D,
	dist,
	sample)

	T = prob.nom.T
	h = prob.nom.h
	M = N + D

	con_shift = 0
	jac_shift = 0

	# sample mean
	for t = 1:T
		# samples
		x_vec = vcat([view(Z, idx.sample[i][prob.sample[i].idx.x[t]]) for i = 1:N]...)
		sample_mean_vec(y) = sample_mean([view(y, (i - 1) * prob.mean.model.n
			.+ (1:prob.mean.model.n)) for i = 1:N])

		r_idx = con_shift .+ (1:prob.mean.model.n)

		c_idx = idx.mean[prob.mean.idx.x[t]]
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(Diagonal(ones(prob.mean.model.n)))
		jac_shift += len

		c_idx = vcat([idx.sample[i][prob.sample[i].idx.x[t]] for i = 1:N]...)
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(-1.0 * ForwardDiff.jacobian(sample_mean_vec, x_vec))
		jac_shift += len

		con_shift += prob.mean.model.n
	end

	# resample
	n_resample = N * prob.mean.model.n
	for t = 1:T-1
		xt = view(Z, idx.xt[t])
		ut = view(Z, idx.ut[t])
		μ = view(Z, idx.mean[prob.mean.idx.x[t]])
		ν = view(Z, idx.mean[prob.mean.idx.u[t]])

		# samples
		# xt⁺ = view(Z, idx.xt[t + 1])

		a1, a2, a3, a4 = sample_dynamics_jacobian(model, xt, ut, μ, ν,
			dist.w, h, t, sample.β)

		r_idx = con_shift .+ (1:n_resample)

		c_idx = idx.xt[t]
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(a1)
		jac_shift += len

		c_idx = idx.ut[t]
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(a2)
		jac_shift += len

		c_idx = idx.mean[prob.mean.idx.x[t]]
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(a3)
		jac_shift += len

		c_idx = idx.mean[prob.mean.idx.u[t]]
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(a4)
		jac_shift += len

		c_idx = idx.xt[t + 1]
		len = length(r_idx) * length(c_idx)
		∇c[jac_shift .+ (1:len)] = vec(-1.0 * Diagonal(ones(n_resample)))
		jac_shift += len
		con_shift += n_resample
	end

	return nothing
end

function constraints_sparsity(con::SampleDynamics,
	prob::DPOProblems, idx::DPOIndices,
	N, D;
	shift_row = 0, shift_col = 0)

	row = []
    col = []

	T = prob.nom.T
	h = prob.nom.h
	M = N + D

	con_shift = 0

	# sample mean
	for t = 1:T
		# samples
		r_idx = shift_row + con_shift .+ (1:prob.mean.model.n)

		c_idx = shift_col .+ idx.mean[prob.mean.idx.x[t]]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ vcat([idx.sample[i][prob.sample[i].idx.x[t]] for i = 1:N]...)
		row_col!(row, col, r_idx, c_idx)

		con_shift += prob.mean.model.n
	end

	# resample
	n_resample = N * prob.mean.model.n
	for t = 1:T-1
		r_idx = shift_row + con_shift .+ (1:n_resample)

		c_idx = shift_col .+ idx.xt[t]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.ut[t]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.mean[prob.mean.idx.x[t]]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.mean[prob.mean.idx.u[t]]
		row_col!(row, col, r_idx, c_idx)

		c_idx = shift_col .+ idx.xt[t + 1]
		row_col!(row, col, r_idx, c_idx)

		con_shift += n_resample
	end

    return collect(zip(row, col))
end

function sample_dynamics(model, xt, ut, μ, ν, w, h, t, β)
	N = 2 * model.n
	D = 2 * model.d
	M = N + D

	w0 = zeros(model.d)

	# propagate samples
	s = zeros(model.n * M)

	for j = 1:M
		if j <= N
			xi = view(xt, (j - 1) * model.n .+ (1:model.n))
			ui = view(ut, (j - 1) * model.m .+ (1:model.m))
			s[(j - 1) * model.n .+ (1:model.n)] = propagate_dynamics(model,
				xi, ui, w0, h, t)
		else
			k = j - N
			s[(j - 1) * model.n .+ (1:model.n)] = propagate_dynamics(model,
				μ, ν,
				β[t] * w[t][k], h, t)
		end
	end

	# resample
	xt⁺ = resample_vec(s, model.n, M, β[t + 1])

	return xt⁺, s
end

function sample_dynamics_jacobian(model, xt, ut, μ, ν, w, h, t, β)
	N = 2 * model.n
	D = 2 * model.d
	M = N + D

	w0 = zeros(model.d)

	dx⁺dxt = zeros(model.n * M, model.n * N)
	dx⁺dut = zeros(model.n * M, model.m * N)
	dx⁺dμ = zeros(model.n * M, model.n)
	dx⁺dν = zeros(model.n * M, model.m)

	dsdxt = zeros(model.n * M, model.n * N)
	dsdut = zeros(model.n * M, model.m * N)
	dsdμ = zeros(model.n * M, model.n)
	dsdν = zeros(model.n * M, model.m)

	xt⁺, s = sample_dynamics(model, xt, ut, μ, ν, w, h, t, β)
	r(y) = resample_vec(y, model.n, M, β[t + 1])
	dx⁺ds = real.(FiniteDiff.finite_difference_jacobian(r, s))

	# s = []
	# A = []
	# B = []

	for j = 1:M
		if j <= N
			xi = view(xt, (j - 1) * model.n .+ (1:model.n))
			ui = view(ut, (j - 1) * model.m .+ (1:model.m))
			_, _A, _B = propagate_dynamics_jacobian(model, xi, ui,
				w0, h, t)
			# push!(s, _s)
			# push!(A, _A)
			# push!(B, _B)

			dsdxt[(j - 1) * model.n .+ (1:model.n),
				(j - 1) * model.n .+ (1:model.n)] = _A
			dsdut[(j - 1) * model.n .+ (1:model.n),
				(j - 1) * model.m .+ (1:model.m)] = _B
		else
			k = j - N
			_, _A, _B = propagate_dynamics_jacobian(model, μ, ν,
				β[t] * w[t][k], h, t)

			# push!(s, _s)
			# push!(A, _A)
			# push!(B, _B)

			dsdμ[(j - 1) * model.n .+ (1:model.n), :] = _A
			dsdν[(j - 1) * model.n .+ (1:model.n), :] = _B
		end
	end

	dx⁺dxt = dx⁺ds * dsdxt
	dx⁺dut = dx⁺ds * dsdut
	dx⁺dμ = dx⁺ds * dsdμ
	dx⁺dν = dx⁺ds * dsdν

	return dx⁺dxt, dx⁺dut, dx⁺dμ, dx⁺dν
end
