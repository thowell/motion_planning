"""
    sample objective
"""
struct SampleObjective <: Objective
    Q
    R
end

sample_objective(Q, R) = SampleObjective(Q, R)

function objective(τ_nom, τ_sample, obj::SampleObjective,
        model_nom, model_sample,
        idx_τ_nom, idx_τ_sample, T)

    J = 0.0

    for t = 1:T
        ȳ = state_output(model_nom, view(τ_nom, idx_τ_nom.x[t]))
        y = state_output(model_sample, view(τ_sample, idx_τ_sample.x[t]))

        J += (y - ȳ)' * obj.Q[t] * (y - ȳ)

        t == T && continue

        v̄ = control_output(model_nom, view(τ_nom, idx_τ_nom.u[t]))
        v = control_output(model_sample, view(τ_sample, idx_τ_sample.u[t]))
        J += (v - v̄)' * obj.R[t] * (v - v̄)
    end

    return J
end

function objective_gradient!(∇J, τ_nom, τ_sample, obj::SampleObjective,
        model_nom, model_sample,
        idx_τ_nom, idx_τ_sample,
        idx_z_nom, idx_z_sample, T)

    for t = 1:T
        ȳ = state_output(model_nom, view(τ_nom, idx_τ_nom.x[t]))
        y = state_output(model_sample, view(τ_sample, idx_τ_sample.x[t]))

        ∇J[state_output_idx(model_nom, idx_z_nom[idx_τ_nom.x[t]])] -= 2.0 * obj.Q[t] * (y - ȳ)
        ∇J[state_output_idx(model_sample, idx_z_sample[idx_τ_sample.x[t]])] += 2.0 * obj.Q[t] * (y - ȳ)

        t == T && continue

        v̄ = control_output(model_nom, view(τ_nom, idx_τ_nom.u[t]))
        v = control_output(model_sample, view(τ_sample, idx_τ_sample.u[t]))

        ∇J[control_output(model_nom, idx_z_nom[idx_τ_nom.u[t]])] -= 2.0 * obj.R[t] * (v - v̄)
        ∇J[control_output(model_sample, idx_z_sample[idx_τ_sample.u[t]])] += 2.0 * obj.R[t] * (v - v̄)
    end

    return nothing
end

function objective(Z, obj::SampleObjective,
		prob::DPOProblems, idx::DPOIndices, N, D)

	J = 0.0

	τ_nom = view(Z, idx.nom)

	# samples
	for i = 1:N
		τ_sample = view(Z, idx.sample[i])
		J += objective(τ_nom, τ_sample, obj,
			prob.nom.model, prob.sample[i].model,
		    prob.nom.idx, prob.sample[i].idx, prob.nom.T)
	end

	return J
end

function objective_gradient!(∇J, Z, obj::SampleObjective,
		prob::DPOProblems, idx::DPOIndices, N, D)

	τ_nom = view(Z, idx.nom)

	for i = 1:N
		τ_sample = view(Z, idx.sample[i])
		objective_gradient!(∇J, τ_nom, τ_sample, obj,
		 	prob.nom.model, prob.sample[i].model,
		    prob.nom.idx, prob.sample[i].idx,
			idx.nom, idx.sample[i],
			prob.nom.T)
	end

	nothing
end
