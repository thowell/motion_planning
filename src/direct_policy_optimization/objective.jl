"""
    sample objective
"""
struct SampleObjective <: Objective
    Q
    R
end

function objective(τ_nom, τ, obj::SampleObjective, model_nom, model_sample,
        idx_nom, idx_sample, T)

    J = 0.0

    for t = 1:T
        ȳ = state_output(model_nom, view(τ_nom, idx_nom.x[t]))
        y = state_output(model_sample, view(τ, idx_sample.x[t]))

        J += (y - ȳ)' * obj.Q[t] * (y - ȳ)

        t == T && continue

        v̄ = control_output(model_nom, view(τ_nom, idx_nom.u[t]))
        v = control_output(model_sample, view(τ, idx_sample.u[t]))
        J += (v - v̄)' * obj.R[t] * (v - v̄)
    end

    return J
end

function objective_gradient!(∇J, τ_nom, τ, obj::SampleObjective,
        model_nom, model_sample, idx_nom, idx_sample, T; shift_sample = 0)

    for t = 1:T
        ȳ = state_output(model_nom, view(τ_nom, idx_nom.x[t]))
        y = state_output(model_sample, view(τ, idx_sample.x[t]))

        ∇J[state_output_idx(model_nom, idx_nom.x[t])] += 2.0 * obj.Q[t] * (y - ȳ)
        ∇J[state_output_idx(model_sample, shift_sample .+ idx_sample.x[t])] += 2.0 * obj.Q[t] * (y - ȳ)

        t == T && continue

        v̄ = control_output(model_nom, view(τ_nom, idx_nom.u[t]))
        v = control_output(model_sample, view(τ, shift_sample .+ idx_sample.u[t]))

        ∇J[control_output(model_nom, idx_nom.u[t])] += 2.0 * obj.R[t] * (v - v̄)
        ∇J[control_output(model_sample, shift_sample .+ idx_sample.u[t])] += 2.0 * obj.R[t] * (v - v̄)
    end

    return nothing
end
