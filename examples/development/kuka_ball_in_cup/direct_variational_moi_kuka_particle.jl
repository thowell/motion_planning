using LinearAlgebra, Plots, ForwardDiff, Ipopt, MathOptInterface
using Interpolations
# using SNOPT7

const MOI = MathOptInterface

include("direct_variational_methods_kuka_particle.jl")
include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")
struct Problem <: MOI.AbstractNLPEvaluator
    nq
    nu
    nγ
    ns

    p_dy
    p_sd
    p_cm

    p_dy_jac
    p_sd_jac
    p_cm_jac

    jac_sparsity

    N
    enable_hessian::Bool
end

function Problem(nq,nu,nγ,ns,N)
    shift = nq+nu+nγ+ns

    p_dy = (N-1)*nq
    p_sd = (N+1)*nγ
    p_cm = N-1

    p_dy_jac = (N-1)*nq*(shift + 2nq)
    p_sd_jac = (N+1)*nγ*nq
    p_cm_jac = (N-1)*(nγ + ns + nq)

    sparsity_dy = sparsity_dynamics()
    sparsity_sd = sparsity_signed_distance(p_dy)
    sparsity_cm = sparsity_complementarity(p_dy+p_sd)

    sparsity = []
    append!(sparsity,sparsity_dy)
    append!(sparsity,sparsity_sd)
    append!(sparsity,sparsity_cm)

    Problem(nq,nu,nγ,ns,p_dy,p_sd,p_cm,p_dy_jac,p_sd_jac,p_cm_jac,sparsity,N,false)
end


MOI.features_available(prob::Problem) = [:Grad, :Jac]
MOI.initialize(prob::Problem, features) = nothing
MOI.jacobian_structure(prob::Problem) = prob.jac_sparsity
MOI.hessian_lagrangian_structure(prob::Problem) = []

function MOI.eval_objective(prob::Problem, Z)
    return cost(Z)
end

function MOI.eval_objective_gradient(prob::Problem, grad_f, Z)
    ∇cost!(grad_f, Z)
end

function MOI.eval_constraint(prob::Problem, g, Z)
    g_dy = view(g,1:prob.p_dy)
    g_sd = view(g,prob.p_dy .+ (1:prob.p_sd))
    g_cm = view(g,(prob.p_dy + prob.p_sd) .+ (1:prob.p_cm))
    dynamics_constraints!(g_dy,Z)
    signed_distance_constraints!(g_sd,Z)
    complementarity_constraints!(g_cm,Z)
end

function MOI.eval_constraint_jacobian(prob::Problem, jac, Z)
    jac_dy = view(jac,1:prob.p_dy_jac)
    jac_sd = view(jac,prob.p_dy_jac .+ (1:prob.p_sd_jac))
    jac_cm = view(jac,(prob.p_dy_jac + prob.p_sd_jac) .+ (1:prob.p_cm_jac))

    ∇dynamics_constraints!(jac_dy,Z)
    ∇signed_distance_constraints!(jac_sd,Z)
    ∇complementarity_constraints!(jac_cm,Z)
end

MOI.eval_hessian_lagrangian(prob::Problem, H, x, σ, μ) = nothing
NN
prob = Problem(nq,nu,nγ,ns,N)

Q0 = [q0,line_trajectory(q1,qN,N)...]
# Q0 = copy(Q)
# Q0 = [q0 for k = 1:N+1]
# Q0 = [q0,[[Q_kuka[k];q0[8:10]] for k = 2:N+1]...]

# U0 = copy(U_hold)
U0 = copy(U_hold_rand)
# U0 = copy(U)
# U0 = copy(Q_kuka)
Γ0 = [0.01*rand(nγ) for k = 1:N-1]
S0 = [ones(ns) for k = 1:N-1]


# Q0 = copy(Q)
# U0 = copy(U)
# Γ0 = copy(Γ)
# S0 = copy(S)

ZZ = packZ(Q0,U0,Γ0,S0)

con_bnds = [(prob.p_dy,:equality),(prob.p_sd,:inequality),(prob.p_cm,:inequality)]

Z_low, Z_up = primal_bounds(q0,q1,qN)
c_low, c_up = constraint_bounds(con_bnds)

nlp_bounds = MOI.NLPBoundsPair.(c_low,c_up)
block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

# solver = Ipopt.Optimizer()
solver = SNOPT7.Optimizer()

Z = MOI.add_variables(solver,NN)

for i = 1:NN
    zi = MOI.SingleVariable(Z[i])
    MOI.add_constraint(solver, zi, MOI.LessThan(Z_up[i]))
    MOI.add_constraint(solver, zi, MOI.GreaterThan(Z_low[i]))
    MOI.set(solver, MOI.VariablePrimalStart(), Z[i], ZZ[i])
end

# Solve the problem
MOI.set(solver, MOI.NLPBlock(), block_data)
MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
@time MOI.optimize!(solver)

# Get the solution
res = MOI.get(solver, MOI.VariablePrimal(), Z)

Q,U,Γ,S = unpackZ(res)

Q_kuka = [Q[k][1:7] for k = 1:N+1]
Q_ee = [Array(end_effector_function(Q[k][1:7])) for k = 1:N+1]
Q_p = [Q[k][8:10] for k = 1:N+1]
#
# Xblk = zeros(3,N+1)
# for k = 1:N+1
#     Xblk[:,k] = Q_p[k]
# end
#
# Eblk = zeros(3,N+1)
# for k = 1:N+1
#     Eblk[:,k] = Q_ee[k]
# end
#
# Ublk = zeros(nu,N-1)
# for k = 1:N-1
#     Ublk[:,k] = U[k]
# end
#
# Γblk = zeros(N-1)
# for k = 1:N-1
#     Γblk[k] = Γ[k][1]
# end
# Sblk = zeros(N-1)
# for k = 1:N-1
#     Sblk[k] = S[k][1]
# end
#
# p = plot(tspan,Xblk[1,2:end],width=2,label="x",legend=:topleft,title="Ball trajectory",xlabel="time (s)",ylabel="position (m)")
# p = plot!(tspan,Xblk[2,2:end],width=2,label="y")
# p = plot!(tspan,Xblk[3,2:end],width=2,label="z")
# savefig(p,joinpath(pwd(),"ball_in_cup_kuka/","ball_trajectory.png"))
#
# p = plot(tspan,Eblk[1,2:end],width=2,label="x",legend=:topleft,title="End effector trajectory",xlabel="time (s)",ylabel="position (m)")
# p = plot!(tspan,Eblk[2,2:end],width=2,label="y")
# p = plot!(tspan,Eblk[3,2:end],width=2,label="z")
# savefig(p,joinpath(pwd(),"ball_in_cup_kuka/","ee_trajectory.png"))
#
# plot(Ublk')
# tspan = range(0,stop=(h*(N-1)),length=N)
#
# p = plot(tspan,[Γblk;Γblk[end]],width=2,linetype=:steppost,title="Contact Force",label="",xlabel="time (s)",ylabel="force (N)")
# savefig(p,joinpath(pwd(),"ball_in_cup_kuka/","force.png"))
#
# p = plot(tspan,[Sblk;Sblk[end]],width=2,linetype=:steppost,title="max slack = $(maximum(S))",label="",xlabel="time (s)",ylabel="slack",ylim=(-1,1))
# savefig(p,joinpath(pwd(),"ball_in_cup_kuka/","slack.png"))
# maximum(S)
#
# maximum(S)


using MeshCatMechanisms
using MeshCat
using GeometryTypes
using CoordinateTransformations

vis = MeshCatMechanisms.Visualizer()
open(vis)
mvis = MechanismVisualizer(mechanismkuka, visualskuka, vis[:base])

function MeshCat.setanimation!(mvis::MechanismVisualizer,
                      times::AbstractVector{<:Real},
                      configurations::AbstractVector{<:AbstractVector{<:Real}},
                      config_p::AbstractVector{<:AbstractVector{<:Real}};
                      fps::Integer=30,
                      play::Bool=true,
                      repetitions::Integer=1)

    q0 = copy(configuration(MeshCatMechanisms.state(mvis)))
    @assert axes(times) == axes(configurations)
    @assert axes(times) == axes(config_p)

    interpolated_configurations = Interpolations.interpolate((times,), configurations, Gridded(Linear()))
    interpolated_configurations2 = Interpolations.interpolate((times,), config_p, Gridded(Linear()))

    animation = MeshCatMechanisms.Animation()
    num_frames = floor(Int, (times[end] - first(times)) * fps)

    setobject!(mvis["load"],HyperSphere(Point3f0(0), convert(Float32,0.05)) ,MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))
    settransform!(mvis["load"], Translation(interpolated_configurations2(0.0)...))

    for frame in 0:num_frames
        time = first(times) + frame / fps
        set_configuration!(MeshCatMechanisms.state(mvis), interpolated_configurations(time))

        atframe(animation, MeshCatMechanisms.visualizer(mvis), frame) do frame_visualizer
            # println(interpolated_configurations2(time))
            settransform!(frame_visualizer["load"], Translation(interpolated_configurations2(time)...))
            MeshCatMechanisms._render_state!(MechanismVisualizer(MeshCatMechanisms.state(mvis), frame_visualizer))
        end
    end
    setanimation!(MeshCatMechanisms.visualizer(mvis), animation, play=play, repetitions=repetitions)
    set_configuration!(MeshCatMechanisms.state(mvis), q0)
end

setanimation!(mvis,t,Q_kuka[2:end],Q_p[2:end])

function output_traj(Q,idx=2:N+1,filename=joinpath(pwd(),"kuka.txt"))
    f = open(filename,"w")
    for k = idx
        q1, q2, q3, q4, q5, q6, q7 = Q[k][1:7]
        str = "$q1 $q2 $q3 $q4 $q5 $q6 $q7"
        if k != N
            str *= " "
        end
        write(f,str)
    end

    close(f)
end

output_traj(Q)
output_traj(U,collect(1:N-1),joinpath(pwd(),"kuka_control.txt"))
