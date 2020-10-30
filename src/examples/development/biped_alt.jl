using MeshCatMechanisms
include(joinpath(pwd(), "src/models/biped_alt.jl"))

# Visualize
include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)

urdf = joinpath(pwd(), "src/models/biped/urdf/biped_left_pinned.urdf")
mechanism = parse_urdf(urdf, floating=false)
mvis = MechanismVisualizer(mechanism,
    URDFVisuals(urdf, package_path=[dirname(dirname(urdf))]), vis)

ϵ = 1.0e-8
θ = 10 * pi / 180
h = model.l2 + model.l1 * cos(θ)
ψ = acos(h / (model.l1 + model.l2))
stride = sin(θ) * model.l1 + sin(ψ) * (model.l1 + model.l2)
x1 = [π - θ, π + ψ, θ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
xT = [π + ψ, π - θ - ϵ, 0.0, θ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
kinematics(model, x1)[1]
kinematics(model, xT)[1] * 2.0

kinematics(model, x1)[2]
kinematics(model, xT)[2]

q1 = transformation_to_urdf_left_pinned(model, x1[1:5])

set_configuration!(mvis, q1)

qT = transformation_to_urdf_left_pinned(model, xT[1:5])
set_configuration!(mvis, qT)

ζ = 11
xM = [π, π - ζ * pi / 180.0, 0.0, 2.0 * ζ * pi / 180.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
qM = transformation_to_urdf_left_pinned(model, xM[1:5])
set_configuration!(mvis, qM)
kinematics(model, xM)[1]
kinematics(model, xM)[2]

x1_foot_des = kinematics(model, x1)[1]
xT_foot_des = kinematics(model, xT)[1]
xc = 0.5 * (x1_foot_des + xT_foot_des)

x1_foot_des * -1.0 + xT_foot_des
# r1 = x1_foot_des - xc
r1 = xT_foot_des - xc
r2 = 0.1

zM_foot_des = r2

function z_foot_traj(x)
    sqrt((1.0 - ((x - xc)^2.0) / (r1^2.0)) * (r2^2.0))
end

foot_x_ref = range(x1_foot_des, stop = xT_foot_des, length = T)
foot_z_ref = z_foot_traj.(foot_x_ref)

# plot(foot_x_ref,foot_z_ref)

@assert norm(Δ(xT)[1:5] - x1[1:5]) < 1.0e-5

# Horizon
T = 21

tf0 = 0.5
h0 = tf0 / (T-1)

# Bounds
ul, uu = control_bounds(model, T, [-20.0; 0.0], [20.0; 5.0 * h0])

xl, xu = state_bounds(model, T,
    x1 = [Inf * ones(5); x1[1:5]],
    xT = [Inf * ones(5); xT[1:5]])


function c_stage!(c,x,u,t,model)
    c[1] = kinematics(model,x[1:5])[2]
    nothing
end
#
# function c_stage!(c,x,t,model)
#     c[1] = kinematics(model,x[1:5])[2]
#     nothing
# end

m_stage = 1
stage_ineq = (1:m_stage)

include("../src/loop_delta.jl")

# Objective
qq = 0.1*[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
Q = [t < T ? Diagonal(qq) : Diagonal(qq) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T-1])
penalty_obj = PenaltyObjective(1000.0,foot_z_ref,[t for t = 1:T])
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    general_constraints=true,
                    m_general=model.nx,
                    general_ineq=(1:0),
                    stage_constraints=true,
                    m_stage=[t==1 ? 0 : m_stage for t = 1:T-1],
                    stage_ineq=[t==1 ? (1:0) : stage_ineq for t = 1:T-1])

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal_step = solve(prob_moi,copy(Z0),nlp=:SNOPT,max_iter=100,time_limit=120)

# Unpack solutions
X_nominal_step, U_nominal_step, H_nominal_step = unpack(Z_nominal_step,prob)

Q_nominal_step = [X_nominal_step[t][1:5] for t = 1:T]
plot(hcat(Q_nominal_step...)')
foot_traj_nom = [kinematics(model,Q_nominal_step[t]) for t = 1:T]

foot_x_nom = [foot_traj_nom[t][1] for t=1:T]
foot_z_nom = [foot_traj_nom[t][2] for t=1:T]

plt_ft_nom = plot(foot_x_nom,foot_z_nom,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:red)

plot(hcat(U_nominal_step...)',linetype=:steppost)

plot(foot_x_nom)
plot(foot_z_nom)

sum(H_nominal_step)
