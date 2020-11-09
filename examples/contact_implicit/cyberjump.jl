include(joinpath(pwd(), "src/models/cybertruck.jl"))
include(joinpath(pwd(), "src/constraints/contact.jl"))
include(joinpath(pwd(), "src/constraints/control_complementarity.jl"))
include(joinpath(pwd(), "src/constraints/obstacles.jl"))

# ramp
jump_slope = 0.3
jump_length = 1.0

function ϕ_func(::CYBERTRUCK, q)
    if q[1] < jump_length
        return @SVector [q[3] - jump_slope * q[1]]
    else
        return @SVector [q[3]]
    end
end

# Horizon
T = 21

# Time step
tf = 2.0
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] = [Inf; 1.0]
_ul = zeros(model.m)
_ul[model.idx_u] .= [0.0; -1.0]

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [-2.0, 2.0, 0.0, -1.0 * pi/ 2.0]
qT = [3.0, 0.0, 0.0, pi / 2.0]

x1 = [q1; q1]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
Qq = Diagonal(ones(model.nq))
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, 100.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0e-1 * ones(model.nu)..., zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [[zeros(model.nq); qT] for t = 1:T],
    [zeros(model.m) for t = 1:T]
    )

obj_penalty = PenaltyObjective(100.0, model.m)
obj = MultiObjective([obj_tracking, obj_penalty])

# Constraints
p_car1 = [3.0, 0.65]
p_car2 = [3.0, -0.65]

function obstacles!(c, x)
    c[1] = circle_obs(x[1], x[2], p_car1[1], p_car1[2], 0.6)
    c[2] = circle_obs(x[1], x[2], p_car2[1], p_car2[2], 0.6)
    nothing
end

n_obs_stage = 2
n_obs_con = n_obs_stage * T
con_obstacles = ObstacleConstraints(n_obs_con, (1:n_obs_con), n_obs_stage)

n_cc_stage = 2
n_cc_con = n_cc_stage * (T - 1)
con_ctrl_comp = ControlComplementarity(n_cc_con, (1:n_cc_con), n_cc_stage)

con_contact = contact_constraints(model, T)

con = multiple_constraints([con_contact, con_ctrl_comp, con_obstacles])

# Problem
prob = problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con
               )

# Trajectory initialization
q_ref = linear_interp(q1, qT, T)
x_ref = configuration_to_state(q_ref)
X0 = deepcopy(x_ref) # linear interpolation on state
U0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

@time Z̄ = solve(prob, copy(Z0), tol = 1.0e-3, c_tol = 1.0e-3)

check_slack(Z̄, prob)
X̄, Ū = unpack(Z̄, prob)

include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)
# visualize!(vis, model, state_to_configuration(X̄), Δt = h)

q = state_to_configuration(X̄)
obj_path = joinpath(pwd(), "src/models/cybertruck/cybertruck.obj")
mtl_path = joinpath(pwd(), "src/models/cybertruck/cybertruck.mtl")
ctm = ModifiedMeshFileObject(obj_path, mtl_path, scale = 0.1)
setobject!(vis["cybertruck"], ctm)
settransform!(vis["cybertruck"], LinearMap(RotZ(pi) * RotX(pi / 2.0)))
Ns = 100
height = range(0, stop = jump_slope * jump_length, length = Ns)
width = range(0, stop = jump_length, length = Ns)
wid = jump_length / Ns
for i = 1:Ns
    setobject!(vis["stair$i"], Rect(Vec(0., 0.0, 0.0), Vec(0.01, 1.0, height[i])))
    settransform!(vis["stair$i"], Translation(width[i], -0.5, 0))
end

setobject!(vis["cybertruck_park1"], ctm)
settransform!(vis["cybertruck_park1"],
    compose(Translation([p_car1[1]; p_car1[2]; 0.0]),
    LinearMap(RotZ(pi + pi / 2) * RotX(pi / 2.0))))

setobject!(vis["cybertruck_park2"], ctm)
settransform!(vis["cybertruck_park2"],
    compose(Translation([p_car2[1]; p_car2[2]; 0.0]),
    LinearMap(RotZ(pi + pi / 2) * RotX(pi / 2.0))))

anim = MeshCat.Animation(convert(Int,floor(1.0 / h)))
for t = 1:length(q)
    MeshCat.atframe(anim, t) do
        settransform!(vis["cybertruck"],
            compose(Translation((q[t][1:3] + ((q[t][1] < jump_length + 1.0 && q[t][1] > -0.1) ? [0.0; 0.0; 0.01] : zeros(3))) ...),
            LinearMap(RotZ(q[t][4] + pi) * RotY((q[t][1] < jump_length + 0.0 && q[t][1] > -0.25) ? tan(jump_slope) : 0.0) * RotX(pi / 2.0))))
    end
end
MeshCat.setanimation!(vis, anim)
