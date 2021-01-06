# Model
include_model("cybertruck")

# ramp
jump_slope = 0.3
jump_length = 1.0

function ϕ_func(::CYBERTRUCK, q)
    if q[1] < jump_length && q[1] > 0.0
        return @SVector [q[3] - jump_slope * q[1]]
    else
        return @SVector [q[3]]
    end
end

# Horizon
T = 21

# Time step
tf = 2.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] = [Inf; 1.0]
_ul = zeros(model.m)
_ul[model.idx_u] .= [-1.0; -1.0]

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [-1.0, 0.0, 0.0, 0.0]
qT = [3.0, 0.0, 0.0, 0.0]

x1 = [q1; q1]
xT = [qT; qT]

xl, xu = state_bounds(model, T,
    x1 = x1, xT = xT)

# Objective
Qq = Diagonal(ones(model.nq))
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(10.0 * Qq, 1000.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0e-1 * ones(model.nu)..., zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_penalty = PenaltyObjective(1.0e5, model.m)
obj = MultiObjective([obj_tracking, obj_penalty])

# Constraints
include_constraints(["contact", "control_complementarity"])

n_cc_stage = 2
n_cc_con = n_cc_stage * (T - 1)
con_ctrl_comp = ControlComplementarity(n_cc_con, (1:n_cc_con), n_cc_stage)

con_contact = contact_constraints(model, T)

con = multiple_constraints([con_contact, con_ctrl_comp])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
q_ref = linear_interpolation(q1, qT, T)
x_ref = configuration_to_state(q_ref)
x0 = deepcopy(x_ref) # linear interpolation on state
u0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include_snopt()
@time z̄ , info = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
open(vis)
default_background!(vis)

obj_path = joinpath(pwd(), "models/cybertruck/cybertruck.obj")
mtl_path = joinpath(pwd(), "models/cybertruck/cybertruck.mtl")

ctm = ModifiedMeshFileObject(obj_path, mtl_path, scale=0.1)
setobject!(vis["cybertruck"], ctm)
settransform!(vis["cybertruck"], LinearMap(RotZ(pi) * RotX(pi / 2.0)))

Ns = 100
height = range(0, stop = jump_slope * jump_length, length = Ns)
width = range(0, stop = jump_length, length = Ns)
wid = jump_length / Ns
for i = 1:Ns
    setobject!(vis["stair$i"],
        Rect(Vec(0., 0.0, 0.0), Vec(0.01, 1.0, height[i])),
            MeshPhongMaterial(color = RGBA(0.5, 0.5, 0.5, 1.0)))
    settransform!(vis["stair$i"], Translation(width[i], -0.5, 0))
end

q = state_to_configuration([[x̄[1] for i = 1:10]...,x̄..., [x̄[end] for i = 1:10]...])
anim = MeshCat.Animation(convert(Int,floor(1.0 / h)))
for t = 1:length(q)
    MeshCat.atframe(anim, t) do
        settransform!(vis["cybertruck"],
            compose(Translation((q[t][1:3] + ((q[t][1] < jump_length + 1.0 && q[t][1] > -0.1) ? [0.0; 0.0; 0.01] : zeros(3))) ...),
            LinearMap(RotZ(q[t][4] + pi) * RotY((q[t][1] < jump_length + 0.0 && q[t][1] > -0.25) ? tan(jump_slope) : 0.0) * RotX(pi / 2.0))))
    end
end
MeshCat.setanimation!(vis, anim)
settransform!(vis["/Cameras/default"],
    compose(LinearMap(RotZ(-pi / 2.0)), Translation(0.0, 3.0, 1.5)))
