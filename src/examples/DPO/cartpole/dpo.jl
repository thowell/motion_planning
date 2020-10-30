
# TVLQR cost
Q_lqr = [(t < T ? Diagonal([10.0;10.0;10.0;10.0])
    : Diagonal(100.0*ones(model_nominal.nx))) for t = 1:T]
R_lqr = [Diagonal([1.0,0.0,0.0,0.0,0.0,0.0,0.0]) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

# Sample
N = 2*model.nx
models = [model_friction for i = 1:N]

# μ_sample = range(0.0975,stop=0.1025,length=N)
# models_coefficients = [CartpoleFriction(1.0,0.2,0.5,9.81,μ_sample[i],
#     nx_friction,nu_friction,nu_policy_friction) for i = 1:N]

β = 1.0
w = 1.0e-3*ones(model_friction.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=1.0,w=ones(model.nx))

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[i]
    xu_traj_sample[i][1] = x1_sample[i]
end

K = TVLQR_gains(model,X_friction_nominal,U_friction_nominal,H_friction_nominal,
    Q_lqr,R_lqr,u_policy=(1:1))

prob_sample = init_sample_problem(prob_friction,models,Q_lqr,R_lqr,H_lqr,
    β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
    n_policy=1,
    general_objective=true,
    sample_general_constraints=true,
    m_sample_general=prob_friction.m_general*N,
    sample_general_ineq=vcat([((i-1)*prob_friction.m_general
        .+ (1:prob_friction.m_general)) for i = 1:N]...))

prob_sample_moi = init_MOI_Problem(prob_sample)

# prob_sample_coefficients = init_sample_problem(prob_friction,models_coefficients,
#     Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
#     xl=xl_traj_sample,
#     xu=xu_traj_sample,
#     n_policy=1,
#     general_objective=true,
#     sample_general_constraints=true,
#     m_sample_general=prob_friction.m_general*N,
#     sample_general_ineq=vcat([((i-1)*prob_friction.m_general
#         .+ (1:prob_friction.m_general)) for i = 1:N]...))
#
# prob_sample_moi_coefficients = init_MOI_Problem(prob_sample_coefficients)

Ū_friction_nominal = deepcopy(U_friction_nominal)

Z0_sample = pack(X_friction_nominal,Ū_friction_nominal,H_friction_nominal[1],
    K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7,time_limit=60*10)
# Z_sample_sol = solve(prob_sample_moi,copy(Z_sample_sol),nlp=:SNOPT7)

using JLD
@save joinpath(pwd(),"examples/trajectories/","cartpole_friction.jld") Z_sample_sol
# @load joinpath(pwd(),"examples/trajectories/","cartpole_friction.jld") Z_sample_sol

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,
    prob_sample)
K_sample = [reshape(Z_sample_sol[prob_sample.idx_K[t]],model.nu,model.nx) for t = 1:T-1]

# X_nom_sample_coefficients, U_nom_sample_coefficients, H_nom_sample_coefficients, X_sample_coefficients, U_sample_coefficients, H_sample_coefficients = unpack(Z_sample_sol_coefficients,prob_sample_coefficients)
# K_sample_coefficients = [reshape(Z_sample_sol_coefficients[prob_sample_coefficients.idx_K[t]],model_friction.nu_policy,model_friction.nx) for t = 1:T-1]
#
# norm(vec(hcat(K_sample...)) - vec(hcat(K_sample_coefficients...)))

# plt = plot(hcat(X_nom_sample_coefficients...)[1:4,:]',label="")
# for i = 1:N
#     plt = plot!(hcat(X_sample_coefficients[i]...)[1:4,:]',label="")
# end
# display(plt)

plt = plot(hcat(X_nom_sample...)[1:4,:]',label="")
for i = 1:N
    plt = plot!(hcat(X_sample[i]...)[1:4,:]',label="")
end
display(plt)

# Plot
t_sample = zeros(T)
for t = 2:T
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

plt_ctrl = plot(title="Cartpole w/ friction control",xlabel="time (s)",
    color=:red,width=2.0)
for i = 1:N
    plt_ctrl = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1:1,:]',label="")
end
plt_ctrl = plot!(t_nominal[1:end-1],hcat(U_nominal...)[1:1,:]',color=:purple,
    width=2.0,label="nominal")
plt_ctrl = plot!(t_sample[1:end-1],hcat(U_nom_sample...)[1:1,:]',color=:orange,
    width=2.0,label="nominal (friction)")
display(plt_ctrl)
savefig(plt_ctrl,joinpath(@__DIR__,"results/cartpole_friction_control.png"))

plt_state = plot(title="Cartpole w/ friction state",xlabel="time (s)",
    color=:red,width=2.0)
for i = 1:N
    plt_state = plot!(t_sample,hcat(X_sample[i]...)[1:4,:]',label="")
end
plt_state = plot!(t_sample,hcat(X_nominal...)[1:4,:]',color=:purple,
    width=2.0,label=["nominal" "" "" ""])
plt_state = plot!(t_sample,hcat(X_nom_sample...)[1:4,:]',color=:orange,
    width=2.0,label=["nominal (friction)" "" "" ""])
display(plt_state)
savefig(plt_state,joinpath(@__DIR__,"results/cartpole_friction_state.png"))

S_nominal = [U_nom_sample[t][7] for t=1:T-1]
@assert norm(S_nominal,Inf) < 1.0e-4
β = [U_nom_sample[t][2:3] for t = 1:T-1]
b = [U_nom_sample[t][2] - U_nom_sample[t][3] for t = 1:T-1]
ψ = [U_nom_sample[t][4] for t = 1:T-1]

(model_friction.mp + model_friction.mc)*model_friction.g*model_friction.μ
plot(hcat(b...)',linetype=:steppost)
plot(b,linetype=:steppost)
plot(ψ,linetype=:steppost)

# Simulate

function simulate_cartpole_friction(Kc,z_nom,u_nom,model,Q,R,T_sim,Δt,z0,w;
        _norm=2,
        ul=-Inf*ones(length(u_nom[1])),
        uu=Inf*ones(length(u_nom[1])),
        friction=false,
        μ=0.1)

    T = length(Kc)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    t_ctrl = range(0,stop=tf,length=T)
    dt_sim = tf/(T_sim-1)

    u_policy = 1:model.nu_policy

    A_state = hcat(z_nom...)
    A_ctrl = hcat(u_nom...)

    z_rollout = [z0]
    u_rollout = []
    J = 0.0
    Jx = 0.0
    Ju = 0.0
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)

        z_cubic = zeros(model.nx)
        for i = 1:model.nx
            interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
            z_cubic[i] = interp_cubic(t)
        end

        z = z_rollout[end] + dt_sim*w[:,tt]
        u = u_nom[k][1:model.nu_policy] - Kc[k]*(z - z_cubic)

        # clip controls
        u = max.(u,ul[u_policy])
        u = min.(u,uu[u_policy])

        if friction
            _u = [u[1]-μ*sign(z_cubic[3])*model.g*(model.mp+model.mc);0.0;0.0]
        else
            _u = u[1]
        end

        push!(z_rollout,rk3(model,z,_u,dt_sim))
        push!(u_rollout,u)

        if _norm == 2
            J += (z_rollout[end]-z_cubic)'*Q[k+1]*(z_rollout[end]-z_cubic)
            J += (u_rollout[end]-u_nom[k][u_policy])'*R[k][u_policy,
                u_policy]*(u_rollout[end]-u_nom[k][u_policy])
            Jx += (z_rollout[end]-z_cubic)'*Q[k+1]*(z_rollout[end]-z_cubic)
            Ju += (u_rollout[end]-u_nom[k][u_policy])'*R[k][u_policy,
                u_policy]*(u_rollout[end]-u_nom[k][u_policy])
        else
            J += norm(sqrt(Q[k+1])*(z_rollout[end]-z_cubic),_norm)
            J += norm(sqrt(R[k][u_policy,u_policy])*(u-u_nom[k][u_policy]),_norm)
            Jx += norm(sqrt(Q[k+1])*(z_rollout[end]-z_cubic),_norm)
            Ju += norm(sqrt(R[k][u_policy,u_policy])*(u-u_nom[k][u_policy]),_norm)
        end
    end
    return z_rollout, u_rollout, J/(T_sim-1), Jx/(T_sim-1), Ju/(T_sim-1)
end

using Distributions
T_sim = 10T
Δt = h0
dt_sim = sum(H_nom_sample)/(T_sim-1)

W = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-5*ones(nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-5*ones(nx)))
w0 = rand(W0,1)


model_sim = model_friction
μ_sim = 0.1

t_sim_nominal = range(0,stop=H_nominal[1]*(T-1),length=T_sim)
t_sim_sample = range(0,stop=H_nom_sample[1]*(T-1),length=T_sim)

# t_sample_c = range(0,stop=H_nom_sample_coefficients[1]*(T-1),length=T)
# t_sim_sample_c = range(0,stop=H_nom_sample_coefficients[1]*(T-1),length=T_sim)

K_nominal = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr,
    u_policy=(1:1))

K_friction_nominal = TVLQR_gains(model,X_friction_nominal,U_friction_nominal,
    H_friction_nominal,Q_lqr,R_lqr,u_policy=(1:1))

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_cartpole_friction(K_nominal,
    X_nominal,U_nominal,
    model_sim,Q_lqr,R_lqr,T_sim,Δt,X_nominal[1],w,ul=ul_friction,uu=uu_friction,
    friction=true,
    μ=μ_sim)
plt_tvlqr_nom = plot(t_nominal,hcat(X_nominal...)[1:2,:]',legend=:topleft,color=:red,
    label=["nominal (no friction)" ""],
    width=2.0,xlabel="time",title="Cartpole",ylabel="state",ylims=(-1,5))
plt_tvlqr_nom = plot!(t_sim_nominal,hcat(z_tvlqr...)[1:2,:]',color=:purple,
    label=["tvlqr" ""],width=2.0)
savefig(plt_tvlqr_nom,joinpath(@__DIR__,"results/cartpole_friction_tvlqr_nom_sim.png"))


z_tvlqr_friction, u_tvlqr_friction, J_tvlqr_friction, Jx_tvlqr_friction, Ju_tvlqr_friction = simulate_cartpole_friction(K_friction_nominal,
    X_friction_nominal,U_friction_nominal,
    model_sim,Q_lqr,R_lqr,T_sim,Δt,X_friction_nominal[1],w,ul=ul_friction,uu=uu_friction,friction=true,
    μ=μ_sim)
plt_tvlqr_friction = plot(t_nominal,hcat(X_friction_nominal...)[1:2,:]',
    color=:red,label=["nominal (sample)" ""],
    width=2.0,xlabel="time",title="Cartpole",ylabel="state",legend=:topleft)
plt_tvlqr_friction = plot!(t_sim_nominal,hcat(z_tvlqr_friction...)[1:2,:]',
    color=:magenta,label=["tvlqr" ""],width=2.0)
savefig(plt_tvlqr_friction,joinpath(@__DIR__,"results/cartpole_friction_tvlqr_friction_sim.png"))

z_sample, u_sample, J_sample, Jx_sample, Ju_sample = simulate_cartpole_friction(K_sample,
    X_nom_sample,U_nom_sample,
    model_sim,Q_lqr,R_lqr,T_sim,Δt,X_nom_sample[1],w,ul=ul_friction,uu=uu_friction,friction=true,
    μ=μ_sim)

plt_sample = plot(t_sample,hcat(X_nom_sample...)[1:2,:]',legend=:bottom,color=:red,
    label=["nominal (sample)" ""],
    width=2.0,xlabel="time",title="Cartpole",ylabel="state")
plt_sample = plot!(t_sim_nominal,hcat(z_sample...)[1:2,:]',color=:orange,
    label=["sample" ""],width=2.0,legend=:topleft)
savefig(plt_sample,joinpath(@__DIR__,"results/cartpole_friction_sample_sim.png"))

# z_sample_c, u_sample_c, J_sample_c, Jx_sample_c, Ju_sample_c = simulate_cartpole_friction(K_sample_coefficients,
#     X_nom_sample_coefficients,U_nom_sample_coefficients,
#     model_sim,Q_lqr,R_lqr,T_sim,Δt,X_nom_sample_coefficients[1],w,ul=ul_friction,
#     uu=uu_friction,friction=true,
#     μ=μ_sim)

# plt_sample = plot(t_sample_c,hcat(X_nom_sample_coefficients...)[1:2,:]',
#     legend=:bottom,color=:red,label=["nominal (sample)" ""],
#     width=2.0,xlabel="time (s)",title="Cartpole",ylabel="state")
# plt_sample = plot!(t_sim_nominal,hcat(z_sample_c...)[1:2,:]',color=:magenta,
#     label=["sample" ""],width=2.0,legend=:topleft)
# savefig(plt_sample,joinpath(@__DIR__,"results/cartpole_friction_sample_sim.png"))

# objective value
J_tvlqr
J_tvlqr_friction
J_sample
# J_sample_c

# state tracking
Jx_tvlqr
Jx_tvlqr_friction
Jx_sample
# Jx_sample_c

# control tracking
Ju_tvlqr
Ju_tvlqr_friction
Ju_sample
# Ju_sample_c

# Visualize
# include("../dynamics/visualize.jl")
# vis = Visualizer()
# open(vis)
#
# visualize!(vis,model,[z_tvlqr,z_tvlqr_friction,z_sample],
#     color=[RGBA(128/255,0/255,128/255),RGBA(255/255,0/255,255/255),RGBA(255/255,165/255,0/255,1.0)],
#     Δt=dt_sim)

using PGFPlots
const PGF = PGFPlots

# nominal trajectory
psx_nom = PGF.Plots.Linear(t_nominal,hcat(X_nominal...)[1,:],mark="",
	style="color=cyan, line width=3pt")
psθ_nom = PGF.Plots.Linear(t_nominal,hcat(X_nominal...)[2,:],mark="",
	style="color=cyan, line width=3pt, densely dashed")

psx_nom_sim_tvlqr = PGF.Plots.Linear(t_sim_nominal,hcat(z_tvlqr...)[1,:],mark="",
	style="color=black, line width=2pt")
psθ_nom_sim_tvlqr = PGF.Plots.Linear(t_sim_nominal,hcat(z_tvlqr...)[2,:],mark="",
	style="color=black, line width=2pt, densely dashed")

a = Axis([psx_nom;psθ_nom;psx_nom_sim_tvlqr;psθ_nom_sim_tvlqr],
    xmin=0., ymin=-1, xmax=sum(H_nominal), ymax=3.5,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"cartpole_sim_tvlqr_no_friction.tikz"), a, include_preamble=false)

# nominal trajectory
psx_nom_friction = PGF.Plots.Linear(t_nominal,hcat(X_friction_nominal...)[1,:],mark="",
	style="color=purple, line width=3pt")
psθ_nom_friction = PGF.Plots.Linear(t_sample,hcat(X_friction_nominal...)[2,:],mark="",
	style="color=purple, line width=3pt, densely dashed")

psx_nom_sim_tvlqr_friction = PGF.Plots.Linear(t_sim_nominal,hcat(z_tvlqr_friction...)[1,:],mark="",
	style="color=black, line width=2pt",legendentry="pos.")
psθ_nom_sim_tvlqr_friction = PGF.Plots.Linear(t_sim_nominal,hcat(z_tvlqr_friction...)[2,:],mark="",
	style="color=black, line width=2pt, densely dashed",legendentry="ang.")

a_tvlqr_friction = Axis([psx_nom_friction;psθ_nom_friction;
    psx_nom_sim_tvlqr_friction;psθ_nom_sim_tvlqr_friction],
    xmin=0., ymin=-1, xmax=sum(H_nominal), ymax=3.5,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"cartpole_sim_tvlqr_friction.tikz"), a_tvlqr_friction, include_preamble=false)

# nominal trajectory
psx_dpo = PGF.Plots.Linear(t_sample,hcat(X_nom_sample...)[1,:],mark="",
	style="color=orange, line width=3pt")
psθ_dpo = PGF.Plots.Linear(t_sample,hcat(X_nom_sample...)[2,:],mark="",
	style="color=orange, line width=3pt, densely dashed")

psx_sim_dpo = PGF.Plots.Linear(t_sim_sample,hcat(z_sample...)[1,:],mark="",
	style="color=black, line width=2pt",legendentry="pos.")
psθ_sim_dpo = PGF.Plots.Linear(t_sim_sample,hcat(z_sample...)[2,:],mark="",
	style="color=black, line width=2pt, densely dashed",legendentry="ang.")

a_dpo = Axis([psx_dpo;psθ_dpo;psx_sim_dpo;
    psθ_sim_dpo],
    xmin=0., ymin=-1, xmax=sum(H_nom_sample), ymax=3.5,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"cartpole_sim_dpo.tikz"), a_dpo, include_preamble=false)
