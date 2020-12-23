using LinearAlgebra
using ForwardDiff
using Ipopt
using Plots
using MathOptInterface
const MOI = MathOptInterface

include("kuka_particle_model.jl")

N = 21
NN = (N-1)*(nq+nu+nγ+ns) + 2*nq
#
# q0 = [0.;0.]
# q1 = [0.;0.]
# qN = [1.;1.]

# q0 = [0.;1.;0.;0.0]
# q1 = [0.;1.;0.;0.0]
# qN = [0.;1.;0.;1.1]

q0 = zeros(nq)
q0[1] = 0
q0[3] = 0
q0[4] = -pi/2
q0[5] = 0.
ee_pos0 = Array(end_effector_function(q0[1:7]))
p_pos0 = ee_pos0
p_pos0[3] -= 0.5
q0[8:10] = p_pos0


q1 = copy(q0)

qN = zeros(nq)
qN[1] = 0
qN[3] = 0
qN[4] = -pi/2
qN[5] = 0.

ee_posN = Array(end_effector_function(qN[1:7]))
p_posN = ee_posN
p_posN[3] += 0.1
# p_posN[2] += 0.5
qN[8:10] = p_posN


qD = zeros(nq)
qD[1] = 0
qD[3] = 0
qD[4] = -pi/2
qD[5] = 0.

ee_posD = Array(end_effector_function(qD[1:7]))
p_posD = ee_posD
qD[8:10] = p_posD


Q = [rand(nq) for k = 1:N+1]
U = [rand(nu) for k = 1:N-1]
Γ = [rand(nγ) for k = 1:N-1]
S = [rand(ns) for k = 1:N-1]

function hold_trajectory(n,m,N, mech::Mechanism, q)
    state = MechanismState(mech)
    nn = num_positions(state)
    set_configuration!(state, q[1:nn])
    vd = zero(state.q)
    u0 = dynamics_bias(state)

    if length(q) > m
        throw(ArgumentError("system must be fully actuated to hold an arbitrary position ($(length(q)) should be > $m)"))
    end
    U0 = zeros(m,N)
    for k = 1:N
        U0[:,k] = u0
    end
    return U0
end

U0_hold = hold_trajectory(7,7,N,mechanismkuka,qN[1:7])
U_hold = [U0_hold[:,k]*h for k = 1:N-1]
uu = zeros(7)
uu = 0.01*rand(7)
# uu[1] = 0.1
# uu[3] = 0.1
# uu[5] = 0.1
U_hold_rand = [U0_hold[:,k]*h .+ uu for k = 1:N-1]

function add_rows_cols!(row,col,_r,_c)
    for cc in _c
        for rr in _r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
end

function packZ(Q,U,Γ,S)
    Z = copy(Q[1])
    for k = 1:N-1
        append!(Z,[Q[k+1];U[k];Γ[k];S[k]])
    end
    append!(Z,Q[N+1])

    return Z
end

unpackZ(Z) = let nq=nq, nu=nu, nγ=nγ, ns=ns, N=N, NN=NN
    shift = nq+nu+nγ+ns
    Q = [k == 1 ? Z[1:nq] : Z[((k-2)*shift+nq) .+ (1:nq)] for k = 1:N+1]
    U = [Z[((k-1)*shift+2nq) .+ (1:nu)] for k = 1:N-1]
    Γ = [Z[((k-1)*shift+2nq+nu) .+ (1:nγ)] for k = 1:N-1]
    S = [Z[((k-1)*shift+2nq+nu+nγ) .+ (1:ns)] for k = 1:N-1]

    return Q,U,Γ,S
end

ZZ = packZ(Q,U,Γ,S)
# Q_,U_,Γ_,S_ = unpackZ(ZZ)
# Q_ == Q
# U_ == U
# Γ_ == Γ
# S_ == S

W = 1.0*Diagonal([ones(7);0.;0.;0.])
WN = 100.0*Diagonal(ones(nq))#Diagonal([10.0*ones(7);1.;1.;1.])
Ru = 1.0*Diagonal(ones(nu))
# Ru[1,1] = 1.0e-3
# Ru[3,3] = 1.0e-3
# Ru[5,5] = 1.0e-3
Rγ = 0.0*Diagonal(ones(nγ))
Rs = 10000.0*Diagonal(ones(1))

Rs*ones(ns)
g(q,u,γ,s,k,qd=qN,scale=diag(W)) = let W=W, Ru=Ru, Rγ=Rγ, Rs=Rs, qN=qN
    (q - qd)'*Diagonal(scale)*(q - qd) + (u - U_hold[k])'*Ru*(u - U_hold[k]) + γ'*Rγ*γ + sum(Rs*s)
end

dgdq(q,u,γ,s,qd=qN,scale=diag(W)) = let W=W, qN=qN
    2*Diagonal(scale)*(q - qd)
end

dgdu(q,u,γ,s,k) = let R=Ru
    2*R*(u - U_hold[k])
end

dgdγ(q,u,γ,s) = let R=Rγ
    2*R*γ
end

dgds(q,u,γ,s) = let R=Rs, ns=ns
    R*ones(ns)
end

gN(q) = let WN=WN, qN=qN
    (q - qN)'*WN*(q - qN)
end

dgNdq(q) = let WN=WN
    2*WN*(q - qN)
end

cost(Z) = let N=N, g=g, gN=gN, W=W, qN=qN
    J = 0.
    Q,U,Γ,S = unpackZ(Z)

    J += (Q[1] - qN)'*W*(Q[1] - qN)

    for k = 1:N-1
        q = Q[k+1]
        u = U[k]
        γ = Γ[k]
        s = S[k]

        qd = qN
        scale = diag(W)
        if k == 11
            qd=copy(qD)
            qd[8] += 0.5
            # scale = zeros(nq)
            scale[8] = 1000.0
        end
        # elseif k == 16
        #     qd=copy(qD)
        #     qd[10] += 0.5
        #     scale = zeros(nq)
        #     scale[10] = 1000.0
        # elseif k == 26
        #     qd=copy(qD)
        #     qd[10] += 0.5
        #     scale = zeros(nq)
        #     scale[8:10] .= 1000.0
        # else
        #     qd=qN
        #     scale = diag(W)
        # end
        J += g(q,u,γ,s,k,qd,scale)
    end

    qN = Q[N+1]
    J += gN(qN)

    return J
end

cost(ZZ)

∇cost!(∇J,Z) = let nq=nq, nu=nu, nγ=nγ, ns=ns, N=N, NN=NN, qN=qN
    J = 0.
    shift = nq+nu+nγ+ns

    Q,U,Γ,S = unpackZ(Z)

    ∇J[1:nq] = 2*W*(Q[1] - qN)

    for k = 1:N-1
        q = Q[k+1]
        u = U[k]
        γ = Γ[k]
        s = S[k]

        qd = qN
        scale = diag(W)
        if k == 11
            qd=copy(qD)
            qd[8] += 0.5
            # scale = zeros(nq)
            scale[8] = 1000.0
        end
        # elseif k == 16
        #     qd=copy(qD)
        #     qd[10] += 0.5
        #     scale = zeros(nq)
        #     scale[10] = 1000.0
        # elseif k == 26
        #     qd=copy(qD)
        #     qd[10] += 0.5
        #     scale = zeros(nq)
        #     scale[8:10] .= 1000.0
        # else
        #     qd=qN
        #     scale = diag(W)
        # end

        ∇J[((k-1)*shift+nq) .+ (1:nq)] = dgdq(q,u,γ,s,qd,scale)
        ∇J[((k-1)*shift + 2nq) .+ (1:nu)] = dgdu(q,u,γ,s,k)
        ∇J[((k-1)*shift + 2nq + nu) .+ (1:nγ)] = dgdγ(q,u,γ,s)
        ∇J[((k-1)*shift + 2nq + nu + nγ) .+ (1:ns)] = dgds(q,u,γ,s)
    end
    ∇J[((N-1)*shift+nq) .+ (1:nq)] = dgNdq(Q[N+1])

    return nothing
end

JJ = zeros(NN)
∇cost!(JJ,ZZ)
JJ

dynamics_constraints!(g,Z) = let nq=nq, N=N
    Q,U,Γ,S = unpackZ(Z)
    shift = nq

    for k = 1:N-1
        q⁺ = Q[k+2]
        q = Q[k+1]
        q⁻ = Q[k]
        u = U[k]
        γ = Γ[k]
        s = S[k]

        g[(k-1)*shift .+ (1:shift)] = var_int(q⁻,q,u,γ,s,q⁺)
    end
    return nothing
end


gg = rand(nq*(N-1))
dynamics_constraints!(gg,ZZ)
gg

∇dynamics_constraints!(jj,Z) = let nq=nq, nu=nu, nγ=nγ, ns=ns, N=N, NN=NN
    Q,U,Γ,S = unpackZ(Z)
    shift = nq*(3nq + nu + nγ + ns)

    for k = 1:N-1
        q⁺ = Q[k+2]
        q = Q[k+1]
        q⁻ = Q[k]
        u = U[k]
        γ = Γ[k]
        s = S[k]
        copyto!(view(jj,(k-1)*shift .+ (1:shift)),vec(∇var_int(q⁻,q,u,γ,s,q⁺)))
    end
    return nothing
end

jj = zeros((N-1)*nq*(3nq + nu + nγ + ns))
∇dynamics_constraints!(jj,ZZ)
jj

sparsity_dynamics(r_shift=0) = let nq=nq, nu=nu, nγ=nγ, ns=ns, N=N, NN=NN
    row = []
    col = []
    r_shift = r_shift
    c_shift = 0
    shift = nq + nu + nγ + ns

    for k = 1:N-1
        r_idx = r_shift .+ (1:nq)

        if k == 1
            c_idx = 1:(shift + 2*nq)
        else
            c_idx = (c_shift) .+ [(1:nq)...,(shift .+ (1:(shift+nq)))...]
        end

        add_rows_cols!(row,col,r_idx,c_idx)

        r_shift += nq
        if k == 1
            c_shift += nq
        else
            c_shift += shift
        end
    end

    collect(zip(row,col))
end

sparsity_dynamics()

(N-1)*(3nq + nu + nγ + ns)*nq

signed_distance_constraints!(g,Z) = let nγ=nγ, N=N
    Q,U,Γ,S = unpackZ(Z)

    for k = 1:N+1
        copyto!(view(g,(k-1)*nγ .+ (1:nγ)),-ϕ(Q[k]))
    end
end

∇signed_distance_constraints!(jj,Z) = let nq=nq, nγ=nγ, N=N
    Q,U,Γ,S = unpackZ(Z)
    shift = nγ*nq

    for k = 1:N+1
        copyto!(view(jj,(k-1)*shift .+ (1:shift)),-vec(∇ϕ(Q[k])))
    end
end

gg = zeros((N+1)*nγ)
signed_distance_constraints!(gg,ZZ)
gg

sparsity_signed_distance(r_shift=0) = let nq=nq, nu=nu, nγ=nγ, ns=ns, N=N
    row = []
    col = []
    r_shift = r_shift
    c_shift = 0
    shift = nq + nu + nγ + ns

    for k = 1:N+1
        r_idx = r_shift .+ (1:nγ)
        c_idx = c_shift .+ (1:nq)

        add_rows_cols!(row,col,r_idx,c_idx)

        r_shift += nγ
        if k == 1
            c_shift += nq
        else
            c_shift += shift
        end
    end

    collect(zip(row,col))
end

sparsity_signed_distance()
(N+1)*(nγ*nq)

complementary_con(γ,s,q⁺) = dot(γ,ϕ(q⁺)) - s[1]
complementary_con(z) = complementary_con(z[1:nγ],z[nγ .+ (1:ns)],z[(nγ+ns) .+ (1:nq)])
∇complementary_con(z) = ForwardDiff.gradient(complementary_con,z)
∇complementary_con(γ,s,q⁺) = ∇complementary_con([γ;s;q⁺])
∇complementary_con(rand(nγ),rand(ns),rand(nq))

complementarity_constraints!(g,Z) = let N=N
    Q,U,Γ,S = unpackZ(Z)

    for k = 1:N-1
        g[k] = complementary_con(Γ[k],S[k],Q[k+2])
    end
end

gg = zeros(N-1)
complementarity_constraints!(gg,ZZ)
gg

∇complementarity_constraints!(jj,Z) = let nq=nq, nγ=nγ, ns=ns, N=N
    Q,U,Γ,S = unpackZ(Z)
    shift = nq+nγ+ns

    for k = 1:N-1
        copyto!(view(jj,(k-1)*shift .+ (1:shift)),∇complementary_con(Γ[k],S[k],Q[k+2]))
    end
end

jj = zeros((N-1)*(nq+nγ+ns))
∇complementarity_constraints!(jj,ZZ)
jj

sparsity_complementarity(r_shift=0) = let nq=nq, nu=nu, nγ=nγ, ns=ns, N=N
    row = []
    col = []
    r_shift = r_shift
    c_shift = 0
    shift = nq + nu + nγ + ns
    idx = nq .+ [((nq+nu) .+ (1:(nγ+ns+nq)))...]

    for k = 1:N-1

        r_idx = r_shift + 1
        c_idx = c_shift .+ idx

        add_rows_cols!(row,col,r_idx,c_idx)

        r_shift += 1
        c_shift += shift
    end
    collect(zip(row,col))
end

sparsity_complementarity()
(N-1)*(nq+nγ+ns)

primal_bounds(q0,q1,qN) = let nq=nq, nu=nu, nγ=nγ, ns=ns, N=N, NN=NN
    Z_low = -Inf*ones(NN)
    Z_up = Inf*ones(NN)

    shift = nq + nu + nγ + ns

    # set γ, s >= 0
    for k = 1:N-1
        Z_low[((k-1)*shift + 2nq + nu) .+ (1:(nγ+ns))] .= 0.
        # Z_up[((k-1)*shift + 2nq + nu) .+ (1:(nγ+ns))] .= 0.
    end

    # # control limits
    # for k = 1:N-1
    #     Z_low[((k-1)*shift + 2nq) .+ (1:(nu))] = [-1.0;0.;-1.0]
    #     Z_up[((k-1)*shift + 2nq) .+ (1:(nu))] = [1.0;0.;1.0]
    # end

    # initial condition
    Z_low[1:nq] = q0
    Z_up[1:nq] = q0
    Z_low[nq .+ (1:nq)] = q1
    Z_up[nq .+ (1:nq)] = q1

    # terminal condition
    Z_low[NN-(nq-1):NN] = qN
    Z_up[NN-(nq-1):NN] = qN

    Z_low, Z_up
end

# primal_bounds(rand(nq),rand(nq))

function constraint_bounds(con)
    con_low = Float64[]
    con_up = Float64[]

    for c in con
        append!(con_up,zeros(c[1]))

        if c[2] == :inequality
            append!(con_low,-Inf*ones(c[1]))
        else
            append!(con_low,zeros(c[1]))
        end
    end
    con_low, con_up
end

# constraint_bounds([(5,:inequality),(6,:eq)])

function line_trajectory(x0::Vector,xf::Vector,N::Int)
    t = range(0,stop=N,length=N)
    slope = (xf .- x0)./N
    x_traj = [slope*t[k] + x0 for k = 1:N]
    x_traj[1] = x0
    x_traj[end] = xf
    x_traj
end
