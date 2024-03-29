"""
    simulate
    - solves 1-step feasibility problem for T time steps
    - initial configurations: q1, q2 (note this can encode initial velocity)
    - time step: h
"""
function simulate(q1, q2, T, h;
        u1 = [zeros(nu) for t = 1:T],
        r_tol = 1.0e-5,
        μ_tol = 1.0e-5,
        z_init = 1.0,
        μ_init = 1.0)

    println("simulation")

    # initialize histories
    q = [q1, q2]
    n = []
    b = []
    Δq1 = []
    Δq2 = []
    Δu1 = []

    # u1 = zeros(3)

    # step
    for t = 1:T
        println("   t = $t")
        q3, n1, b1, _Δq1, _Δq2, _Δu1, status = step(q[end-1], q[end], u1[t], h,
            r_tol = r_tol,
            μ_tol = μ_tol,
            z_init = z_init,
            μ_init = μ_init)

        if !status
            @error "failed step (t = $t)"
            return q, n, b
        else
            push!(q, q3)
            push!(n, n1)
            push!(b, b1)
            push!(Δq1, _Δq1)
            push!(Δq2, _Δq2)
            push!(Δu1, _Δu1)
        end
    end

    return q, n, b, Δq1, Δq2, Δu1
end
