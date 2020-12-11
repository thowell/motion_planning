struct Quadrotor2DArm{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    # body
    lb    # length
    mb # mass
    Jb    # inertia

    # link 1
    l1
    lc1
    m1
    J1

    # link
    l2
    lc2
    m2
    J2

    g    # gravity
end

function kinematics(model::Quadrotor2DArm, q)
    @SVector [q[1] + model.l1 * sin(q[3] + q[4]) + model.l2 * sin(q[3] + q[4] + q[5]),
              q[2] - model.l1 * cos(q[3] + q[4]) - model.l2 * cos(q[3] + q[4] + q[5])]
end

function jacobian(model::Quadrotor2DArm, q)
    a = model.l1 * cos(q[3] + q[4]) + model.l2 * cos(q[3] + q[4] + q[5])
    b = model.l2 * cos(q[3] + q[4] + q[5])
    c = model.l1 * sin(q[3] + q[4]) + model.l2 * sin(q[3] + q[4] + q[5])
    d = model.l2 * sin(q[3] + q[4] + q[5])
    @SMatrix [1.0 0.0 a a b;
              0.0 1.0 c c d]
end

function B_func(model::Quadrotor2DArm, q)
    a = model.l1 * cos(q[3] + q[4]) + model.l2 * cos(q[3] + q[4] + q[5])
    b = model.l2 * cos(q[3] + q[4] + q[5])
    c = model.l1 * sin(q[3] + q[4]) + model.l2 * sin(q[3] + q[4] + q[5])
    d = model.l2 * sin(q[3] + q[4] + q[5])

    @SMatrix [-0.5 * model.lb * sin(q[3]) -0.5 * model.lb * sin(q[3]) 0.0 0.0 1.0 0.0;
              0.5 * model.lb * cos(q[3]) 0.5 * model.lb * cos(q[3]) 0.0 0.0 0.0 1.0;
              0.0 0.0 0.0 0.0 a c;
              0.0 0.0 1.0 0.0 a c;
              0.0 0.0 0.0 1.0 b d]
end
