"""
    Acrobot
"""

struct Acrobot{T} <: Model
    n::Int
    m::Int
    d::Int

    m1::T    # mass link 1
    J1::T    # inertia link 1
    l1::T    # length link 1
    lc1::T   # length to COM link 1

    m2::T    # mass link 2
    J2::T    # inertia link 2
    l2::T    # length link 2
    lc2::T   # length to COM link 2

    g::T     # gravity

    b1::T    # joint friction
    b2::T
end

function M(model::Acrobot, x)
    a = (model.J1 + model.J2 + model.m2 * model.l1 * model.l1
         + 2.0 * model.m2 * model.l1 * model.lc2 * cos(x[2]))

    b = model.J2 + model.m2 * model.l1 * model.lc2 * cos(x[2])

    c = model.J2

    @SMatrix [a b;
              b c]
end

function τ(model::Acrobot, x)
    a = (-1.0 * model.m1 * model.g * model.lc1 * sin(x[1])
         - model.m2 * model.g * (model.l1 * sin(x[1])
         + model.lc2 * sin(x[1] + x[2])))

    b = -1.0 * model.m2 * model.g * model.lc2 * sin(x[1] + x[2])

    @SVector [a,
              b]
end

function C(model::Acrobot, x)
    a = -2.0 * model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[4]
    b = -1.0 * model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[4]
    c = model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[3]
    d = 0.0

    @SMatrix [a b;
              c d]
end

function B(model::Acrobot, x)
    @SMatrix [0.0;
              1.0]
end

function f(model::Acrobot, x, u, w)
    q = view(x, 1:2)
    v = view(x, 3:4)
    qdd = M(model, q) \ (-1.0 * C(model, x) * v
            + τ(model, q) + B(model, q) * u - [model.b1; model.b2] .* v)

    @SVector [x[3],
              x[4],
              qdd[1],
              qdd[2]]
end

function k_mid(model::Acrobot, x)
    @SVector [model.l1 * sin(x[1]),
              -1.0 * model.l1 * cos(x[1])]
end

function k_ee(model::Acrobot, x)
    @SVector [model.l1 * sin(x[1]) + model.l2 * sin(x[1] + x[2]),
              -1.0 * model.l1 * cos(x[1]) - model.l2 * cos(x[1] + x[2])]
end

n, m, d = 4, 1, 0
model = Acrobot(n, m , d, 1.0, 0.33, 1.0, 0.5, 1.0, 0.33, 1.0, 0.5, 9.81, 0.1, 0.1)

# visualization
function visualize!(vis, model::Acrobot, x;
        color=[RGBA(0.0, 0.0, 0.0, 1.0) for i = 1:length(q)],
        r = 0.1, Δt = 0.1)

    N = length(x)

    for i = 1:N
        l1 = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l1),
            convert(Float32, 0.025))
        setobject!(vis["l1$i"], l1, MeshPhongMaterial(color = color[i]))
        l2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l2),
            convert(Float32, 0.025))
        setobject!(vis["l2$i"], l2, MeshPhongMaterial(color = color[i]))

        setobject!(vis["elbow$i"], Sphere(Point3f0(0.0),
            convert(Float32, 0.05)),
            MeshPhongMaterial(color = color[i]))
        setobject!(vis["ee$i"], Sphere(Point3f0(0.0),
            convert(Float32, 0.05)),
            MeshPhongMaterial(color = color[i]))
    end

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(x[1])

        MeshCat.atframe(anim,t) do
            for i = 1:N
                p_mid = [k_mid(model, x[i][t])[1], 0.0, k_mid(model, q[i][t])[2]]
                p_ee = [k_ee(model, q[i][t])[1], 0.0, k_ee(model, q[i][t])[2]]

                settransform!(vis["l1$i"], cable_transform(zeros(3), p_mid))
                settransform!(vis["l2$i"], cable_transform(p_mid, p_ee))

                settransform!(vis["elbow$i"], Translation(p_mid))
                settransform!(vis["ee$i"], Translation(p_ee))
            end
        end
    end

    # settransform!(vis["/Cameras/default"],
    #    compose(Translation(0.0 , 0.0 , 0.0), LinearMap(RotZ(pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
