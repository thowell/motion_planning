# visualization
function visualize!(vis, model::DoublePendulum, x;
        color=RGBA(0.0, 0.0, 0.0, 1.0),
        r = 0.1, Δt = 0.1)


    i = 1
    l1 = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l1),
        convert(Float32, 0.025))
    setobject!(vis["l1$i"], l1, MeshPhongMaterial(color = color))
    l2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l2),
        convert(Float32, 0.025))
    setobject!(vis["l2$i"], l2, MeshPhongMaterial(color = color))

    setobject!(vis["elbow$i"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = color))
    setobject!(vis["ee$i"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = color))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(x)
    for t = 1:T

        MeshCat.atframe(anim,t) do
            p_mid = [kinematics_elbow(model, x[t])[1], 0.0, kinematics_elbow(model, x[t])[2]]
            p_ee = [kinematics(model, x[t])[1], 0.0, kinematics(model, x[t])[2]]

            settransform!(vis["l1$i"], cable_transform(zeros(3), p_mid))
            settransform!(vis["l2$i"], cable_transform(p_mid, p_ee))

            settransform!(vis["elbow$i"], Translation(p_mid))
            settransform!(vis["ee$i"], Translation(p_ee))
        end
    end

    # settransform!(vis["/Cameras/default"],
    #    compose(Translation(0.0 , 0.0 , 0.0), LinearMap(RotZ(pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end

# visualization
function visualize_elbow!(vis, model::DoublePendulum, x;
        color=RGBA(0.0, 0.0, 0.0, 1.0),
        r = 0.1, Δt = 0.1)

    i = 1
    l1 = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l1),
        convert(Float32, 0.025))
    setobject!(vis["l1"], l1, MeshPhongMaterial(color = color))
    l2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l2),
        convert(Float32, 0.025))
    setobject!(vis["l2"], l2, MeshPhongMaterial(color = color))

    setobject!(vis["elbow_nominal"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
	setobject!(vis["elbow_limit"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 1.0, 0.0, 1.0)))
    setobject!(vis["ee"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	ϵ = 1.0e-1
    T = length(x)
    for t = 1:T

        MeshCat.atframe(anim,t) do
            p_mid = [kinematics_elbow(model, x[t])[1], 0.0, kinematics_elbow(model, x[t])[2]]
            p_ee = [kinematics(model, x[t])[1], 0.0, kinematics(model, x[t])[2]]

            settransform!(vis["l1"], cable_transform(zeros(3), p_mid))
            settransform!(vis["l2"], cable_transform(p_mid, p_ee))

            settransform!(vis["elbow_nominal"], Translation(p_mid))
			settransform!(vis["elbow_limit"], Translation(p_mid))
            settransform!(vis["ee"], Translation(p_ee))

			if x[t][2] <= -0.5 * π + ϵ || x[t][2] >= 0.5 * π - ϵ
				setvisible!(vis["elbow_nominal"], false)
				setvisible!(vis["elbow_limit"], true)
			else
				setvisible!(vis["elbow_nominal"], true)
				setvisible!(vis["elbow_limit"], false)
			end
        end
    end

    # settransform!(vis["/Cameras/default"],
    #    compose(Translation(0.0 , 0.0 , 0.0), LinearMap(RotZ(pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
