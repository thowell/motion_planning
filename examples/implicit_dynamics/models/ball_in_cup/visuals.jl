# Visualization
function visualize!(mvis, model::KukaParticle, q;
		verbose = false, r_ball = 0.035, Δt = 0.1)

	setobject!(vis["ball"], Sphere(Point3f0(0),
				convert(Float32, r_ball)),
				MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0,1.0)))

	settransform!(vis["ball"], compose(Translation(0.66,3.0,0.0)))

	state = model.state_cache1[Float64]

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T
        q_kuka = kuka_q(q[t])
		q_particle = particle_q(q[t])
		set_configuration!(state, kuka_q(q[t]))

        MeshCat.atframe(anim,t) do
			set_configuration!(mvis,q_kuka)
            settransform!(vis["ball"], compose(Translation(q_particle), LinearMap(RotZ(0))))
		end
    end

    MeshCat.setanimation!(vis, anim)
end
