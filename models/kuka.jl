using MeshCat, MeshCatMechanisms, RigidBodyDynamics

include(joinpath(pwd(),"models/kuka/kuka_utils.jl"))
urdf_path = joinpath(pwd(), "models/kuka/temp/kuka.urdf")
kuka = MeshCatMechanisms.parse_urdf(urdf_path, remove_fixed_tree_joints = true)
kuka_visuals = MeshCatMechanisms.URDFVisuals(urdf_path)

state = MechanismState(kuka)
state_cache = StateCache(kuka)
result = DynamicsResult(kuka)
result_cache = DynamicsResultCache(kuka)

vis = Visualizer()
mvis = MechanismVisualizer(kuka, kuka_visuals, vis[:base])
open(vis)

# Kuka iiwa arm parsed from URDF using RigidBodyDynamics.jl
struct Kuka{T} <: Model
	n::Int
	m::Int
	d::Int

    state_cache1
    state_cache2
    state_cache3

    result_cache1
    result_cache2
    result_cache3
end

n = 14
m = 7
d = 0

results_cache1 = DynamicsResultCache(kuka)
results_cache2 = DynamicsResultCache(kuka)
results_cache3 = DynamicsResultCache(kuka)

state_cache1 = StateCache(kuka)
state_cache2 = StateCache(kuka)
state_cache3 = StateCache(kuka)

function f(model::Kuka, x, u, w)
	@error "not implemented"
end

model = Kuka(
	n, m, d,
    state_cache1, state_cache2, state_cache3,
    results_cache1, results_cache2, results_cache3)
