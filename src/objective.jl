abstract type Objective end

"""
    empty objective
"""
struct EmptyObjective <: Objective end
objective(Z, obj::EmptyObjective, idx, T) = 0.0
objective_gradient!(∇J, Z, obj::EmptyObjective, idx, T) = nothing

"""
    multiple objectives
"""
struct MultiObjective <: Objective
    obj::Vector{Objective}
end

function objective(Z, obj::MultiObjective, idx, T)
    return sum([objective(Z, o, idx, T) for o in obj.obj])
end

function objective_gradient!(∇J, Z, obj::MultiObjective, idx, T)
    for o in obj.obj
        objective_gradient!(∇J, Z, o, idx, T)
    end
    return nothing
end

function include_objective(str::String)
    include(joinpath(pwd(), "src/objectives", str * ".jl"))
end

function include_objective(strs::Vector{String})
    for s in strs
        include_objective(s)
    end
end
