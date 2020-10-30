"""
    Interface to SNOPT. Modified from official repository
    (https://github.com/snopt/SNOPT7.jl) to have increased workspace size.
    SNOPT license is required.
"""

const libsnopt7 = "libsnopt7"

mutable struct snoptWorkspace
    status::Int

    # Workspace
    leniw::Int
    lenrw::Int
    iw::Vector{Int32}
    rw::Vector{Float64}

    leniu::Int
    lenru::Int
    iu::Vector{Int32}
    ru::Vector{Float64}

    x::Vector{Float64}
    lambda::Vector{Float64}
    obj_val::Float64

    num_inf::Int
    sum_inf::Float64

    iterations::Int
    major_itns::Int
    run_time::Float64

    function snoptWorkspace(leniw::Int,lenrw::Int)
        prob = new(0,leniw, lenrw,
                   zeros(Int,leniw), zeros(Float64,lenrw), 0, 0)
        finalizer(freeWorkspace!,prob)
        prob
    end
end

function freeWorkspace!(prob::snoptWorkspace)
    ccall((:f_snend, libsnopt7),
          Cvoid, (Ptr{Cint}, Cint, Ptr{Float64}, Cint),
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
end

# Exit codes
SNOPT_status = Dict(
    1=>:Solve_Succeeded,
    2=>:Feasible_Point_Found,
    3=>:Solved_To_Acceptable_Level,
    4=>:Solved_To_Acceptable_Level,
    5=>:Solved_To_Acceptable_Level,
    6=>:Solved_To_Acceptable_Level,
    11=>:Infeasible_Problem_Detected,
    12=>:Infeasible_Problem_Detected,
    13=>:Infeasible_Problem_Detected,
    14=>:Infeasible_Problem_Detected,
    15=>:Infeasible_Problem_Detected,
    16=>:Infeasible_Problem_Detected,
    21=>:Unbounded_Problem_Detected,
    22=>:Unbounded_Problem_Detected,
    31=>:Maximum_Iterations_Exceeded,
    32=>:Maximum_Iterations_Exceeded,
    34=>:Maximum_CpuTime_Exceeded,
    41=>:Numerical_Difficulties,
    42=>:Numerical_Difficulties,
    43=>:Numerical_Difficulties,
    44=>:Numerical_Difficulties,
    45=>:Numerical_Difficulties,
    51=>:User_Supplied_Function_Error,
    52=>:User_Supplied_Function_Error,
    56=>:User_Supplied_Function_Error,
    61=>:User_Supplied_Function_Undefined,
    62=>:User_Supplied_Function_Undefined,
    63=>:User_Supplied_Function_Undefined,
    71=>:User_Requested_Stop,
    72=>:User_Requested_Stop,
    73=>:User_Requested_Stop,
    74=>:User_Requested_Stop,
    81=>:Insufficient_Memory,
    82=>:Insufficient_Memory,
    83=>:Insufficient_Memory,
    91=>:Invalid_Problem_Definition,
    92=>:Invalid_Problem_Definition,
    999=>:Internal_Error)

# Callbacks
function obj_wrapper!(mode_::Ptr{Cint}, nnobj::Cint, x_::Ptr{Float64},
                      f_::Ptr{Float64}, g_::Ptr{Float64},
                      status::Cint)
    x    = unsafe_wrap(Array, x_, Int(nnobj))
    mode = unsafe_load(mode_)

    if mode == 0 || mode == 2
        obj = convert(Float64, eval_f(x)) :: Float64
        unsafe_store!(f_,obj)
    end

    if mode == 1 || mode == 2
        g = unsafe_wrap(Array,g_,Int(nnobj))
        eval_grad_f(x,g)
    end
    return
end

function con_wrapper!(mode_::Ptr{Cint}, nncon::Cint, nnjac::Cint, negcon::Cint,
                      x_::Ptr{Float64}, c_::Ptr{Float64}, J_::Ptr{Float64},
                      status::Cint)

    x    = unsafe_wrap(Array, x_, Int(nnjac))
    mode = unsafe_load(mode_)

    if mode == 0 || mode == 2
        c = unsafe_wrap(Array, c_, Int(nncon))
        eval_g(x, c)
    end

    if mode == 1 || mode == 2
        J = unsafe_wrap(Array,J_,Int(negcon))
        eval_jac_g(x, J)
    end
    return
end


# SNOPT7 routines
function initialize(printfile::String, summfile::String)
    prob = snoptWorkspace(30500,3000)

    ccall((:f_sninitx, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Ptr{UInt8}, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          printfile, length(printfile), summfile, length(summfile),
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return prob
end

function initialize(printfile::String, summfile::String,
                    leniw::Int, lenrw::Int)
    prob = snoptWorkspace(leniw, lenrw)

    ccall((:f_sninitx, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Cint,
           Ptr{UInt8}, Cint, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          printfile, plen, prob.iprint, summfile, slen, prob.isumm,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return prob
end

# make larger workspace
function initialize(printfile::String, summfile::String,ws::Tuple)
    prob = snoptWorkspace(ws[1],ws[2])

    ccall((:f_sninitx, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Ptr{UInt8}, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          printfile, length(printfile), summfile, length(summfile),
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return prob
end

function readOptions(prob::snoptWorkspace, specsfile::String)
    status = [0]
    ccall((:f_snspecf, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          specsfile, length(specsfile), status,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    prob.status = status[1]
    return Int(prob.status)
end

function snopt!(prob::snoptWorkspace, start::String, name::String,
                m::Int, n::Int, nnCon::Int, nnObj::Int, nnJac::Int,
                fObj::Float64, iObj::Int,
                confun::Function, objfun::Function,
                #eval_f::Function, eval_grad_f::Function,
                #eval_g::Function, eval_jac_g::Function,
                J::SparseMatrixCSC, bl::Vector{Float64}, bu::Vector{Float64},
                hs::Vector{Int}, x::Vector{Float64})

    @assert n+m == length(x) == length(bl) == length(bu)
    @assert n+m == length(hs)

    prob.iu = [0]
    prob.ru = [0.]

    prob.x      = copy(x)
    prob.lambda = zeros(Float64,n+m)
    pi          = zeros(Float64,m)

    obj_callback = @cfunction($objfun, Cvoid,
                             (Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                              Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}))
    con_callback = @cfunction($confun, Cvoid,
                             (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                              Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}))

    valJ = J.nzval
    indJ = convert(Array{Cint}, J.rowval)
    locJ = convert(Array{Cint}, J.colptr)
    neJ  = length(valJ)

    status  = [0]
    nS      = [0]
    nInf    = [0]
    sInf    = [0.0]
    obj_val = [0.0]
    miniw   = [0]
    minrw   = [0]

    ccall((:f_snoptb, libsnopt7), Cvoid,
          (Ptr{UInt8}, Ptr{UInt8},
           Cint, Cint, Cint, Cint, Cint, Cint,
           Cint, Cdouble,
           Ptr{Cvoid}, Ptr{Cvoid},
           Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint},
           Ptr{Float64}, Ptr{Float64}, Ptr{Cint},
           Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
           Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cint}, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          start, name, m, n, neJ, nnCon, nnObj, nnJac,
          iObj, fObj,
          con_callback, obj_callback,
          valJ, indJ, locJ,
          bl, bu, hs, prob.x, pi, prob.lambda,
          status, nS, nInf, sInf, obj_val,
          miniw, minrw,
          prob.iu, prob.leniu, prob.ru, prob.lenru,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)

    prob.status  = status[1]
    prob.obj_val = obj_val[1]

    prob.num_inf = nInf[1]
    prob.sum_inf = sInf[1]

    prob.iterations = prob.iw[421]
    prob.major_itns = prob.iw[422]

    prob.run_time   = prob.rw[462]

    return Int(prob.status)
end

function setOption!(prob::snoptWorkspace, optstring::String)
    # Set SNOPT7 option via string
    if !isascii(optstring)
        error("SNOPT7: Non-ASCII parameters not supported")
    end

    errors = [0]
    ccall((:f_snset, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          optstring, length(optstring), errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return errors[1]
end

function setOption!(prob::snoptWorkspace, keyword::String, value::Int)
    # Set SNOPT7 integer option
    if !isascii(keyword)
        error("SNOPT7: Non-ASCII parameters not supported")
    end

    errors = [0]
    ccall((:f_snseti, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          optstring, length(optstring), value, errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return errors[1]
end

function setOption!(prob::snoptWorkspace, keyword::String, value::Float64)
    # Set SNOPT7 real option
    if !isascii(keyword)
        error("SNOPT7: Non-ASCII parameters not supported")
    end

    errors = [0]
    ccall((:f_snseti, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Cdouble, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          optstring, length(optstring), value, errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return errors[1]
end

mutable struct VariableData
    lower::Float64
    upper::Float64
    x::Float64
    g::Vector{Float64}
end
VariableData() = VariableData(-Inf, Inf, 0.0, [])

mutable struct Optimizer <: MOI.AbstractOptimizer
    sense::MOI.OptimizationSense
    objective::Union{MOI.SingleVariable,MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64},Nothing}
    variables::Vector{VariableData}
    linear_constraints::Vector{VariableData}
    nonlin::Union{MOI.NLPBlockData,Nothing}

    n::Int
    nncon::Int
    nlcon::Int

    workspace::Union{snoptWorkspace,Nothing}
    options::Dict{String, Any}
end
Optimizer(;options...) = Optimizer(MOI.MIN_SENSE, nothing, [], [], nothing, 0, 0, 0, nothing, options)

function MOI.empty!(model::Optimizer)
    model.sense = MOI.MIN_SENSE
    model.objective = nothing
    empty!(model.variables)
    empty!(model.linear_constraints)
    model.nonlin = nothing
    model.n = 0
    model.nncon = 0
    model.nlcon = 0
end

function MOI.is_empty(model::Optimizer)
    return model.objective == nothing &&
           isempty(model.variables) &&
           isempty(model.linear_constraints) &&
           model.nonlin == nothing &&
           model.n == 0 && model.nncon == 0 && model.nlcon == 0
end

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false)
    return MOI.Utilities.default_copy_to(model, src, copy_names)
end

# MOI objects that SNOPT supports
MOI.supports(::Optimizer, ::MOI.NLPBlock) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.SingleVariable}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart,::Type{MOI.VariableIndex}) = true  #?
MOI.supports(::Optimizer, ::MOI.RawParameter) = true

MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.Interval{Float64}}) = true

MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.Interval{Float64}}) = true

MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.variables)

function MOI.set(model::Optimizer, p::MOI.RawParameter, value)
    model.options[p.name] = value
    return
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return
end

# Linear/quadratic objective
function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction,
                 func::Union{MOI.SingleVariable, MOI.ScalarAffineFunction, MOI.ScalarQuadraticFunction})
    model.objective = func
    return
end

# Variables
function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value::Float64)
    model.variables[vi.value].x = value
    return
end

function MOI.add_variable(model::Optimizer)
    push!(model.variables, VariableData())
    return MOI.VariableIndex(length(model.variables))
end

function MOI.add_variables(model::Optimizer, n::Int)
    return [MOI.add_variable(model) for i in 1:n]
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, lt::MOI.LessThan{Float64})
    # Upper bound on x
    vi = v.variable
    model.variables[vi.value].upper = lt.upper
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, gt::MOI.GreaterThan{Float64})
    # Lower bound on x
    vi = v.variable
    model.variables[vi.value].lower = gt.lower
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, eq::MOI.EqualTo{Float64})
    # Fixed variable
    vi = v.variable
    model.variables[vi.value].lower = eq.value
    model.variables[vi.value].upper = eq.value
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, bds::MOI.Interval{Float64})
    # Lower/upper bounds on x
    vi = v.variable
    model.variables[vi.value].lower = bds.lower
    model.variables[vi.value].upper = bds.upper
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{Float64}}(vi.value)
end

# Linear constraints
macro define_add_constraint(set_type)
    quote
        function MOI.add_constraint(model::Optimizer, v::MOI.ScalarAffineFunction{Float64},
                                    set::$set_type)
            # Bounds on constraint
            x = VariableData()
            if typeof(set) == MOI.LessThan{Float64}
                x.upper = set.upper - v.constant
            elseif typeof(set) == MOI.GreaterThan{Float64}
                x.lower = set.lower - v.constant
            elseif typeof(set) == MOI.EqualTo{Float64}
                x.lower = set.value - v.constant
                x.upper = set.value - v.constant
            elseif typeof(set) == MOI.Interval{Float64}
                x.lower = set.lower - v.constant
                x.upper = set.upper - v.constant
            end

            x.g = [ t.coefficient for t in v.terms ]

            push!(model.linear_constraints, x)
            return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},$set_type}(length(model.linear_constraints))
        end
    end
end
@define_add_constraint(MOI.LessThan{Float64})
@define_add_constraint(MOI.GreaterThan{Float64})
@define_add_constraint(MOI.EqualTo{Float64})
@define_add_constraint(MOI.Interval{Float64})

# Nonlinear constraints
function MOI.set(model::Optimizer, ::MOI.NLPBlock, nonlin::MOI.NLPBlockData)
    model.nonlin = nonlin
    return
end

# Solve
function MOI.optimize!(model::Optimizer;
        ws::Tuple=(),outfile::String="snopt.out")

    n     = length(model.variables)
    nncon = model.nonlin !== nothing ? length(model.nonlin.constraint_bounds) : 0
    nlcon = length(model.linear_constraints)

    m     = nncon + length(model.linear_constraints)

    model.n = n
    model.nncon = nncon
    model.nlcon = nlcon

    # Bounds/initial point
    #    Variables
    bl = [v.lower for v in model.variables]
    bu = [v.upper for v in model.variables]
    x  = [v.x     for v in model.variables]
    #    Nonlinear constraints
    if nncon > 0
        for bound in model.nonlin.constraint_bounds
            push!(bl, bound.lower)
            push!(bu, bound.upper)
            push!(x,  0.0)
        end
    end

    #    Linear constraints
    for v in model.linear_constraints
        push!(bl, v.lower)
        push!(bu, v.upper)
        push!( x, v.x)
    end

    # Nonlinear data
    if nncon > 0
        features = MOI.features_available(model.nonlin.evaluator)
        has_objgrad = (:Grad in features)
        has_jacobian = (:Jac in features)

        MOI.initialize(model.nonlin.evaluator, features)

        # Jacobian
        sparseJ = MOI.jacobian_structure(model.nonlin.evaluator)  # returns tuple of coordinates
        nnjac = n

        j_rows = [sparseJ[i][1] for i in 1:length(sparseJ)]
        j_cols = [sparseJ[i][2] for i in 1:length(sparseJ)]
        j_vals = zeros(length(sparseJ))
    else
        nnjac  = 0
        j_rows = []
        j_cols = []
        j_vals = []
    end

    #   Add linear constraint Jacobian
    i = nncon
    for v in model.linear_constraints
        i = i+1
        for j in 1:n
            push!(j_rows, i)
            push!(j_cols, j)
            push!(j_vals, v.g[j])
        end
    end
    J = sparse(j_rows, j_cols, j_vals)

    # Minimize or maximize?




    if model.nonlin !== nothing
        # Nonlinear constraints/objective
        nnobj = n

        # Callbacks
        function eval_obj(mode_::Ptr{Cint}, n_::Ptr{Cint}, x_::Ptr{Float64},
                          f_::Ptr{Float64}, g_::Ptr{Float64}, status::Ptr{Cint})
            mode = unsafe_load(mode_)
            n    = unsafe_load(n_)

            x    = unsafe_wrap(Array, x_, Int(n))

            if mode == 0 || mode == 2
                if model.nonlin.has_objective
                    obj = MOI.eval_objective(model.nonlin.evaluator, x)
                elseif model.objective !== nothing
                    obj = eval_function(model.objective, x)
                else
                    obj = 0.0
                end
                unsafe_store!(f_,obj)
            end

            if mode == 1 || mode == 2
                g = unsafe_wrap(Array,g_, Int(n))
                if model.nonlin.has_objective
                    MOI.eval_objective_gradient(model.nonlin.evaluator, g, x)
                end
            end
            return
        end

        function eval_con(mode_::Ptr{Cint},
                          nncon_::Ptr{Cint}, nnjac_::Ptr{Cint}, negcon_::Ptr{Cint},
                          x_::Ptr{Float64}, c_::Ptr{Float64}, J_::Ptr{Float64},
                          status::Ptr{Cint})
            mode   = unsafe_load(mode_)
            nncon  = unsafe_load(nncon_)
            nnjac  = unsafe_load(nnjac_)
            negcon = unsafe_load(negcon_)

            x = unsafe_wrap(Array, x_, Int(nnjac))
            if mode == 0 || mode == 2
                c = unsafe_wrap(Array, c_, Int(nncon))
                MOI.eval_constraint(model.nonlin.evaluator, c, x)
            end

            if mode == 1 || mode == 2
                J = unsafe_wrap(Array,J_, Int(negcon))

                # MOI returns Jacobian in order specified by j_rows, j_cols
                # SNOPT needs it in sparse-by-col format
                JJ = zeros(negcon)
                MOI.eval_constraint_jacobian(model.nonlin.evaluator, JJ, x)
                vals = sparse(j_rows,j_cols,JJ).nzval
                for k in 1:negcon
                    J[k] = vals[k]
                end
            end
            return
        end

        # SNOPT
        fobj = 0.0
        iobj = 0
        hs   = zeros(Int,n+m)


        # make larger workspace
        if ws == ()
            # set workspace size (10x recommended by manual)
            ws = (1000*(n+m),2000*(n+m))
        end
        model.workspace = SNOPT7.initialize(outfile,"screen",ws)

        # model.workspace = SNOPT7.initialize("snopt.out","screen")

        for (name,value) in model.options
            # Replace underscore with space
            keyword = replace(String(name), "_" => " ")
            SNOPT7.setOption!(model.workspace, "$keyword $value")
        end
        SNOPT7.setOption!(model.workspace, "Summary frequency 1")
        SNOPT7.setOption!(model.workspace, "Print frequency 1")

        start = "Cold"
        name  = "prob"
        status = SNOPT7.snopt!(model.workspace, start, name,
                              m, n, nncon, nnobj, nnjac,
                              fobj, iobj, eval_con, eval_obj,
                              J, bl, bu, hs, x)
        SNOPT7.freeWorkspace!(model.workspace)

    else
        # Linear/quadratic objective

    end

end

# Return info
function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.workspace === nothing
        return MOI.NO_SOLUTION
    end

    status = SNOPT_status[model.workspace.status]
    if status in (:Solve_Succeeded,
                  :Feasible_Point_Found)
        return MOI.LOCALLY_SOLVED
    elseif status ==:Infeasible_Problem_Detected
        return MOI.LOCALLY_INFEASIBLE
    elseif status ==:Unbounded_Problem_Detected
        return MOI.INFEASIBLE_OR_UNBOUNDED
    elseif status == :Solved_To_Acceptable_Level
        return MOI.ALMOST_LOCALLY_SOLVED
    elseif status == :User_Requested_Stop
        return MOI.INTERRUPTED
    elseif status == :Maximum_Iterations_Exceeded
        return MOI.ITERATION_LIMIT
    elseif status == :Maximum_CpuTime_Exceeded
        return MOI.TIME_LIMIT
    elseif status == (:Invalid_Problem_Definition,
                      :User_Supplied_Function_Error,
                      :User_Supplied_Function_Undefined)
        return MOI.INVALID_MODEL
    elseif status == :Insufficient_Memory
        return MOI.MEMORY_LIMIT
    elseif status == :Numerical_Difficulties
        return MOI.NUMERICAL_ERROR
    else
        error("Unrecognized SNOPT7 status $status")
    end
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.workspace !== nothing) ? 1 : 0
end

function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
    if model.workspace === nothing
        return MOI.NO_SOLUTION
    end
    status = SNOPT_status[model.workspace.status]
    if status == :Solve_Succeeded
        return MOI.FEASIBLE_POINT
    elseif status == :Feasible_Point_Found
        return MOI.FEASIBLE_POINT
    elseif status == :Solved_To_Acceptable_Level
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == :Infeasible_Problem_Detected
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(model::Optimizer, ::MOI.DualStatus)
    if model.workspace === nothing
        return MOI.NO_SOLUTION
    end
    status = SNOPT_status[model.workspace.status]
    if status == :Solve_Succeeded
        return MOI.FEASIBLE_POINT
    elseif status == :Feasible_Point_Found
        return MOI.FEASIBLE_POINT
    elseif status == :Solved_To_Acceptable_Level
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == :Infeasible_Problem_Detected
        return MOI.UNKNOWN_RESULT_STATUS # unbounded?
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveValue)
    if model.workspace === nothing
        @error("ObjectiveValue not available.")
    end
    return model.workspace.obj_val
end

function MOI.get(model::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    if model.workspace === nothing
        @error("VariablePrimal not available.")
    end
    return model.workspace.x[vi.value]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,S}) where {S}
    if model.workspace === nothing
        @error("ConstraintPrimal not available.")
    end
    vi = MOI.VariableIndex(ci.value)
    return model.workspace.x[vi.value]
end

macro define_constraint_primal(set_type)
    quote
        function MOI.get(model::Optimizer, ::MOI.ConstraintPrimal,
                         ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, $set_type})
            if model.workspace === nothing
                @error("ConstraintPrimal not available.")
            end
            if !(1 <= ci.value <= model.nlcon)
                @error("Invalid constraint index ", ci.value)
            end
            return model.workspace.x[ci.value + model.n+model.nncon]
        end
    end
end
@define_constraint_primal(MOI.LessThan{Float64})
@define_constraint_primal(MOI.GreaterThan{Float64})
@define_constraint_primal(MOI.EqualTo{Float64})
@define_constraint_primal(MOI.Interval{Float64})


function MOI.get(model::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,S}) where {S}
    if model.workspace === nothing
        @error("ConstraintDual not available.")
    end
    vi = MOI.VariableIndex(ci.value)
    return model.workspace.lambda[vi.value]
end


function MOI.get(model::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}) where {S}
    if model.workspace === nothing
        @error("ConstraintDual not available.")
    end
    @assert 1 <= ci.value <= model.nlcon
    return model.workspace.lambda[ci.value + model.n + model.nncon]
end

function MOI.get(model::Optimizer, ::MOI.NLPBlockDual)
    if model.workspace === nothing
        @error("NLPBlockDual not available.")
    end
    return model.workspace.lambda[model.n+1:model.n+model.nncon]
end
