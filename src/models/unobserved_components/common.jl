mutable struct UCHyperparameters
    parameter_name::String
    variable_names::Vector{String}
    vals::Dict{String, Float64}
    initial_vals::Dict{String, Float64}

    function UCHyperparameters(parameter_name::String,
                                constant,
                                level,
                                ar_I_order,
                                ar_II_order,
                                seasonal_length)

        variable_names = String[]
        vals = Dict{String, Float64}()
        initial_vals = Dict{String, Float64}()

        if constant
            var_name = parameter_name * "_cst"
            push!(variable_names, var_name)
            vals[var_name] = NaN
            initial_vals[var_name] = NaN
            return new(parameter_name, variable_names, vals, initial_vals)
        end

        if level == "RW"
            var_name = parameter_name * "_RW_κ"
            push!(variable_names, var_name)
            vals[var_name] = NaN
            initial_vals[var_name] = NaN
        elseif level == "RW + slope"
            var_name = parameter_name * "_RW_κ"
            push!(variable_names, var_name)
            vals[var_name] = NaN
            initial_vals[var_name] = NaN
            var_name = parameter_name * "_slope_κ"
            push!(variable_names, var_name)
            vals[var_name] = NaN
            initial_vals[var_name] = NaN
        elseif level == "IRW"
            var_name = parameter_name * "_IRW_κ"
            push!(variable_names, var_name)
            vals[var_name] = NaN
            initial_vals[var_name] = NaN
        end

        if ar_I_order > 0
            var_name = parameter_name * "_AR_I_κ"
            push!(variable_names, var_name)
            vals[var_name] = NaN
            initial_vals[var_name] = NaN
            for i in 1:ar_I_order
                var_name = parameter_name * "_AR_I_ϕ_$i"
                push!(variable_names, var_name)
                vals[var_name] = NaN
                initial_vals[var_name] = NaN
            end
        end

        if ar_II_order > 0
            var_name = parameter_name * "_AR_II_κ"
            push!(variable_names, var_name)
            vals[var_name] = NaN
            initial_vals[var_name] = NaN
            for i in 1:ar_I_order
                var_name = parameter_name * "_AR_II_ϕ_$i"
                push!(variable_names, var_name)
                vals[var_name] = NaN
                initial_vals[var_name] = NaN
            end
        end

        if seasonal_length > 0
            var_name = parameter_name * "_seasonal_κ"
            push!(variable_names, var_name)
            vals[var_name] = NaN
            initial_vals[var_name] = NaN
            for i in 1:seasonal_length - 1
                var_name = parameter_name * "_init_seasonal_$i"
                push!(variable_names, var_name)
                vals[var_name] = NaN
                initial_vals[var_name] = NaN
            end
        end

        return new(parameter_name, variable_names, vals, initial_vals)
    end
end

mutable struct UCParameterDynamics
    constant::Bool
    level::String
    ar_I_order::Int
    ar_II_order::Int
    seasonal_length::Int
    hyperparameters::UCHyperparameters

    function UCParameterDynamics(parameter_name::String;
                                 level = "",
                                 ar_I_order = 0,
                                 ar_II_order = 0,
                                 seasonal_length = 0)

        constant = isempty(level) && iszero(ar_I_order) && iszero(ar_II_order) && iszero(seasonal_length)
        hyperparameters = UCHyperparameters(parameter_name,
                                            constant,
                                            level,
                                            ar_I_order,
                                            ar_II_order,
                                            seasonal_length)

        return new(constant, level, ar_I_order, ar_I_order, seasonal_length, hyperparameters)
    end
end

mutable struct UCParameterComponents
    parameter_name::String
    parameter_values::Vector{Float64}
    level::Vector{Float64}
    ar_I::Vector{Float64}
    ar_II::Vector{Float64}
    seasonal::Vector{Float64}

    function UCParameterComponents(n::Int, parameter_name::String)
        return new(
            parameter_name,
            zeros(n+1),
            zeros(n+1),
            zeros(n+1),
            zeros(n+1),
            zeros(n+1)
        )
    end
end

function uc_recursion(y::Vector{Float64}, 
                      parameter_dynamics::Vector{UCParameterDynamics})
    
    
end