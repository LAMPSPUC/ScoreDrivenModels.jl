function Base.show(io::IO, est::EstimationStats{D, T}) where {D, T}
    println(io, "--------------------------------------------------------")
    println(io, "Distribution:                 ", D)
    println(io, "Number of observations:       ", Int(est.num_obs))
    println(io, "Number of unknown parameters: ", Int(est.np))
    println(io, "Log-likelihood:               ", @sprintf("%.4f", est.loglikelihood))
    println(io, "AIC:                          ", @sprintf("%.4f", est.aic))
    println(io, "BIC:                          ", @sprintf("%.4f", est.bic))
    println(io, "Jarque Bera p value:          ", @sprintf("%.4f", est.jarquebera_p_value))
    print_coefs_stats(est.coefs_stats)
    return nothing
end

function print_coefs_stats(coefs_stats::CoefsStats{T}) where T
    println("--------------------------------------------------------")
    println("Parameter      Estimate   Std.Error     t stat   p-value")
    offset = 1
    for i in coefs_stats.unknowns.ω
        p_c, p_std, p_t_stat, p_p_val = print_coefs_sta(coefs_stats, offset)
        p = build_print(p_c, p_std, p_t_stat, p_p_val, "omega_$i")
        println(p)
        offset += 1
    end
    sorted_keys_A = sort(collect(keys(coefs_stats.unknowns.A)))
    for k in sorted_keys_A
        for i in coefs_stats.unknowns.A[k]
            p_c, p_std, p_t_stat, p_p_val = print_coefs_sta(coefs_stats, offset)
            ind = round(Int, sqrt(i))
            p = build_print(p_c, p_std, p_t_stat, p_p_val, "A_$(k)_$ind$ind")
            println(p)
            offset += 1
        end
    end
    sorted_keys_B = sort(collect(keys(coefs_stats.unknowns.B)))
    for k in sorted_keys_B
        for i in coefs_stats.unknowns.B[k]
            p_c, p_std, p_t_stat, p_p_val = print_coefs_sta(coefs_stats, offset)
            ind = round(Int, sqrt(i))
            p = build_print(p_c, p_std, p_t_stat, p_p_val, "B_$(k)_$ind$ind")
            println(p)
            offset += 1
        end
    end
    return nothing
end

function print_coefs_sta(sta, offset::Int)
    p_c      = @sprintf("%.4f", sta.coefs[offset])
    p_std    = @sprintf("%.4f", sta.std_errors[offset])
    p_t_stat = @sprintf("%.4f", sta.t_stat[offset])
    p_p_val  = @sprintf("%.4f", sta.p_values[offset])
    return p_c, p_std, p_t_stat, p_p_val
end

function build_print(p_c, p_std, p_t_stat, p_p_val, param)
    p = ""
    p *= param
    p *= " "^max(0, 23 - length(p_c) - length(param))
    p *= p_c
    p *= " "^max(0, 12 - length(p_std))
    p *= p_std
    p *= " "^max(0, 11 - length(p_t_stat))
    p *= p_t_stat
    p *= " "^max(0, 10 - length(p_p_val))
    p *= p_p_val
    return p
end
