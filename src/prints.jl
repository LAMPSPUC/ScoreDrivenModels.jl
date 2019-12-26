function Base.show(io::IO, est::EstimationStatsSDM)
    println("--------------------------------------------------------")
    println("log-likelihood: ", @sprintf("%.4f", est.loglikelihood))
    println("            np: ", Int(est.np))
    println("           AIC: ", @sprintf("%.4f", est.aic))
    println("           BIC: ", @sprintf("%.4f", est.bic))
    print_coefs_stats(est.coefs_stats)
    return nothing
end

function print_coefs_stats(coefs_stats)
    println("--------------------------------------------------------")
    println("Parameter      Estimate   Std.Error     t stat   p-value")
    offset = 1
    for i in coefs_stats.unknowns.ω
        p_c, p_std, p_t_stat, p_p_val = print_coefs_sta(coefs_stats, offset)
        p = build_print(p_c, p_std, p_t_stat, p_p_val, "ω_$i")
        println(p)
        offset += 1
    end
    for k in sort(collect(keys(coefs_stats.unknowns.A)))
        for i in coefs_stats.unknowns.A[k]
            p_c, p_std, p_t_stat, p_p_val = print_coefs_sta(coefs_stats, offset)
            ind = round(Int, sqrt(i))
            p = build_print(p_c, p_std, p_t_stat, p_p_val, "A_$(k)_$ind$ind")
            println(p)
            offset += 1
        end
    end
    for k in sort(collect(keys(coefs_stats.unknowns.B)))
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