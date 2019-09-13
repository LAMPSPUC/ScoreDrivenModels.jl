function fill_ω!(sd_model::SDModel, ω_tilde::Vector{T}) where T
    for i in eachindex(ω_tilde)
        sd_model.ω = ω_tilde[i]
    end
    return 
end

function fill_A!(sd_model::SDModel, A_tilde::Vector{T}) where T
    for i in eachindex(A_tilde)
        sd_model.A[i, i] = A_tilde[i]
    end
    return 
end

function fill_B!(sd_model::SDModel, B_tilde::Vector{T}) where T
    for i in eachindex(B_tilde)
        sd_model.B[i, i] = B_tilde[i]
    end
    return 
end

