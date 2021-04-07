function variable(m::Model,k;opt...)
    push!(m[:Vs][k],num_variables(m)+1)
    variable(m;opt...)
end

function Model(optimizer,K;opt...)
    m = Model(optimizer;opt...)
    m[:Vs]= Dict(k=>Int[] for k in K)
    return m
end
