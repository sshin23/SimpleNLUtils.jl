mutable struct KKTErrorEvaluator{G,C,J}
    f::Vector{Float64}
    c::Vector{Float64}
    
    jac::SparseMatrixCSC{Float64,Int}
    jac_view::SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}
    
    grad!::G
    con!::C
    jac!::J
end

function KKTErrorEvaluator(m::Model)
    
    obj = m.obj
    con = m.con
    p = m.p
    
    grad = Gradient(obj)
    jacobian,jac_sparsity = SparseJacobian(con)
    
    _obj = @inline x->non_caching_eval(obj,x,p)
    _con! = @inline (y,x)->non_caching_eval(con,y,x,p)
    _grad! = @inline function (y,x)
        y .= 0
        obj(x,p)
        non_caching_eval(grad,y,x,p)
    end
    _jac! = @inline function (J,x)
        J .= 0
        con(DUMMY,x,p)
        non_caching_eval(jacobian,J,x,p)
    end
    _jac_fill_sparsity! = @inline (I,J)->fill_sparsity!(I,J,jac_sparsity)
    
    nnz_jac = length(jac_sparsity)

    I = Vector{Int}(undef,nnz_jac)
    J = Vector{Int}(undef,nnz_jac)
    _jac_fill_sparsity!(I,J)

    f = similar(m.x)
    c = similar(m.l)
    
    tmp = sparse(J,I,collect(1:nnz_jac),num_variables(m),num_constraints(m))
    jac = SparseMatrixCSC{Float64,Int64}(tmp.m,tmp.n,tmp.colptr,tmp.rowval,Vector{Float64}(undef,nnz_jac))

    idx = Vector{Int}(undef,length(tmp.nzval))
    for i=eachindex(tmp.nzval)
        idx[tmp.nzval[i]] = i
    end
    jac_view = view(jac.nzval,idx)
    
    return KKTErrorEvaluator(f,c,jac,jac_view,_grad!,_con!,_jac!)
end

function (evaluator::KKTErrorEvaluator{G,C,J})(x,l,gl) where {G,C,J}
    
    evaluator.grad!(evaluator.f,x)
    evaluator.con!(evaluator.c,x)
    evaluator.jac!(evaluator.jac_view,x)

    mul!(evaluator.f,evaluator.jac,l,1.,1.)
    evaluator.c .-= gl

    return max(norm(evaluator.f,Inf),norm(evaluator.c,Inf))
end
