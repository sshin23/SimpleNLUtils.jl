get_terms(e::E) where E <: Expression = (terms=Expression[];_get_terms!(e,terms);terms)
_get_terms!(e::Expression2{typeof(+),E1,E2},terms) where {E1,E2} = (_get_terms!(e.e1,terms);_get_terms!(e.e2,terms))
function _get_terms!(e::Expression2{typeof(*),E1,E2},terms,mul=1) where {E1,E2 <: ExpressionSum}
    _get_terms!(e.e2,terms,mul*e.e1)
end
function _get_terms!(e::Expression2{typeof(*),E1,E2},terms,mul=1) where {E1 <: ExpressionSum, E2}
    _get_terms!(e.e2,terms,mul*e.e2)
end
function _get_terms!(e::Expression2{typeof(*),E1,E2},terms,mul=1) where {E1 <: ExpressionSum, E2 <: ExpressionSum}
    error("not supported")
end
function _get_terms!(e::ExpressionSum{E,I},terms,mul=1) where {E,I}
    _get_terms!(e.inner,terms)
    for e in e.es
        _get_terms!(e,terms,mul)
    end
end
function _get_terms!(e::ExpressionSum{E,Nothing},terms,mul=1) where E
    for e in e.es
        _get_terms!(e,terms,mul)
    end
end
_get_terms!(e::E,terms,mul=1) where E = push!(terms,mul ==1 ? e : e*mul)

get_entries(f::Sink{Field}) = get_entries(inner(f))
get_entries(f::Field1{E,I}) where {E,I} = (entries=[];_get_entries!(f,entries);entries)
function _get_entries!(f::Field1{E,I},entries) where {E,I}
    _get_entries!(inner(f),entries)
    for e in f.es
        _get_entries!(e,entries)
    end
end
function _get_entries!(f::Field1{E,Nothing},entries) where E
    for e in f.es
        _get_entries!(e,entries)
    end
end
_get_entries!(e::E,entries) where E = push!(entries,e)

get_entries_expr(f::Sink{Field}) = get_entries_expr(inner(f))
function get_entries_expr(f::Field1{E,I}) where {E,I}
    entries = get_entries(f)
    fs = Vector{Expression}(undef,length(entries))
    for c in entries
        fs[index(c)] = c.e
    end
    return fs
end

sparsity(e::E) where E = (indices=Int[];_sparsity!(e::E,indices);indices)
_sparsity!(e::Variable,indices) = @inbounds union!(indices,index(e))
_sparsity!(e::Parameter,indices) = nothing
_sparsity!(e::Constant,indices) = nothing
_sparsity!(e::Expression1{F,F1},indices) where {F,F1} = _sparsity!(e.e1,indices)
_sparsity!(e::Expression2{F,F1,F2},indices) where {F,F1,F2} = (_sparsity!(e.e1,indices);_sparsity!(e.e2,indices))
_sparsity!(e::Expression2{F,F1,F2},indices) where {F,F1<:Real,F2} = _sparsity!(e.e2,indices)
_sparsity!(e::Expression2{F,F1,F2},indices) where {F,F1,F2<:Real} = _sparsity!(e.e1,indices)
function _sparsity!(e::ExpressionSum{E,I},indices) where {E,I}
    _sparsity!(e.inner,indices)
    @simd for i in eachindex(e.es)
        @inbounds  _sparsity!(e.es[i],indices)
    end
end
function _sparsity!(e::ExpressionSum{E,Nothing},indices) where E
    @simd for i in eachindex(e.es)
        @inbounds _sparsity!(e.es[i],indices)
    end
end
