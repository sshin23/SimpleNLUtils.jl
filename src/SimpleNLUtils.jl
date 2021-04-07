module SimpleNLUtils

import SimpleNL: Model, variable, Expression, Expression1, Expression2, Gradient, SparseJacobian, ExpressionSum, Variable, Constant, Parameter, Sink, Field, Field1, num_variables, num_constraints, Constraint, inner, index, fill_sparsity!, non_caching_eval, DUMMY
import SparseArrays: SparseMatrixCSC, sparse
import LinearAlgebra: mul!, norm

export get_terms, get_entries, sparsity, set_KKT_error_evaluator!

include("utils.jl")
include("structured.jl")
include("kkt.jl")

end # module
