########################################################################################
#                                     Metrics
########################################################################################
"""
    euclidean_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64
Compute the euclidean distance between two vectors.

# Arguments
- `x::Array{Float64,1}`: First vector.
- `y::Array{Float64,1}`: Second vector.

# Returns
- `Float64`: Euclidean distance between `x` and `y`.
"""
function euclidean_distance(x, y)
    return sqrt(sum((x .- y) .^ 2))
end

"""
    manhattan_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64
Compute the manhattan distance between two vectors.

# Arguments
- `x::Array{Float64,1}`: First vector.
- `y::Array{Float64,1}`: Second vector.

# Returns
- `Float64`: Manhattan distance between `x` and `y`.
"""
function manhattan_distance(x, y)
    return sum(abs.(x - y))
end

"""
    cosine_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64
Compute the cosine distance between two vectors.

# Arguments
- `x::Array{Float64,1}`: First vector.
- `y::Array{Float64,1}`: Second vector.

# Returns
- `Float64`: Cosine distance between `x` and `y`.
"""
function cosine_distance(x, y)
    return 1 - dot(x, y) / (norm(x) * norm(y))
end

"""
    mahalanobis_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64
Compute the mahalanobis distance between two vectors.

# Arguments
- `x::Array{Float64,1}`: First vector.
- `y::Array{Float64,1}`: Second vector.

# Returns
- `Float64`: Mahalanobis distance between `x` and `y`.
"""
function mahalanobis_distance(x, y)
    return sqrt((x - y)' * inv(cov(x, y)) * (x - y))
end