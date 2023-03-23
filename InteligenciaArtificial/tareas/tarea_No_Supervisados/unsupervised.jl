using Statistics

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
function euclidean_distance(x::Array{Float64,1}, y::Array{Float64,1}):Float64
    return sqrt(sum((x-y).^2))
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
function manhattan_distance(x::Array{Float64,1}, y::Array{Float64,1}):Float64
    return sum(abs.(x-y))
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
function cosine_distance(x::Array{Float64,1}, y::Array{Float64,1}):Float64
    return 1 - dot(x,y)/(norm(x)*norm(y))
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
function mahalanobis_distance(x::Array{Float64,1}, y::Array{Float64,1}):Float64
    return sqrt((x-y)'*inv(cov(x,y))*(x-y))
end

########################################################################################
#                            Clustering Algorithms
########################################################################################
"""
    mountain_clustering(X::Array{Float64,2}, k::Int64, metric::Function)::Array{Int64,1}
Computes the mountain clustering algorithm for a given dataset.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `k::Int64`: Number of clusters.
- `metric::Function`: Distance metric.

# Returns
- `Array{Int64,1}`: Cluster labels.
"""
function mountain_clustering(X::Array{Float64, 2}, k::Int64, metric::Function)::Array{Int64, 1}
    m,n = size(X) # m = number of samples, n = number of features
    #------------------------------------------------------------------
    # First: setup a grid matrix of n-dimensions (V)
    # (n = the dimension of input data vectors)
    # The gridding granularity is 'gr' = # of grid points per dimension
    #------------------------------------------------------------------
    gr = 10
    # Dimension vector
    v_dim = gr * ones(1, n)
    # Mountain matrix
    M = zeros(tuple(v_dim...))
    σ = 0.5

    n = 4
    cur = ones(1,n);
    for i = 1:n
        for j = 1:i
        cur[i] = cur[i]*v_dim[j];
        end
    end
    
end