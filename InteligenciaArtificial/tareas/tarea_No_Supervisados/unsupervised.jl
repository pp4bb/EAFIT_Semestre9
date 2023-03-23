using Statistics
using Logging
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
function manhattan_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64
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
function cosine_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64
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
function mahalanobis_distance(x::Array{Float64,1}, y::Array{Float64,1})::Float64
    return sqrt((x - y)' * inv(cov(x, y)) * (x - y))
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
function mountain_clustering(X::Array{Float64,2}, sigma::Float64, beta::Float64, gr::Int64, metric::Function)
    # Cluster centers
    C = Vector()
    # m = number of samples, n = number of features
    m, n = size(X)
    #------------------------------------------------------------------
    # First: setup a grid matrix of n-dimensions (V)
    # (n = the dimension of input data vectors)
    # The gridding granularity is 'gr' = # of grid points per dimension
    #------------------------------------------------------------------
    # create a grid matrix of n-dimensions
    V = zeros(gr^n, n)
    # create a vector of gr equally spaced points in the range [0,1]
    v = range(0, stop=1, length=gr)
    # create a matrix of all possible combinations of the points in v
    # (this is the grid matrix)
    for i = 1:n
        V[:, i] = repeat(v, inner=gr^(i - 1), outer=gr^(n - i))
    end
    #------------------------------------------------------------------
    # Second: compute the distance between each data vector and each
    # grid point, and calculate the mountain height at each grid point
    #------------------------------------------------------------------
    # compute the distance between each data vector and each grid point
    D = zeros(m, gr^n)
    for i = 1:m
        for j = 1:gr^n
            D[i, j] = metric(X[i, :], V[j, :])
        end
    end
    # calculate the mountain height at each grid point
    H = zeros(gr^n)
    # Mountain density function
    mountain(d, σ=sigma) = exp(-d^2 / (2 * sigma^2))

    for i = 1:1
        H[i] = sum(mountain.(D[:, i]))
    end


    while true
        #------------------------------------------------------------------
        # Third: find the  highest peak in the mountain
        #------------------------------------------------------------------
        # find the highest peak in the mountain and append it to C
        peak = argmax(H)
        @info H
        # display the values of H that are greater than 0.0
        # (this is to see the evolution of the mountain)
        display(H[H.>0.1])
        push!(C, V[peak, :])
        #------------------------------------------------------------------
        # Fourth: compute the value of the new mountain height at each grid
        # point. 
        # It requires to eliminate the influence of the previous 
        # cluster center.
        #------------------------------------------------------------------
        # update the mountain heights by eliminating the influence of the
        # previous cluster center
        for i = 1:gr^n
            H[i] = H[i] - H[peak] * mountain(metric(V[i, :], V[peak, :]), beta)
        end
        #------------------------------------------------------------------
        # Repeat until the cluster centers do not change or if the peak
        # is lower than a threshold.
        #------------------------------------------------------------------
        # check if the peak is lower than a threshold
        if length(C) > 1 && C[end-1] == C[end]
            break
        end
    end
    #------------------------------------------------------------------
    # Fifth: assign each data vector to the nearest cluster center
    #------------------------------------------------------------------
    labels = zeros(Int64, m)

    for i = 1:m
        labels[i] = argmin(metric.(X[i, :], C))
    end

    return C, labels


end