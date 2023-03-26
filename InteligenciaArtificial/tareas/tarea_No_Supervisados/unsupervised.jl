using Statistics
using Logging
include("utils.jl")

########################################################################################
#                            centerslustering Algorithms
########################################################################################
"""
    mountain_clustering(X::Array{Float64,2}, k::Int64, metric::Function)
centersomputes the mountain clustering algorithm for a given dataset.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `k::Int64`: Number of clusters.
- `metric::Function`: Distance metric.

# Returns
- `Array{Array{Float64,1},1}`: centersluster centers.
- `Array{Int64,1}`: centersluster labels.
"""
function mountain_clustering(X::Array{Float64,2}; sigma::Float64=0.5, beta::Float64=0.5, gr::Int64=10, metric::Function=euclidean_distance)
    # cluster centers
    centers = Vector()
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
    M = zeros(gr^n)
    # Mountain density function
    density(d, σ) = exp(-(d^2 / (2 * σ^2)))

    for i = 1:gr^n
        M[i] = sum(density.(D[:, i], sigma))
    end


    while true
        #------------------------------------------------------------------
        # Third: find the  highest peak in the mountain
        #------------------------------------------------------------------
        # find the highest peak in the mountain and append it to centers
        peak = argmax(M)
        # (this is to see the evolution of the mountain)
        push!(centers, V[peak, :])
        highest = M[peak]
        #------------------------------------------------------------------
        # Fourth: compute the value of the new mountain height at each grid
        # point. 
        # It requires to eliminate the influence of the previous 
        # cluster center.
        #------------------------------------------------------------------
        # update the mountain heights by eliminating the influence of the
        # previous cluster center
        for i = 1:gr^n
            M[i] = M[i] - highest * density(metric(V[i, :], V[peak, :]), beta)
        end
        # break criterion
        if length(centers) > 1 && centers[end-1] == centers[end] || length(centers) > m
            break
        end
    end
    #------------------------------------------------------------------
    # Fifth: assign each data vector to the nearest cluster center
    #------------------------------------------------------------------
    labels = zeros(Int64, m)

    for i = 1:m
        distances = [metric(X[i, :], c) for c in centers]
        minidx = argmin(distances)
        labels[i] = minidx
    end
    # filter centers
    centers, labels = filter_centers(X, centers, labels, metric)
    return centers, labels
end

"""
    subtracting_clustering(X::Array{Float64,2}, r::Float64, metric::Function)::Array{Int64,1}
centersomputes the subtracting clustering algorithm for a given dataset.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `r::Float64`: Radius.
- `metric::Function`: Distance metric.

# Returns
- `Array{Array{Float64,1},1}`: centersluster centers.
- `Array{Int64,1}`: centersluster labels.
"""
function subtracting_clustering(X::Array{Float64,2}; ra::Float64, rb::Float64=0.5, metric::Function=euclidean_distance)
    # cluster centers
    centers = Vector()
    # m = number of samples, n = number of features
    m, n = size(X)
    # Density function
    density(d, r) = exp(-(d^2 / (r/2)^2))
    #------------------------------------------------------------------
    # First: compute the density of each data vector
    #------------------------------------------------------------------
    # precompute pairwise distances between data points
    distances = zeros(m, m)
    for i = 1:m
        for j = 1:m
            distances[i, j] = metric(X[i, :], X[j, :])
        end
    end
    D = zeros(m)
    # compute the density of each data vector
    for i = 1:m
        D[i] = sum(density.(distances[i, :], ra))
    end
    while true
        #------------------------------------------------------------------
        # Second: find the data vector with the highest density and make it
        # a cluster center
        #------------------------------------------------------------------
        # find the data vector with the highest density and append it to centers
        peak = argmax(D)
        push!(centers, X[peak, :])
        densest = D[peak]
        #------------------------------------------------------------------
        # Third: compute the new density of each data vector. It requires to
        # eliminate the influence of the previous cluster center.
        #------------------------------------------------------------------
        # update the density by eliminating the influence of the previous
        # cluster center
        for i = 1:m
            D[i] = D[i] - densest * density(metric(X[i, :], X[peak, :]), rb)
        end
        # break criterion
        if length(centers) > 1 && centers[end-1] == centers[end] || length(centers) > m
            break
        end
    end

    #------------------------------------------------------------------
    # Fourth: assign each data vector to the nearest cluster center
    #------------------------------------------------------------------
    labels = zeros(Int64, m)
    
    for i = 1:m
        distances = [metric(X[i, :], c) for c in centers]
        minidx = argmin(distances)
        labels[i] = minidx
    end
    # filter centers
    centers, labels = filter_centers(X, centers, labels, metric)
    return centers, labels
end

"""

"""
function kmeans_clustering(X::Array{Float64,2}; k::Int64, metric::Function=euclidean_distance)
    # m = number of samples, n = number of features
    m, n = size(X)
    # membership matrix
    U = zeros(m, k)
    #------------------------------------------------------------------
    # First: randomly select k data vectors as cluster centers
    #------------------------------------------------------------------
    centers = [X[i,1:n] for i in rand(1:size(X, 1), k)]
    # Vector to track the loss function at each iteration
    losses = Vector()
    while true
        # precompute the distances between each data vector and each cluster center
        distances = zeros(m, k)
        for i = 1:m
            for j = 1:k
                distances[i, j] = metric(X[i, :], centers[j])
            end
        end
        #------------------------------------------------------------------
        # Second: Determine the membership matrix U
        #------------------------------------------------------------------
        U = zeros(m, k)
        for i = 1:m
            minidx = argmin(distances[i, :])
            U[i, minidx] = 1
        end
        #------------------------------------------------------------------
        # Third: update the cluster centers
        #------------------------------------------------------------------
        for i = 1:k
            coordinates = mean(X[U[:, i] .== 1, :], dims=1)
            centers[i] = [coordinates[1, j] for j in 1:n]
        end
        #------------------------------------------------------------------
        # Fourth: compute the loss function
        #------------------------------------------------------------------
        loss = sum([sum(distances[i, U[i, :] .== 1].^2) for i = 1:m])
        push!(losses, loss)
        # break criterion
        if length(losses) > 1 && losses[end-1] == losses[end]
            break
        end
    end
    #------------------------------------------------------------------
    # Fifth: assign each data vector to the nearest cluster center using
    # the membership matrix
    #------------------------------------------------------------------
    labels = zeros(Int64, m)
    for i = 1:m
        labels[i] = argmax(U[i, :])
    end
    centers, labels = filter_centers(X, centers, labels, metric)
    return centers, labels

end

