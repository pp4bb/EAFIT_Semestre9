using Statistics, LinearAlgebra
using Clustering
using Logging
include("utils.jl")

########################################################################################
#                            Clustering Algorithms
########################################################################################
"""
    mountain_clustering(X::Array{Float64,2}, k::Int64, metric::Function)
computes the mountain clustering algorithm for a given dataset.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `k::Int64`: Number of clusters.
- `metric::Function`: Distance metric.

# Returns
- `Array{Array{Float64,1},1}`: cluster centers.
- `Array{Int64,1}`: cluster labels.
"""
function mountain_clustering(
    X::Array{Float64,2}; 
    sigma::Float64=0.5, 
    beta::Float64=0.5, 
    gr::Int64=10, 
    metric::Function=euclidean_distance
    )
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
    opt_mtn_args(X::Array{Float64,2}, gr::Int64, metric::Function)
computes the optimal parameters for the mountain clustering algorithm.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `gr::Int64`: Grid granularity.
- `metric::Function`: Distance metric.
- `ranges::StepRangeLen`: Range of sigma and beta.

# Returns
- `Tuple{Float64,Float64}`: Optimal sigma and beta.
"""
function opt_mtn_args(X::Array{Float64,2}; 
    gr::Int64=10, 
    metric::Function=euclidean_distance, 
    ranges::StepRangeLen=0.1:0.1:1.0
    )
    # Distances between data points
    distances = pairwise(metric, eachrow(X))
    sigmas = ranges
    betas = ranges
    # all combinations of sigma and beta
    args = [(s, b) for s in sigmas, b in betas]
    max_s = 0
    max_args = (0, 0)
    for i in eachindex(args)
        centers, clusters = mountain_clustering(X, sigma=args[i][1], beta=args[i][2], gr=gr, metric=metric)
        # if all data points are assigned to the same cluster continue
        if length(unique(clusters)) == 1
            continue
        end
        # compute silhouette score
        s = mean(silhouettes(clusters, distances))
        if s > max_s
            max_s = s
            max_args = args[i]
        end
    end
    return max_s, max_args
end

"""
    subtracting_clustering(X::Array{Float64,2}, r::Float64, metric::Function)::Array{Int64,1}
computes the subtracting clustering algorithm for a given dataset.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `r::Float64`: Radius.
- `metric::Function`: Distance metric.

# Returns
- `Array{Array{Float64,1},1}`: cluster centers.
- `Array{Int64,1}`: cluster labels.
"""
function subtracting_clustering(X::Array{Float64,2};
    ra::Float64, 
    rb::Float64=0.5, 
    metric::Function=euclidean_distance
    )
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
    opt_sub_args(X::Array{Float64,2}, r::Float64, metric::Function)::Array{Int64,1}
computes the optimal parameters for the subtracting clustering algorithm.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `ranges::StepRangeLen`: Range of values for sigma and beta.
"""
function opt_sub_args(X; ranges::StepRangeLen=0.1:0.1:1.0)
    # Distances between data points
    distances = pairwise(euclidean_distance, eachrow(X))
    ra = ranges
    # all combinations of sigma and beta
    args = [(i, 1.5*i) for i in ra]
    max_s = 0
    max_args = (0, 0)
    for i in eachindex(args)
        centers, clusters = subtracting_clustering(X, ra=args[i][1], rb=args[i][2])
        # if all data points are assigned to the same cluster continue
        if length(unique(clusters)) == 1
            continue
        end
        # compute silhouette score
        s = mean(silhouettes(clusters, distances))
        if s > max_s
            max_s = s
            max_args = args[i]
        end
    end
    return max_s, max_args
end

"""
    kmeans_clustering(X::Array{Float64,2}, k::Int64, metric::Function)::Array{Int64,1}
computes the k-means clustering algorithm for a given dataset.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `k::Int64`: Number of centers.
- `metric::Function`: Distance metric.

# Returns
- `Array{Array{Float64,1},1}`: cluster centers.
- `Array{Int64,1}`: cluster labels.
- `Array{Float64,1}`: loss function.
"""
function kmeans_clustering(
    X::Array{Float64,2}; 
    k::Int64,
    metric::Function=euclidean_distance
    )
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
    return centers, labels, losses
end

"""
    fuzzyCmeans_clustering(X::Array{Float64,2}, k::Int64, e::Float64, metric::Function)
computes the fuzzy c-means clustering algorithm for a given dataset.

# Arguments
- `X::Array{Float64,2}`: Dataset.
- `k::Int64`: Number of centers.
- `e::Float64`: Fuzziness parameter.
- `metric::Function`: Distance metric.

# Returns
- `Array{Array{Float64,1},1}`: cluster centers.
- `Array{Int64,1}`: cluster labels.
- `Array{Float64,1}`: loss function.
"""
function fuzzyCmeans_clustering(
    X::Array{Float64,2};
    k::Int64, 
    e::Float64=2.0, 
    metric::Function=euclidean_distance
    )
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
            for j = 1:k
                U[i, j] = 1 / sum([(distances[i, j] / distances[i, l])^(2/(e-1)) for l = 1:k])
                if isnan(U[i, j])
                    U[i, j] = 1
                end
            end
        end

        #------------------------------------------------------------------
        # Third: update the cluster centers
        #------------------------------------------------------------------
        for i = 1:k
            numerator = sum([U[j, i]^e * X[j, :] for j = 1:m], dims=1)
            denominator = sum([U[j, i]^e for j = 1:m])
            coordinates = (numerator / denominator)[1]
            centers[i] = coordinates
        end
        #------------------------------------------------------------------
        # Fourth: compute the loss function
        #------------------------------------------------------------------
        # compute pairwise distances between each point and each center
        D = pairwise(metric, eachrow(X), centers)
        Ue = U .^ e # raise membership matrix U to the e-th power       
        UD = Ue .* D # compute the element-wise product between Ue and D       
        loss = sum(UD, dims=2) # compute the sum of the rows of UD
        loss = sum(loss) # compute the total loss
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
    return centers, labels, losses
end

"""
    agnes_clustering(X::Array{Float64, 2}, k::Int64, metric::Function)
computes the agglomerative hierarchical clustering algorithm for a given dataset.

# Arguments
- `X::Array{Float64, 2}`: Dataset.
- `k::Int64`: Number of clusters.
- `metric::Function`: Distance metric.

# Returns
- `Array{Array{Float64, 1}, 1}`: cluster centers.
- `Array{Int64, 1}`: cluster labels.
"""
function agnes_clustering(
    X::Array{Float64, 2}; 
    k::Int64, 
    metric::Function=euclidean_distance
    )
    # m = number of samples, n = number of features
    m, n = size(X)
    #------------------------------------------------------------------
    # First: initialice clustering with all data points as singletons
    # and compute the distances between each pair of data points
    #------------------------------------------------------------------
    centers = [X[i,1:n] for i in 1:m]
    distances = pairwise(metric, eachrow(X), centers)
    # set diagonal to Inf to avoid self-merging
    distances[diagind(distances)] .= Inf
    # iterate until we have k clusters
    while length(centers) > k
        #------------------------------------------------------------------
        # Second: find the closest pair of clusters
        #------------------------------------------------------------------
        minidx = argmin(distances)
        #------------------------------------------------------------------
        # Third: merge the closest pair of clusters
        #------------------------------------------------------------------
        # compute the new center
        newcenter = (centers[minidx[1]] + centers[minidx[2]]) / 2
        # remove the old centers
        deleteat!(centers, max(minidx[1], minidx[2]))
        deleteat!(centers, min(minidx[1], minidx[2]))
        # add the new center
        push!(centers, newcenter)
        #------------------------------------------------------------------
        # Fourth: update the distances between each pair of clusters
        #------------------------------------------------------------------
        cmat = transpose(hcat(centers...))
        cmat[diagind(cmat)] .= Inf
        distances = pairwise(metric, eachrow(cmat), cmat)
    end
    #------------------------------------------------------------------
    # Fifth: assign each data vector to the nearest cluster center
    #------------------------------------------------------------------
    labels = zeros(Int64, m)
    for i = 1:m
        labels[i] = argmin([metric(X[i, :], centers[j]) for j in eachindex(centers)])
    end
    # filter the centers
    centers, labels = filter_centers(X, centers, labels, metric)
    return centers, labels
end
