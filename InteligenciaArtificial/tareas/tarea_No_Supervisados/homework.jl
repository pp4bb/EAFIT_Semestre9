using Pkg
Pkg.instantiate()
#run(`pip install -r requirements.txt`)
using MLDatasets, DataFrames, CSV
using Statistics, StatsBase
using ColorSchemes
using PyCall
umap=pyimport("umap")
include("unsupervised.jl")
include("metrics.jl")
include("utils.jl")

#------------------------------------------------------------------
# LOAD AND PREPARE THE DATA
#------------------------------------------------------------------
# load and normalize the original data
original = CSV.read("boston_housing.csv", DataFrame)
X = Matrix(original[:, 1:end])
X = Matrix(X)
dt = fit(UnitRangeTransform, X, dims=1)
original = StatsBase.transform(dt, X)
# load and normalize the expanded data
expanded = CSV.read("boston_housing_expanded.csv", DataFrame) # Already normalized
expanded = Matrix(expanded[:, 1:end])
# apply the UMAP algorithm to the original data to obtain the embedding
# n_components was set to 3 to be able to plot the data
embedding = umap.UMAP(n_components=3).fit_transform(original)
# store the data in a vector
data = [expanded, original, embedding]
#------------------------------------------------------------------
# CLUSTERING
#------------------------------------------------------------------
"""
    homework(datasets::Vector{Matrix{Float64}}; metric::Function=euclidean_distance)
Runs the homework for the given datasets.

# Arguments
- `datasets::Vector{Matrix{Float64}}`: Vector of datasets to cluster.
- `metric::Function=euclidean_distance`: Distance metric to use.
"""
function homework(datasets::Vector{Matrix{Float64}}; metric::Function=euclidean_distance)
    # Calculate the distances between the points
    distances = [pairwise(metric, eachrow(X)) for X in datasets]
    gr = 10
    for i in eachindex(datasets)
        @info size(datasets[i])
        # Get the best parameters for the other cluster Algorithms
        best_mtn = opt_mtn_args(datasets[i], gr=gr) # Get the best parameters for mountain clustering
        best_sub = opt_sub_args(datasets[i]) # Get the best parameters for subtracting clustering
        best_args = maximum([best_mtn[1], best_sub[1]]) # Compare the mean(silhouettes)
        if best_args == best_mtn[1]
            centers, clusters = mountain_clustering(datasets[i], sigma=best_mtn[2][1], beta=best_mtn[2][2], gr=gr, metric=metric)
        else
            centers, clusters = subtracting_clustering(datasets[i], ra=best_sub[2][1], rb=best_sub[2][2], metric=metric)
        end
        # Get the number of clusters
        n_clusters = length(centers)
        @info "Number of clusters: $(n_clusters)"
        # Compute the kmeans, fuzzyCmeans and agnes clustering
        kmeans_results = kmeans_clustering(datasets[i], k=n_clusters, metric=metric)
        fuzzyCmeans_results = fuzzyCmeans_clustering(datasets[i], k=n_clusters, e=2.0, metric=metric)
        agnes_results = agnes_clustering(datasets[i], k=n_clusters, metric=metric)
        # Validate using the silhouette index and the rand index
        # Silhouette index
        kmeans_silhouette = mean(silhouettes(kmeans_results[2], distances[i]))
        fuzzyCmeans_silhouette = mean(silhouettes(fuzzyCmeans_results[2], distances[i]))
        agnes_silhouette = mean(silhouettes(agnes_results[2], distances[i]))
        best_silhouette = maximum([kmeans_silhouette, fuzzyCmeans_silhouette, agnes_silhouette])
        # Get the best clustering
        if best_silhouette == kmeans_silhouette
            @info "Kmeans is the best clustering according to the silhouette index"
            best_clustering = kmeans_results
        elseif best_silhouette == fuzzyCmeans_silhouette
            @info "FuzzyCmeans is the best clustering according to the silhouette index"
            best_clustering = fuzzyCmeans_results
        else
            @info "Agnes is the best clustering according to the silhouette index"
            best_clustering = agnes_results
        end
        # Rand index with respect to the best clustering
        kmeans_rand = rand_index(kmeans_results[1], best_clustering[1])
        @info "Kmeans rand index: $(kmeans_rand)"
        fuzzyCmeans_rand = rand_index(fuzzyCmeans_results[1], best_clustering[1])
        @info "FuzzyCmeans rand index: $(fuzzyCmeans_rand)"
        agnes_rand = rand_index(agnes_results[1], best_clustering[1])
        @info "Agnes rand index: $(agnes_rand)"

        # Plot the results if the data is 2D or 3D
        if size(datasets[i], 2) == 2 || size(datasets[i], 2) == 3
            p1 = plot_clusters(X, clusters=clusters, centers=centers, title="Exploratory Clustering")
            p2 = plot_clusters(X, clusters=kmeans_results[2], centers=kmeans_results[1], title="Kmeans Clustering")
            p3 = plot_clusters(X, clusters=fuzzyCmeans_results[2], centers=fuzzyCmeans_results[1], title="FuzzyCmeans Clustering")
            p4 = plot_clusters(X, clusters=agnes_results[2], centers=agnes_results[1], title="Agnes Clustering")
            plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 1000))
        end
    end
end

function main()
    homework(data)
end