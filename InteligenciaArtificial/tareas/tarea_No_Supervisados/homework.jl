using Pkg
Pkg.instantiate()
run(`pip install -r requirements.txt`)
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
embedding = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.4).fit_transform(original)
# store the data in a vector
#data = [expanded, original, embedding]
data = [Matrix{Float64}(D) for D in [expanded, original, embedding]]
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
    # grid param for mountain clustering
    gr = 10
    # ranges for the other clustering algorithms
    for i in eachindex(datasets)
        m, n = size(datasets[i])
        if n < 4
            rangesM = 0.1:0.1:5.0
            rangesS = 0.1:0.1:10.0
        else
            rangesM = 0.1:0.1:1.0
            rangesS = 0.1:0.1:1.0
        end
        #print 30x"_"
        println(repeat("_", 88))
        @info "DATASET $(i)"
        @info "size: $(m) x $(n)" 
        println(repeat("_", 88))
        if n > 4
            mnt_data = umap.UMAP(n_components=4, n_neighbors=20, min_dist=0.4).fit_transform(datasets[i])
        else
            mnt_data = datasets[i]
        end
        #change the typo of mnt_data to Matrix{Float64}
        mnt_data = Matrix{Float64}(mnt_data)
        # Get the best parameters for the other cluster Algorithms
        @info "Optmizing parameters for mountain clustering..."
        best_mtn = opt_mtn_args(mnt_data, gr=gr, ranges=rangesM) # Get the best parameters for mountain clustering
        @info "Optmizing parameters for subtracting clustering..."
        best_sub = opt_sub_args(datasets[i], ranges=rangesS) # Get the best parameters for subtracting clustering
        best_args = maximum([best_mtn[1], best_sub[1]]) # Compare the mean(silhouettes)
        if best_args == best_mtn[1]
            @info "Mountain clustering is the best clustering according to the silhouette index"
            centers, clusters = mountain_clustering(mnt_data, sigma=best_mtn[2][1], beta=best_mtn[2][2], gr=gr, metric=metric)
            @info "The best args are: $(best_mtn[2])"
        else
            @info "Subtracting clustering is the best clustering according to the silhouette index"
            centers, clusters = subtracting_clustering(datasets[i], ra=best_sub[2][1], rb=best_sub[2][2], metric=metric)
            @info "The best args are: $(best_sub[2])"
        end
        # Get the number of clusters
        n_clusters = length(centers)
        @info "Optimal number of clusters: $(n_clusters)"
        # Compute the kmeans, fuzzyCmeans and agnes clustering
        @info "Computing kmeans"
        kmeans_results = kmeans_clustering(datasets[i], k=n_clusters, metric=metric)
        @info "Computing fuzzyCmeans"
        fuzzyCmeans_results = fuzzyCmeans_clustering(datasets[i], k=n_clusters, e=2.0, metric=metric)
        @info "Computing agnes"
        agnes_results = agnes_clustering(datasets[i], k=n_clusters, metric=metric)
        # Validate using the silhouette index and the rand index
        # Silhouette index
        @info "Computing silhouette index"
        kmeans_silhouette = mean(silhouettes(kmeans_results[2], distances[i]))
        fuzzyCmeans_silhouette = mean(silhouettes(fuzzyCmeans_results[2], distances[i]))
        agnes_silhouette = mean(silhouettes(agnes_results[2], distances[i]))
        silhouette_results = [kmeans_silhouette, fuzzyCmeans_silhouette, agnes_silhouette]
        @info silhouette_results
        best_silhouette = maximum(silhouette_results)
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
        kmeans_rand = randindex(kmeans_results[2], best_clustering[2])[end]
        @info "Kmeans rand index: $(kmeans_rand)"
        fuzzyCmeans_rand = randindex(fuzzyCmeans_results[2], best_clustering[2])[end]
        @info "FuzzyCmeans rand index: $(fuzzyCmeans_rand)"
        agnes_rand = randindex(agnes_results[2], best_clustering[2])[end]
        @info "Agnes rand index: $(agnes_rand)"

        # Plot the loss functions of kmean and fuzzyCmeans
        p1 = plot(kmeans_results[3], title="Kmeans Loss", xlabel="Iterations", ylabel="Loss")
        p2 = plot(fuzzyCmeans_results[3], title="FuzzyCmeans Loss", xlabel="Iterations", ylabel="Loss")
        p = plot(p1, p2, layout=(1, 2), size=(1000, 500))
        # Save the plot
        savefig(p, "loss_$(i).png")
        # Plot the results if the data is 2D or 3D
        if size(datasets[i], 2) == 2 || size(datasets[i], 2) == 3
            p1 = plot_clusters(datasets[i], clusters=clusters, centers=centers, title="Exploratory Clustering")
            p2 = plot_clusters(datasets[i], clusters=kmeans_results[2], centers=kmeans_results[1], title="Kmeans Clustering")
            p3 = plot_clusters(datasets[i], clusters=fuzzyCmeans_results[2], centers=fuzzyCmeans_results[1], title="FuzzyCmeans Clustering")
            p4 = plot_clusters(datasets[i], clusters=agnes_results[2], centers=agnes_results[1], title="Agnes Clustering")
            p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 1000))
            # Save the plot
            savefig(p, "clustering_$(i).png")
        end
    end
end

function run()
    homework(data)
end