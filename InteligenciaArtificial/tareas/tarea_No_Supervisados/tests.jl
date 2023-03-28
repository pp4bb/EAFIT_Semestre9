using MLDatasets, DataFrames
using Statistics, StatsBase
using ColorSchemes
using PyCall
umap=pyimport("umap")

include("unsupervised.jl")
include("metrics.jl")
include("utils.jl")

# Load Iris dataset
iris = Iris()
# Normalize data in [0, 1]
X = Matrix(iris.features[!, [:petalwidth, :petallength, :sepalwidth]])
dt = fit(UnitRangeTransform, X, dims=1)
X = StatsBase.transform(dt, X)

function mountain()
    centers, clusters = mountain_clustering(X, sigma=0.1, beta=0.5, gr=20, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    plot_clusters(X, clusters=clusters, centers=centers, title="Mountain Clustering", xlabel="Petal Width", ylabel="Petal Length", zlabel="Sepal Width")
end

function subtracting()
    centers, clusters = subtracting_clustering(X, ra=1.0, rb=1.5, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    plot_clusters(X, clusters=clusters, centers=centers, title="Subtracting Clustering", xlabel="Petal Width", ylabel="Petal Length", zlabel="Sepal Width")
end

function kmeans()
    centers, clusters, losses = kmeans_clustering(X, k=3, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    p1 = plot_clusters(X, clusters=clusters, centers=centers, title="Kmeans Clustering", xlabel="Petal Width", ylabel="Petal Length", zlabel="Sepal Width")
    p2 = plot(losses, title="K-Means Clustering", xlabel="Iterations", ylabel="Loss")
    display(p1)
    display(p2)
end

function fuzzyCmeans()
    centers, clusters, losses = fuzzyCmeans_clustering(X, k=3, e=2.0, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    p1 = plot_clusters(X, clusters=clusters, centers=centers, title="FuzzyCmeans Clustering", xlabel="Petal Width", ylabel="Petal Length", zlabel="Sepal Width")
    p2 = plot(losses, title="Fuzzy C-Means Clustering", xlabel="Iterations", ylabel="Loss")
    display(p1)
    display(p2)
end

function agnes()
    centers, clusters = agnes_clustering(X, k=3, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    plot_clusters(X, clusters=clusters, centers=centers, title="Agnes Clustering", xlabel="Petal Width", ylabel="Petal Length", zlabel="Sepal Width")
end

function umap_emb()
    embedding = umap.UMAP().fit_transform(X)
end

function indexes()
    opt_sub_args(X)
end