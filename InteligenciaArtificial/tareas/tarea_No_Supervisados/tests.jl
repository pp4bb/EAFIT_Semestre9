using MLDatasets, DataFrames
using Statistics, StatsBase
using ColorSchemes
include("unsupervised.jl")
include("metrics.jl")
include("utils.jl")

# Load Iris dataset
iris = Iris()
# Normalize data in [0, 1]
X = Matrix(iris.features[!, [:petalwidth, :petallength]])
dt = fit(UnitRangeTransform, X, dims=1)
X = StatsBase.transform(dt, X)

function mountain()
    centers, clusters = mountain_clustering(X, sigma=0.5, beta=0.5, gr=20, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    plot_clusters(X, clusters, centers, "Mountain Clustering", "Petal Width", "Petal Length", "Sepal Width")
end

function subtracting()
    centers, clusters = subtracting_clustering(X, ra=0.5, rb=1.5*0.5, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    plot_clusters(X, clusters, centers, "Subtractin Clustering", "Petal Width", "Petal Length", "Sepal Width")
end

function kmeans()
    centers, clusters, losses = kmeans_clustering(X, k=3, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    p1 = plot_clusters(X, clusters, centers, "K-Means Clustering", "Petal Width", "Petal Length", "Sepal Width")
    p2 = plot(losses, title="K-Means Clustering", xlabel="Iterations", ylabel="Loss")
    display(p1)
    display(p2)
end

function fuzzyCmeans()
    centers, clusters, losses = fuzzyCmeans_clustering(X, k=3, e=2.0, metric=euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    p1 = plot_clusters(X, clusters, centers, "Fuzzy C-Means Clustering", "Petal Width", "Petal Length", "Sepal Width")
    p2 = plot(losses, title="Fuzzy C-Means Clustering", xlabel="Iterations", ylabel="Loss")
    display(p1)
    display(p2)
end