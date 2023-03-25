using MLDatasets, DataFrames
using Statistics, StatsBase
using Plots
gr()
using ColorSchemes
include("unsupervised.jl")

function test()
    # Load Iris dataset
    iris = Iris()
    # Normalize data in [0, 1]
    X = Matrix(iris.features[!, [:sepallength, :sepalwidth]])
    dt = fit(UnitRangeTransform, X, dims=1)
    X = StatsBase.transform(dt, X)
    centers, clusters = mountain_clustering(X, 0.2, 4.0, 10, euclidean_distance)
    @info "Number of clusters: $(length(centers))"
    @info "Cluster centers" centers
    # Generate a color palette that matches the number of clusters
    n_clusters = length(clusters)
    palette = distinguishable_colors(n_clusters)

    # Plot clusters with axes labels with sizes 20 and 30 respectively
    plot(title="Mountain Clustering", xlabel="Sepal Length", ylabel="Sepal Width", size=(600, 400))
    scatter!(X[:, 1], X[:, 2], group=clusters, color_palette=palette,legend=false)

    # Add cluster centers to the plot with the corresponding color from the palette
    for i in eachindex(centers)
        scatter!((centers[i][1], centers[i][2]),
            markersize=7,
            markerstrokewidth=2,
            color=palette[i])
    end
    # show plot
    display(plot!(size=(600, 400)))
end