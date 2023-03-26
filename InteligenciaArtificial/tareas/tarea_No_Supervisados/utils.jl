using Plots

# filter cluster centers
function filter_centers(X, centers, labels, metric)
    # get only the unique cluster centers
    centers = unique(centers)
    # remove the centers with no data vectors assigned to them
    indexes = Vector{Int64}()
    for i = 1:length(centers)
        if !any(labels .== i)
            push!(indexes, i)
        end
    end
    centers = deleteat!(centers, indexes)
    # Remove centers that are too close to each other
    # (this is to avoid having too many centers)
    for i in eachindex(centers)
        for j = i+1:length(centers)
            if metric(centers[i], centers[j]) < 0.1
                centers[j] = centers[i]
            end
        end
    end
    centers = unique(centers)
    # Reassign cluster labels
    m = size(X, 1)
    for i = 1:m
        distances = zeros(length(centers))
        for j in eachindex(centers)
            distances[j] = metric(X[i, :], centers[j])
        end
        labels[i] = argmin(distances)
    end
    return centers, labels
end

# Plot clusters
function plot_clusters(X, clusters, centers=nothing, title="", xlabel="", ylabel="", zlabel="")
    # Generate a color palette that matches the number of clusters
    n_clusters = length(clusters)
    palette = distinguishable_colors(n_clusters)

    # Plot clusters with axes labels with sizes 20 and 30 respectively.
    plot(title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, size=(600, 400))
    # plot 2d if X has 2 features, otherwise plot 3d
    if size(X, 2) == 2
        scatter!(X[:, 1], X[:, 2], group=clusters, color_palette=palette,legend=false)
        # if centers are provided, add them to the plot
        if centers !== nothing
            # Add cluster centers to the plot with the corresponding color from the palette
            for i in eachindex(centers)
                scatter!((centers[i][1], centers[i][2]),
                    markersize=7,
                    markerstrokewidth=2,
                    color=palette[i])
            end
        end
    else
        scatter!(X[:, 1], X[:, 2], X[:, 3], group=clusters, color_palette=palette,legend=false)
        # if centers are provided, add them to the plot
    if centers !== nothing
        # Add cluster centers to the plot with the corresponding color from the palette
        for i in eachindex(centers)
            scatter!((centers[i][1], centers[i][2], centers[i][3]),
                markersize=7,
                markerstrokewidth=2,
                color=palette[i])
        end
    end
    end
    
    display(plot!(size=(600, 400)))
end