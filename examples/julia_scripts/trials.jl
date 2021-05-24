using Images
using Distributed
using Revise
using LinearAlgebra

#addprocs(4)
@everywhere using DPMMSubClusters
@everywhere using Random
@everywhere using Statistics


# Run the DPMM code to test the changes:
function main()

    println("\n------ Start main ------\n")
    
    Random.seed!(12345)

    # --- Toy data #1:
    # N = 20000 #Number of points
    # D = 2 # Dimension
    # modes = 20 # Number of Clusters
    # var_scale = 100.0 # The variance of the MV-Normal distribution where the clusters means are sampled from.

    # --- Toy data #2:
    N = 10^4  # number of points
    D = 2   # data dimension
    modes = 6   # number of modes
    var_scale = 80.0
    
    # --- Extract the data:
    data, labels_gt, clusters_gt = DPMMSubClusters.generate_gaussian_data(N, D, modes, var_scale)

    # --- Changing the lables to be incorrect (to see how the splits work)
    labels_gt[labels_gt.==3] .= 2
    labels_gt[labels_gt.==4] .= 30
    # labels_gt[labels_gt.==5] .= 4
    # labels_gt[labels_gt.==6] .= 5
    labels_gt[labels_gt.==6] .= 18
    #labels_gt[labels_gt.==19] .= 180

    # --- Shuffle data and gt_labels:
    data, labels_gt = shuffle_data_points_and_labels(data, labels_gt)
    init_clusters = length(unique(labels_gt))
    
    # --- hyper params #1:
    # hyper_prior = DPMMSubClusters.niw_hyperparams(1, zeros(Float32,(D)), 5, Matrix{Float64}(I, D, D)*0.5)
    # alpha = 10.
    # iters = 200

    # --- Hyper params #2:
    D, N = size(data)
    m = zeros(Float32,(D))
    k = init_clusters  #1.0
    nu = 130.  # should be > D
    psi = cov(data')*0.01  # shape (D,D)
    hyper_prior = DPMMSubClusters.niw_hyperparams(k, m, nu, psi)
    alpha = 1.
    iters = 200

    # Original label counts:
    unique_gt_labels = sort(unique(labels_gt))
    label_counts = zeros(Int(init_clusters))
    for i=1:length(labels_gt)
        l = Int(labels_gt[i])
        ind_unmapped = findall(x->x==labels_gt[i], unique_gt_labels)[1]
        label_counts[ind_unmapped] = label_counts[ind_unmapped] + 1
    end
    for i=1:length(label_counts)
        println("label " * string(unique_gt_labels[i]) * ": " * string(label_counts[i]))
    end

    #Run the model:
    # "outlier_params=labels_gt": this is a workaround to pass additional varibale to python wrapper
    labels, clusters, weights = DPMMSubClusters.fit(data, hyper_prior, alpha, iters=iters, verbose=true, outlier_params=labels_gt)

    println("\n------ Finished main ------\n")

end




# Shuffle the N data points and their labels in the same way. x shape should be (D,N).
function shuffle_data_points_and_labels(x, lbl)
    D, N = size(x)
    inds = shuffle(1:N)

    x_new = zeros(Float32,(D, N))  # DxN
    lbl_new = zeros(Float32,(N,))  # Nx1
    for i=1:N
        x_new[:, i:i] = x[:, inds[i]:inds[i]]
        lbl_new[i] = lbl[inds[i]]
    end

    return x_new, lbl_new
end

main()