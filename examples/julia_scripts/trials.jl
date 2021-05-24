using Images
using Distributed
using Revise

#addprocs(4)
@everywhere using VersatileHDPMixtureModels
@everywhere using DPMMSubClusters
@everywhere using Random
@everywhere using Statistics


# Run the DPMM code to test the changes:
function main()

    println("\n------ Start main ------\n")
    
    Random.seed!(12345)

    # Get gaussian data:
    N = 10^4  # number of points
    D = 2   # data dimension
    modes = 6   # number of modes
    var_scale = 80.0
    data, labels_gt, clusters_gt = DPMMSubClusters.generate_gaussian_data(N, D, modes, var_scale)

    # Changing the lables to be incorrect (to see how the splits work)
    #labels_gt[labels_gt.==3] .= 2
    #labels_gt[labels_gt.==4] .= 3
    #labels_gt[labels_gt.==5] .= 4
    #labels_gt[labels_gt.==6] .= 5

    # Shuffle data and gt_labels:
    data, labels_gt = shuffle_data_points_and_labels(data, labels_gt)
    
    # Hyper params:
    alpha = 1.
    iters = 200
    init_clusters = length(unique(labels_gt))

    # Define prior:
    D, N = size(data)
    m = zeros(Float32,(D))
    k = init_clusters  #1.0
    nu = 130.  # should be > D
    psi = cov(data')*0.01  # shape (D,D)

    hyper_prior = DPMMSubClusters.niw_hyperparams(k, m, nu, psi)

    # Original label counts:
    label_counts = zeros(Int(init_clusters))
    for i=1:length(labels_gt)
        l = Int(labels_gt[i])
        label_counts[l] = label_counts[l] + 1
    end
    for i=1:length(label_counts)
        println("label " * string(i) * ": " * string(label_counts[i]))
    end

    #Run the model:
    labels, clusters, weights = DPMMSubClusters.fit(data, hyper_prior, alpha, labels_gt, iters=iters, init_clusters=init_clusters, verbose=true)

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