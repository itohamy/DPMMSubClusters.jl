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

    # Hyper params:
    alpha = 1.
    iters = 200

    # Define prior:
    D, N = size(data)
    m = zeros(Float32,(D))
    k = 1.0
    nu = 130.  # should be > D
    psi = cov(data')  # !!check the shapes!!

    hyper_prior = DPMMSubClusters.niw_hyperparams(k, m, nu, psi)

    #Run the model:
    labels, clusters, weights = DPMMSubClusters.fit(data, hyper_prior, alpha, labels_gt, iters=iters, init_clusters=5, verbose=true)

    println("\n------ Finished main ------\n")

end



main()