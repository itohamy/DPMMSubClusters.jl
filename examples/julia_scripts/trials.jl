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
    data, labels_gt, clusters_gt = DPMMSubClusters.generate_gaussian_data(10^4, 2, 6, 80.0)

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
    labels, clusters, weights = DPMMSubClusters.fit(data, hyper_prior, alpha, iters=iters, verbose=true)

    println("\n------ Finished main ------\n")

end



main()