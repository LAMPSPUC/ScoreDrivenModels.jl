# Distributions 

If you want to add a new distribution please feel free to make a pull request.

Each distribution must have the following methods:
* score
* fisher information
* log likelihood
* link interface
    * link
    * unlink
    * jacobian_link
* update_dist
* num_params

For more details, please read section [Implementing a new distribution](https://lampspuc.github.io/ScoreDrivenModels.jl/latest/manual/#Implementing-a-new-distribution-1) 
in the documentation
