# Distributions 

If you want to add a new distribution please feel free to make a pull Request.

Each distribution must have the following methods:
* score
* fisher information
* log likelihood
* link interface
    * param_to_param_tilde
    * param_tilde_to_param
    * jacobian_param_tilde
* update_dist
* num_params
