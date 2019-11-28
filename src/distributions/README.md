# Distributions 

If you want to add a new distribution please feel free to make a pull Request.

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
