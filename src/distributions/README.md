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

For more details please see the section Implementing a new distribution in the documentation
<!-- TODO add link to the section -->