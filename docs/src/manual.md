# Manual

## Links

TODO rewrite this. The reparametrization changes the dynamics.
<!-- Links are reparametrizations utilized to assure certain parameter is within . For example, if we want to estimate a parameter ``f`` which is by definition strictly positive, then an obvious way to estimate ``f`` via numerical optimization is to model ``\tilde{f} = \ln{f}``. We refer to this procedure as **linking**. After obtaining the optimal value of ``\tilde{f}``, we can then **unlink** it to obtain ``f`` by computing ``f = e^{\tilde{f}}``. -->

### Types of links

The abstract type `Link` subsumes any type of link that can be expressed.

```@docs
ScoreDrivenModels.IdentityLink
ScoreDrivenModels.LogLink
```

### Link functions

```@docs
ScoreDrivenModels.link
ScoreDrivenModels.unlink
ScoreDrivenModels.jacobian_link
```

