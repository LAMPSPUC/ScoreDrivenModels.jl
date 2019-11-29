# Manual

## Links

Links are reparametrizations utilized in the estimation. For example, if we want to estimate a parameter ``f`` which is by definition strictly positive, then an obvious way to estimate ``f`` via numerical optimization is to model ``\tilde{f} = \ln{f}``. We refer to this procedure as **linking**. After obtaining the optimal value of ``\tilde{f}``, we can then **unlink** it to obtain ``f``.

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

