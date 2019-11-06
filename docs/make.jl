using Documenter, ScoreDrivenModels

makedocs(
    modules = [ScoreDrivenModels],
    doctest  = false,
    clean    = true,
    format   = Documenter.HTML(),
    sitename = "ScoreDrivenModels.jl",
    authors  = "Guilherme Bodin and Raphael Saavedra",
    pages   = [
        "Home" => "index.md",
        "manual.md",
        "examples.md"
    ]
)

deploydocs(
    repo = "github.com/LAMPSPUC/ScoreDrivenModels.jl.git",
)