using Documenter, GAS

makedocs(
    modules = [GAS],
    doctest  = false,
    clean    = true,
    format   = Documenter.HTML(mathengine = Documenter.MathJax()),
    sitename = "GAS.jl",
    authors  = "Guilherme Bodin and Raphael Saavedra",
    pages   = [
        "Home" => "index.md",
        "manual.md",
        "examples.md"
    ]
)

deploydocs(
    repo = "github.com/LAMPSPUC/GAS.jl.git",
)