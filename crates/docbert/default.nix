# Builds the `docbert` binary (or one of its accelerated variants).
#
# Defaults to a plain CPU build; pass `name` plus the relevant
# `buildFeatures` / `buildInputs` / `extraEnv` / `extraPreBuild` from
# the call site to opt into `cuda` / `metal` / etc. `cargoPackage`
# defaults to `"docbert"` because `mkPackage` derives it from the
# part of `name` before any `-` (so `docbert-cuda` still maps to the
# right workspace member).
#
# `bundleUi` and `shellCompletions` default to `true` for this binary
# via `mkPackage`'s heuristic on `cargoPackage == "docbert"`, so the
# React UI gets bundled and the `completions` subcommand emits shell
# completion scripts at install time.
{
  mkPackage,
  name ? "docbert",
  buildFeatures ? [ ],
  buildInputs ? [ ],
  nativeBuildInputs ? [ ],
  extraEnv ? { },
  extraPreBuild ? "",
}:

mkPackage {
  inherit
    name
    buildFeatures
    buildInputs
    nativeBuildInputs
    extraEnv
    extraPreBuild
    ;
}
