# Builds the `rustbert` binary (or one of its accelerated variants).
#
# No UI bundling, no shell completions — `mkPackage`'s defaults for
# `bundleUi` / `shellCompletions` resolve to `false` for any
# `cargoPackage` other than `"docbert"`. The same `cuda` / `metal`
# feature plumbing is wired through because rustbert depends on
# docbert-core, which is the heavy GPU-aware crate.
{
  mkPackage,
  name ? "rustbert",
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
