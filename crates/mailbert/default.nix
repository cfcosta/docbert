# Builds the `mailbert` binary (or one of its accelerated variants).
#
# No UI bundling — the React tree belongs to docbert alone — but
# `shellCompletions` is on, because `mailbert completions <shell>`
# exists the same way docbert's does. `mkPackage`'s heuristic only
# turns it on for `cargoPackage == "docbert"`, so it's opted into
# explicitly here.
#
# The `cuda` / `metal` feature plumbing is wired through like the
# other crates': mailbert reaches the embedding model through
# docbert-core and the k-means / MaxSim kernels through
# docbert-plaid, both of which build against a GPU.
#
# `gpg` stays a runtime lookup on `$PATH` (mailbert shells out to it
# to open an encrypted body), not a build input — same as docbert
# leaves its own external tools unwrapped.
{
  mkPackage,
  name ? "mailbert",
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

  shellCompletions = true;
}
