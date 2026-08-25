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
# `capnproto` is a build-time tool, not a runtime one: sequoia-ipc
# (reached through sequoia-gpg-agent) generates its Cap'n Proto glue
# in a build script and panics without `capnp` on `$PATH`. It is added
# here rather than at each call site so every variant inherits it.
#
# Nothing external is needed at runtime. §5.4 opens an encrypted body
# in-process with sequoia, and the one operation that needs a secret
# key goes to gpg-agent over its socket, so no `gpg` is spawned.
{
  mkPackage,
  capnproto,
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
    extraEnv
    extraPreBuild
    ;

  nativeBuildInputs = nativeBuildInputs ++ [ capnproto ];

  shellCompletions = true;
}
