# React/Vite UI bundled into the docbert binary at compile time.
#
# Built once via bun2nix and copied into `crates/docbert/ui/dist` by
# the docbert package's `preBuild`. Keeping the source filter here
# (instead of in flake.nix) means the list of files that participate
# in the build hash lives next to the files themselves.
#
# Adding a new top-level UI config (e.g. `postcss.config.js`) means
# adding it to the fileset below.
{ bun2nix, lib }:

let
  uiSrc = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./bun.lock
      ./bun.nix
      ./eslint.config.js
      ./index.html
      ./package.json
      ./public
      ./src
      ./tsconfig.app.json
      ./tsconfig.json
      ./tsconfig.node.json
      ./vite.config.ts
    ];
  };
in
bun2nix.mkDerivation {
  pname = "docbert-ui";
  src = uiSrc;
  packageJson = ./package.json;
  bunDeps = bun2nix.fetchBunDeps { bunNix = ./bun.nix; };
  buildPhase = ''
    bun run build
  '';
  installPhase = ''
    mkdir $out
    cp -rf dist/* $out/
  '';
}
