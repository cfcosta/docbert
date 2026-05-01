{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    bun2nix = {
      url = "github:nix-community/bun2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nvidia-cutlass = {
      url = "github:NVIDIA/cutlass/7d49e6c7e2f8896c47f586706e67e1fb215529dc";
      flake = false;
    };
  };

  outputs =
    {
      bun2nix,
      nvidia-cutlass,
      nixpkgs,
      rust-overlay,
      treefmt-nix,
      ...
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f (
            let
              pkgs = import nixpkgs {
                inherit system;
                overlays = [
                  (import rust-overlay)
                  bun2nix.overlays.default
                ];
                config.allowUnfree = true;
              };

              rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

              rustPlatform = pkgs.makeRustPlatform {
                rustc = rust;
                cargo = rust;
              };

              formatter =
                (treefmt-nix.lib.evalModule pkgs {
                  projectRootFile = "flake.nix";

                  settings = {
                    allow-missing-formatter = true;
                    verbose = 0;

                    global.excludes = [ "*.lock" ];

                    formatter = {
                      nixfmt.options = [ "--strict" ];

                      rustfmt = {
                        package = rust;

                        options = [
                          "--config-path"
                          (toString ./rustfmt.toml)
                        ];
                      };
                    };
                  };

                  programs = {
                    nixfmt.enable = true;
                    oxfmt.enable = true;
                    rustfmt = {
                      enable = true;
                      package = rust;
                    };
                    taplo.enable = true;
                  };
                }).config.build.wrapper;

              # Build inputs for the UI derivation. Limited to the files
              # bun + Vite actually need so that local `bun install` /
              # `bun run build` runs (which mutate `node_modules/` and
              # `dist/` inside the source tree) don't bust the Nix
              # derivation hash and force a rebuild from scratch.
              # Adding a new top-level config file (e.g. a postcss
              # config) means adding it here too.
              uiSrc = pkgs.lib.fileset.toSource {
                root = ./crates/docbert/ui;
                fileset = pkgs.lib.fileset.unions [
                  ./crates/docbert/ui/bun.lock
                  ./crates/docbert/ui/bun.nix
                  ./crates/docbert/ui/eslint.config.js
                  ./crates/docbert/ui/index.html
                  ./crates/docbert/ui/package.json
                  ./crates/docbert/ui/public
                  ./crates/docbert/ui/src
                  ./crates/docbert/ui/tsconfig.app.json
                  ./crates/docbert/ui/tsconfig.json
                  ./crates/docbert/ui/tsconfig.node.json
                  ./crates/docbert/ui/vite.config.ts
                ];
              };

              uiPath = pkgs.bun2nix.mkDerivation {
                pname = "docbert-ui";
                src = uiSrc;
                packageJson = ./crates/docbert/ui/package.json;
                bunDeps = pkgs.bun2nix.fetchBunDeps { bunNix = ./crates/docbert/ui/bun.nix; };
                buildPhase = ''
                  bun run build
                '';
                installPhase = ''
                  mkdir $out
                  cp -rf dist/* $out/
                '';
              };

              # Build inputs for every rust derivation in the workspace.
              # Same rationale as `uiSrc` above, on a bigger scale: the
              # rust build is 10-30 minutes cold and the previous
              # `src = ./.` baked the entire repo into the derivation
              # hash — including `target/` (mutated by every local
              # `cargo build`), `.jj/` and `.git/` (every commit /
              # snapshot), `output/` / `pkg/` / `db/` (data dirs), and
              # the unfiltered `crates/docbert/ui/` tree. Any of those
              # drifting between two `nix build` runs forced every
              # crate to recompile from scratch.
              #
              # We deliberately exclude `crates/docbert/ui/` from the
              # rust src — `mkPackage` populates the prebuilt `dist/`
              # via `preBuild` from the cached `uiPath` derivation when
              # `bundleUi` is set, and Cargo never touches the rest of
              # the UI source tree.
              rustSrc = pkgs.lib.fileset.toSource {
                root = ./.;
                fileset = pkgs.lib.fileset.unions [
                  ./Cargo.toml
                  ./Cargo.lock
                  ./rust-toolchain.toml
                  ./rustfmt.toml
                  ./deny.toml
                  (pkgs.lib.fileset.difference ./crates ./crates/docbert/ui)
                  ./tests
                ];
              };

              # Builds one workspace binary as a Nix package.
              #
              # Defaults are tuned to the two binaries we ship today:
              # - `cargoPackage` defaults to the part of `name` before
              #   the first `-`, so `docbert`, `docbert-cuda`, and
              #   `docbert-metal` all build the same `docbert` workspace
              #   member with different feature flags.
              # - `bundleUi` and `shellCompletions` default to `true`
              #   for `cargoPackage == "docbert"` because that's the
              #   binary that embeds the React UI and exposes a
              #   `completions` subcommand. Other binaries (rustbert
              #   today, hypothetical future ones) opt in explicitly.
              #
              # Override any default by passing it explicitly. Adding a
              # third workspace binary with its own quirks (different
              # main program, different bundling) is just another set
              # of arguments here, no new helper required.
              mkPackage =
                {
                  name,
                  cargoPackage ? builtins.head (pkgs.lib.splitString "-" name),
                  mainProgram ? cargoPackage,
                  bundleUi ? cargoPackage == "docbert",
                  shellCompletions ? cargoPackage == "docbert",
                  buildFeatures ? [ ],
                  buildInputs ? [ ],
                  nativeBuildInputs ? [ ],
                  extraEnv ? { },
                  extraPreBuild ? "",
                }:
                let
                  uiPreBuild = pkgs.lib.optionalString bundleUi ''
                    rm -rf crates/docbert/ui/dist
                    mkdir -p crates/docbert/ui
                    cp -r ${uiPath} crates/docbert/ui/dist
                  '';
                  completionsPostInstall = pkgs.lib.optionalString shellCompletions ''
                    mkdir -p $out/share/bash-completion/completions
                    mkdir -p $out/share/zsh/site-functions
                    mkdir -p $out/share/fish/vendor_completions.d

                    $out/bin/${mainProgram} completions bash > $out/share/bash-completion/completions/${mainProgram}
                    $out/bin/${mainProgram} completions zsh > $out/share/zsh/site-functions/_${mainProgram}
                    $out/bin/${mainProgram} completions fish > $out/share/fish/vendor_completions.d/${mainProgram}.fish
                  '';
                in
                rustPlatform.buildRustPackage (
                  {
                    inherit
                      name
                      buildFeatures
                      buildInputs
                      nativeBuildInputs
                      ;
                    src = rustSrc;
                    cargoBuildFlags = [
                      "-p"
                      cargoPackage
                    ];
                    cargoTestFlags = [
                      "-p"
                      cargoPackage
                    ];
                    doCheck = false;
                    cargoLock.lockFile = ./Cargo.lock;
                    RUSTFLAGS = "-C target-cpu=native";
                    meta.mainProgram = mainProgram;
                    preBuild = uiPreBuild + extraPreBuild;
                    postInstall = completionsPostInstall;
                  }
                  // extraEnv
                );
            in
            {
              inherit
                formatter
                mkPackage
                pkgs
                rust
                system
                ;
            }
          )
        );
    in
    {
      packages = forEachSupportedSystem (
        { mkPackage, pkgs, ... }:
        let
          cudaNativeBuildInputs = with pkgs; [
            cudaPackages.cuda_nvcc
            autoAddDriverRunpath
          ];
          cudaBuildInputs = with pkgs.cudaPackages; [
            cuda_nvcc
            cudatoolkit
            cudnn
          ];
          cudaEnv = {
            CUDA_COMPUTE_CAP = "80";
            CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          };
          # cudaforge fetches NVIDIA/cutlass via git at build time.
          # Pre-stage a sandbox-resident copy with a stubbed `.git` so
          # the build doesn't need network and `git rev-parse HEAD`
          # returns the pinned commit. Shared between every
          # `*-cuda` output that pulls candle-flash-attn.
          cudaforgeEnv = cudaEnv // {
            CUDAFORGE_HOME = "/tmp/cudaforge-cache";
          };
          cudaforgePreBuild = ''
            dest=$CUDAFORGE_HOME/git/checkouts/cutlass-7d49e6c7e2f8896c
            mkdir -p $CUDAFORGE_HOME/git/checkouts
            cp -r ${nvidia-cutlass} $dest
            chmod -R u+w $dest

            # Stub a minimal .git dir so cudaforge's `git rev-parse HEAD`
            # returns the expected commit hash and skips any network fetch.
            mkdir -p $dest/.git/objects $dest/.git/refs
            echo "7d49e6c7e2f8896c47f586706e67e1fb215529dc" > $dest/.git/HEAD
          '';

        in
        {
          default = mkPackage { name = "docbert"; };

          docbert = mkPackage { name = "docbert"; };

          docbert-cuda = mkPackage {
            name = "docbert-cuda";
            buildFeatures = [ "cuda" ];
            nativeBuildInputs = cudaNativeBuildInputs ++ [ pkgs.git ];
            buildInputs = cudaBuildInputs;
            extraEnv = cudaforgeEnv;
            extraPreBuild = cudaforgePreBuild;
          };

          docbert-metal = mkPackage {
            name = "docbert-metal";
            buildFeatures = [ "metal" ];
          };

          rustbert = mkPackage { name = "rustbert"; };

          rustbert-cuda = mkPackage {
            name = "rustbert-cuda";
            buildFeatures = [ "cuda" ];
            nativeBuildInputs = cudaNativeBuildInputs ++ [ pkgs.git ];
            buildInputs = cudaBuildInputs;
            extraEnv = cudaforgeEnv;
            extraPreBuild = cudaforgePreBuild;
          };

          rustbert-metal = mkPackage {
            name = "rustbert-metal";
            buildFeatures = [ "metal" ];
          };
        }
      );

      formatter = forEachSupportedSystem ({ formatter, ... }: formatter);

      devShells = forEachSupportedSystem (
        {
          pkgs,
          rust,
          formatter,
          ...
        }:
        {
          default = pkgs.mkShell (
            {
              name = "docbert";

              buildInputs =
                with pkgs;
                [
                  bun2nix.packages.${pkgs.stdenv.hostPlatform.system}.default
                  formatter
                  rust

                  bacon
                  bun
                  cargo-deny
                  cargo-mutants
                  cargo-nextest
                  uv
                ]
                ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux (
                  with pkgs.cudaPackages;
                  [
                    cuda_nvcc
                    cudatoolkit
                    cudnn
                  ]
                );
            }
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              CUDA_COMPUTE_CAP = "80";
              CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";

              shellHook = ''
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              '';
            }
          );
        }
      );
    };
}
