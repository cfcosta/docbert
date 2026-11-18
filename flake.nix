{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # The toolchain and the package builder: Rust's own release
    # manifests, and a builder that keeps compiled dependencies in a
    # derivation of their own.
    rust-dev-tools = {
      url = "github:cfcosta/rust-dev-tools.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    bun2nix = {
      url = "github:cfcosta/bun2nix";
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
      rust-dev-tools,
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
                overlays = [ bun2nix.overlays.default ];
                config.allowUnfree = true;
              };

              # Stable from rust-toolchain.toml, with rustfmt from a pinned
              # nightly: rustfmt.toml's import options are nightly-only.
              # The linker stays the platform default.
              rdt = rust-dev-tools.setup pkgs {
                name = "docbert";
                rust = rust-dev-tools.version.fromToolchainFile ./rust-toolchain.toml // {
                  rustfmt = rust-dev-tools.version.fromToolchain "nightly" "2026-08-04";
                };
                overrides = {
                  linux.useMold = false;
                  darwin.useLLD = false;
                };
              };
              inherit (rdt) rust;

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
                          "${./rustfmt.toml}"
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

              # Each crate's build logic lives in its own default.nix.
              # `callPackage` autoloads the rest of the args from `pkgs`
              # plus the extended scope below.
              uiPath = pkgs.callPackage ./crates/docbert-webui/ui { };

              # Build inputs for every rust derivation in the workspace.
              # Same rationale as `uiSrc` above, on a bigger scale: the
              # rust build is 10-30 minutes cold and the previous
              # `src = ./.` baked the entire repo into the derivation
              # hash — including `target/` (mutated by every local
              # `cargo build`), `.jj/` and `.git/` (every commit /
              # snapshot), `output/` / `pkg/` / `db/` (data dirs), and
              # the unfiltered `crates/docbert-webui/ui/` tree. Any of those
              # drifting between two `nix build` runs forced every
              # crate to recompile from scratch.
              #
              # We deliberately exclude `crates/docbert-webui/ui/` from the
              # rust src — `mkPackage` populates the prebuilt `dist/`
              # via `preBuild` from the cached `uiPath` derivation when
              # `bundleUi` is set, and Cargo never touches the rest of
              # the UI source tree. Stray `default.nix` files inside
              # `crates/*/` are harmless to cargo (it ignores anything
              # not referenced from `Cargo.toml`).
              rustSrc = pkgs.lib.fileset.toSource {
                root = ./.;
                fileset = pkgs.lib.fileset.unions [
                  ./Cargo.toml
                  ./Cargo.lock
                  ./rust-toolchain.toml
                  ./rustfmt.toml
                  ./deny.toml
                  (pkgs.lib.fileset.difference ./crates ./crates/docbert-webui/ui)
                  ./tests
                ];
              };

              # Builds one workspace binary as a Nix package.
              #
              # Defaults are tuned to the binaries we ship today:
              # - `cargoPackage` defaults to the part of `name` before
              #   the first `-`, so `docbert`, `docbert-cuda`, and
              #   `docbert-metal` all build the same `docbert` workspace
              #   member with different feature flags.
              # - `bundleUi` and `shellCompletions` default to `true`
              #   for `cargoPackage == "docbert"` because that's the
              #   binary that embeds the React UI and exposes a
              #   `completions` subcommand. Other binaries opt in
              #   explicitly — mailbert takes `shellCompletions` and
              #   leaves the UI behind, rustbert takes neither.
              #
              # Override any default by passing it explicitly. Adding a
              # fourth workspace binary with its own quirks (different
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
                    rm -rf crates/docbert-webui/ui/dist
                    mkdir -p crates/docbert-webui/ui
                    cp -r ${uiPath} crates/docbert-webui/ui/dist
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
                rdt.buildRustPackage {
                  inherit
                    name
                    buildFeatures
                    buildInputs
                    nativeBuildInputs
                    ;
                  pname = cargoPackage;
                  inherit ((builtins.fromTOML (builtins.readFile ./crates/${cargoPackage}/Cargo.toml)).package)
                    version
                    ;
                  src = rustSrc;
                  cargoPackages = [ cargoPackage ];
                  doCheck = false;
                  env = {
                    RUSTFLAGS = "-C target-cpu=native";
                  }
                  // extraEnv;
                  meta.mainProgram = mainProgram;
                  # `extraPreBuild` stages what build scripts read, and
                  # dependencies (candle-flash-attn) have build scripts too.
                  preBuildDependencies = extraPreBuild;
                  preBuild = uiPreBuild + extraPreBuild;
                  postInstall = completionsPostInstall;
                  # rust-dev-tools remaps the toolchain's paths in panic
                  # locations and debug info; fail the build if a reference
                  # survives anyway, since it would pull the toolchain into
                  # the closure (~2 GiB per `*-cuda` output).
                  disallowedReferences = [ rust ];
                };

              # Extended `callPackage` so each crate's `default.nix`
              # can take `mkPackage` as a function arg the same way it
              # takes `lib`, `bun2nix`, etc.
              callPackage = pkgs.lib.callPackageWith (pkgs // { inherit mkPackage; });
            in
            {
              inherit
                callPackage
                formatter
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
        { callPackage, pkgs, ... }:
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

          # Each crate's `default.nix` returns a function that takes
          # variant args (name, buildFeatures, …) and produces the
          # final derivation; partial-applying it once per crate here
          # keeps the variant call sites short.
          docbert = callPackage ./crates/docbert;
          mailbert = callPackage ./crates/mailbert;
          rustbert = callPackage ./crates/rustbert;
        in
        {
          default = docbert { };

          docbert = docbert { };

          docbert-cuda = docbert {
            name = "docbert-cuda";
            buildFeatures = [ "cuda" ];
            nativeBuildInputs = cudaNativeBuildInputs ++ [ pkgs.git ];
            buildInputs = cudaBuildInputs;
            extraEnv = cudaforgeEnv;
            extraPreBuild = cudaforgePreBuild;
          };

          docbert-metal = docbert {
            name = "docbert-metal";
            buildFeatures = [ "metal" ];
          };

          mailbert = mailbert { };

          mailbert-cuda = mailbert {
            name = "mailbert-cuda";
            buildFeatures = [ "cuda" ];
            nativeBuildInputs = cudaNativeBuildInputs ++ [ pkgs.git ];
            buildInputs = cudaBuildInputs;
            extraEnv = cudaforgeEnv;
            extraPreBuild = cudaforgePreBuild;
          };

          mailbert-metal = mailbert {
            name = "mailbert-metal";
            buildFeatures = [ "metal" ];
          };

          rustbert = rustbert { };

          rustbert-cuda = rustbert {
            name = "rustbert-cuda";
            buildFeatures = [ "cuda" ];
            nativeBuildInputs = cudaNativeBuildInputs ++ [ pkgs.git ];
            buildInputs = cudaBuildInputs;
            extraEnv = cudaforgeEnv;
            extraPreBuild = cudaforgePreBuild;
          };

          rustbert-metal = rustbert {
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
                  # sequoia-ipc generates its Cap'n Proto glue in a
                  # build script, so `mailbert` needs `capnp` to build.
                  capnproto
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
                # export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              '';
            }
          );
        }
      );
    };
}
