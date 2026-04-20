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

              uiPath = pkgs.bun2nix.mkDerivation {
                pname = "docbert-ui";
                src = ./crates/docbert/ui;
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

              mkPackage =
                {
                  name,
                  cargoPackage ? "docbert",
                  mainProgram ? "docbert",
                  buildFeatures ? [ ],
                  buildInputs ? [ ],
                  nativeBuildInputs ? [ ],
                  extraEnv ? { },
                  extraPreBuild ? "",
                }:
                rustPlatform.buildRustPackage (
                  {
                    inherit name buildInputs buildFeatures;
                    nativeBuildInputs = nativeBuildInputs;
                    src = ./.;
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
                    preBuild =
                      pkgs.lib.optionalString (uiPath != null) ''
                        rm -rf crates/docbert/ui/dist
                        mkdir -p crates/docbert/ui
                        cp -r ${uiPath} crates/docbert/ui/dist
                      ''
                      + extraPreBuild;

                    postInstall = ''
                      # Generate shell completions
                      mkdir -p $out/share/bash-completion/completions
                      mkdir -p $out/share/zsh/site-functions
                      mkdir -p $out/share/fish/vendor_completions.d

                      $out/bin/docbert completions bash > $out/share/bash-completion/completions/docbert
                      $out/bin/docbert completions zsh > $out/share/zsh/site-functions/_docbert
                      $out/bin/docbert completions fish > $out/share/fish/vendor_completions.d/docbert.fish
                    '';

                    meta.mainProgram = mainProgram;
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

        in
        {
          default = mkPackage { name = "docbert"; };

          docbert = mkPackage { name = "docbert"; };

          docbert-cuda = mkPackage {
            name = "docbert-cuda";
            buildFeatures = [ "cuda" ];
            nativeBuildInputs = cudaNativeBuildInputs ++ [ pkgs.git ];
            buildInputs = cudaBuildInputs;
            extraEnv = cudaEnv // {
              CUDAFORGE_HOME = "/tmp/cudaforge-cache";
            };
            extraPreBuild = ''
              dest=$CUDAFORGE_HOME/git/checkouts/cutlass-7d49e6c7e2f8896c
              mkdir -p $CUDAFORGE_HOME/git/checkouts
              cp -r ${nvidia-cutlass} $dest
              chmod -R u+w $dest

              # Stub a minimal .git dir so cudaforge's `git rev-parse HEAD`
              # returns the expected commit hash and skips any network fetch.
              mkdir -p $dest/.git/objects $dest/.git/refs
              echo "7d49e6c7e2f8896c47f586706e67e1fb215529dc" > $dest/.git/HEAD
            '';
          };

          docbert-metal = mkPackage {
            name = "docbert-metal";
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
