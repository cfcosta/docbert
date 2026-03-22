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
  };

  outputs =
    {
      nixpkgs,
      rust-overlay,
      treefmt-nix,
      ...
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
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
                overlays = [ (import rust-overlay) ];
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
                      rustfmt.package = rust;
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

              mkDocbert =
                {
                  name ? "docbert",
                  buildFeatures ? [ ],
                  buildInputs ? [ ],
                  nativeBuildInputs ? [ ],
                  extraEnv ? { },
                }:
                rustPlatform.buildRustPackage (
                  {
                    inherit
                      name
                      buildInputs
                      nativeBuildInputs
                      buildFeatures
                      ;
                    src = ./.;
                    cargoLock = {
                      lockFile = ./Cargo.lock;
                      outputHashes = {
                        "pylate-rs-1.0.4" = "sha256-l2bmTgAbxHa5ivdFqMrLts5O+MZSSWXKRK/rsVjeCzs=";
                      };
                    };
                    RUSTFLAGS = "-C target-cpu=native";

                    postInstall = ''
                      # Generate shell completions
                      mkdir -p $out/share/bash-completion/completions
                      mkdir -p $out/share/zsh/site-functions
                      mkdir -p $out/share/fish/vendor_completions.d

                      $out/bin/docbert completions bash > $out/share/bash-completion/completions/docbert
                      $out/bin/docbert completions zsh > $out/share/zsh/site-functions/_docbert
                      $out/bin/docbert completions fish > $out/share/fish/vendor_completions.d/docbert.fish
                    '';

                    meta.mainProgram = "docbert";
                  }
                  // extraEnv
                );
            in
            {
              inherit
                system
                pkgs
                rust
                formatter
                mkDocbert
                ;
            }
          )
        );
    in
    {
      packages = forEachSupportedSystem (
        { mkDocbert, pkgs, ... }:
        {
          default = mkDocbert { };
          docbert = mkDocbert { };
          docbert-cuda = mkDocbert {
            name = "docbert-cuda";
            buildFeatures = [ "cuda" ];

            nativeBuildInputs = with pkgs; [
              cudaPackages.cuda_nvcc
              autoAddDriverRunpath
            ];

            buildInputs = with pkgs.cudaPackages; [
              cuda_nvcc
              cudatoolkit
              cudnn
            ];

            extraEnv = {
              CUDA_COMPUTE_CAP = "80";
              CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
            };
          };
          docbert-metal = mkDocbert {
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
          default = pkgs.mkShell {
            name = "docbert";

            buildInputs = with pkgs; [
              rust
              formatter

              cargo-nextest
              cargo-mutants
              bacon
            ];
          };
        }
      );
    };
}
