{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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
      flake-utils,
      pre-commit-hooks,
      rust-overlay,
      treefmt-nix,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config.allowUnfree = true;
        };
        inherit (pkgs) mkShell;

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
              prettier.enable = true;
              rustfmt = {
                enable = true;
                package = rust;
              };
              taplo.enable = true;
            };
          }).config.build.wrapper;

        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;

          hooks = {
            deadnix.enable = true;
            nixfmt-rfc-style.enable = true;
            treefmt = {
              enable = true;
              package = formatter;
            };
          };
        };

        mkDocbert =
          {
            name ? "docbert",
            extraFeatures ? [ ],
            extraBuildInputs ? [ ],
            extraNativeBuildInputs ? [ ],
            extraEnv ? { },
          }:
          rustPlatform.buildRustPackage (
            {
              inherit name;
              src = ./.;
              cargoLock.lockFile = ./Cargo.lock;
              buildFeatures = extraFeatures;
              buildInputs = [ pkgs.dbus.dev ] ++ extraBuildInputs;
              nativeBuildInputs = [ pkgs.pkg-config ] ++ extraNativeBuildInputs;

              RUSTFLAGS = "-C target-cpu=native";
            }
            // extraEnv
          );
      in
      {
        packages = {
          default = mkDocbert { };
          docbert = mkDocbert { };
          docbert-cuda = mkDocbert {
            name = "docbert-cuda";
            extraFeatures = [ "cuda" ];

            extraNativeBuildInputs = with pkgs.cudaPackages; [ cuda_nvcc ];

            extraBuildInputs = with pkgs.cudaPackages; [
              cuda_nvcc
              cudatoolkit
              cudnn
            ];

            extraEnv = {
              # Required so bindgen_cuda doesn't try to run nvidia-smi
              # (unavailable in the nix build sandbox).
              # Targets Ampere (sm_80) for broad forward compatibility.
              CUDA_COMPUTE_CAP = "80";
              # Point bindgen_cuda to the CUDA toolkit in the nix store
              # instead of searching /usr/local/cuda and other standard paths.
              CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
            };
          };
        };

        formatter = formatter;

        checks = { inherit pre-commit-check; };

        devShells.default = mkShell {
          name = "docbert";

          buildInputs = with pkgs; [
            rust
            formatter

            cargo-nextest
            cargo-mutants
            bacon

            pkg-config
            dbus.dev
          ];
        };
      }
    );
}
