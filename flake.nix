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
  };

  outputs =
    {
      bun2nix,
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
                    cargoLock = {
                      lockFile = ./Cargo.lock;
                      outputHashes = {
                        "pylate-rs-1.0.4" = "sha256-l2bmTgAbxHa5ivdFqMrLts5O+MZSSWXKRK/rsVjeCzs=";
                      };
                    };
                    RUSTFLAGS = "-C target-cpu=native";
                    preBuild = pkgs.lib.optionalString (uiPath != null) ''
                      rm -rf crates/docserver/ui/dist
                      mkdir -p crates/docserver/ui
                      cp -r ${uiPath} crates/docserver/ui/dist
                    '';

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
            nativeBuildInputs = cudaNativeBuildInputs;
            buildInputs = cudaBuildInputs;
            extraEnv = cudaEnv;
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
                  cargo-mutants
                  cargo-nextest
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
