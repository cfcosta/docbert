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

              mkRustWorkspacePackage =
                {
                  name,
                  cargoPackage,
                  mainProgram ? cargoPackage,
                  buildFeatures ? [ ],
                  buildInputs ? [ ],
                  nativeBuildInputs ? [ ],
                  extraEnv ? { },
                  uiDist ? null,
                  postInstall ? "",
                }:
                rustPlatform.buildRustPackage (
                  {
                    inherit
                      name
                      buildInputs
                      buildFeatures
                      postInstall
                      ;
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
                    preBuild = pkgs.lib.optionalString (uiDist != null) ''
                      rm -rf crates/docserver/ui/dist
                      mkdir -p crates/docserver/ui
                      cp -r ${uiDist}/dist crates/docserver/ui/dist
                    '';

                    meta.mainProgram = mainProgram;
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
                mkRustWorkspacePackage
                ;
            }
          )
        );
    in
    {
      packages = forEachSupportedSystem (
        {
          mkRustWorkspacePackage,
          pkgs,
          system,
          ...
        }:
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
          default = mkRustWorkspacePackage {
            name = "docbert";
            cargoPackage = "docbert";
            postInstall = ''
              # Generate shell completions
              mkdir -p $out/share/bash-completion/completions
              mkdir -p $out/share/zsh/site-functions
              mkdir -p $out/share/fish/vendor_completions.d

              $out/bin/docbert completions bash > $out/share/bash-completion/completions/docbert
              $out/bin/docbert completions zsh > $out/share/zsh/site-functions/_docbert
              $out/bin/docbert completions fish > $out/share/fish/vendor_completions.d/docbert.fish
            '';
          };
          docbert = mkRustWorkspacePackage {
            name = "docbert";
            cargoPackage = "docbert";
            postInstall = ''
              # Generate shell completions
              mkdir -p $out/share/bash-completion/completions
              mkdir -p $out/share/zsh/site-functions
              mkdir -p $out/share/fish/vendor_completions.d

              $out/bin/docbert completions bash > $out/share/bash-completion/completions/docbert
              $out/bin/docbert completions zsh > $out/share/zsh/site-functions/_docbert
              $out/bin/docbert completions fish > $out/share/fish/vendor_completions.d/docbert.fish
            '';
          };
          docbert-cuda = mkRustWorkspacePackage {
            name = "docbert-cuda";
            cargoPackage = "docbert";
            buildFeatures = [ "cuda" ];
            nativeBuildInputs = cudaNativeBuildInputs;
            buildInputs = cudaBuildInputs;
            extraEnv = cudaEnv;
            postInstall = ''
              # Generate shell completions
              mkdir -p $out/share/bash-completion/completions
              mkdir -p $out/share/zsh/site-functions
              mkdir -p $out/share/fish/vendor_completions.d

              $out/bin/docbert completions bash > $out/share/bash-completion/completions/docbert
              $out/bin/docbert completions zsh > $out/share/zsh/site-functions/_docbert
              $out/bin/docbert completions fish > $out/share/fish/vendor_completions.d/docbert.fish
            '';
          };
          docbert-metal = mkRustWorkspacePackage {
            name = "docbert-metal";
            cargoPackage = "docbert";
            buildFeatures = [ "metal" ];
            postInstall = ''
              # Generate shell completions
              mkdir -p $out/share/bash-completion/completions
              mkdir -p $out/share/zsh/site-functions
              mkdir -p $out/share/fish/vendor_completions.d

              $out/bin/docbert completions bash > $out/share/bash-completion/completions/docbert
              $out/bin/docbert completions zsh > $out/share/zsh/site-functions/_docbert
              $out/bin/docbert completions fish > $out/share/fish/vendor_completions.d/docbert.fish
            '';
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
                  rust
                  formatter

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
