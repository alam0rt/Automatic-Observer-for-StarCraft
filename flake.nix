{
  description = "Automatic Observer for StarCraft: replay feature extraction (OpenBW + BWEM + FAP) and model training";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";

    # Only Stardust's 3rdparty/ tree is used: OpenBW, its BWAPI port, BWEM and
    # Stardust's modified FAP (which takes upgraded unit values directly).
    stardust = {
      url = "github:alam0rt/Stardust/22d93d7a55d0a0494384474a456fd7ee26baee97";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    stardust,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      # torch-bin bundles the CUDA runtime libraries, which are unfree.
      config.allowUnfree = true;
    };
    llvm = pkgs.llvmPackages;

    # StarCraft 1.16.1. Same name and hash as nix-config's pkgs/starcraft-1161,
    # so this resolves to the store path that is already there.
    starcraftZip = pkgs.fetchurl {
      name = "Starcraft_1161.zip";
      url = "https://davechurchill.ca/starcraft/files/Starcraft_1161.zip";
      hash = "sha256-G58L9bcZxZ7ERWO6Dfg0v8cIczIxXXqeZ7BzEmiukNw=";
    };

    # OpenBW opens Patch_rt.mpq, BrooDat.mpq and StarDat.mpq by exact name
    # (openbw/data_loading.h), and Linux is case-sensitive: the zip ships them
    # as STARDAT.MPQ, BROODAT.MPQ and patch_rt.mpq. Kept byte-identical to
    # Stardust's flake so both share one store path.
    scData = pkgs.runCommand "starcraft-mpqs" {nativeBuildInputs = [pkgs.unzip];} ''
      unzip -j ${starcraftZip} STARDAT.MPQ BROODAT.MPQ patch_rt.mpq -d mpq
      mkdir -p $out
      mv mpq/STARDAT.MPQ $out/StarDat.mpq
      mv mpq/BROODAT.MPQ $out/BrooDat.mpq
      mv mpq/patch_rt.mpq $out/Patch_rt.mpq
    '';

    stardustSrc = pkgs.applyPatches {
      name = "stardust-src";
      src = stardust;
      # Several BWAPI headers use std::unique_ptr / std::shared_ptr without
      # including <memory>, which current libstdc++ no longer pulls in
      # transitively.
      postPatch = ''
        grep -rlE 'std::(unique|shared)_ptr' 3rdparty/openbw/bwapi/bwapi/include | while read -r header; do
          grep -q '#include <memory>' "$header" || sed -i '1i #include <memory>' "$header"
        done
      '';
    };

    scExtract = llvm.stdenv.mkDerivation {
      pname = "sc-extract";
      version = "0.1.0";
      src = ./extractor;

      nativeBuildInputs = [pkgs.cmake pkgs.ninja pkgs.makeWrapper];
      cmakeFlags = ["-DSTARDUST_DIR=${stardustSrc}"];
      # The cc-wrapper's _FORTIFY_SOURCE trips over OpenBW's headers.
      hardeningDisable = ["fortify"];

      postInstall = ''
        wrapProgram $out/bin/sc-extract --set-default OPENBW_MPQ_PATH ${scData}/
      '';

      meta = {
        description = "Replay a Brood War game in OpenBW and dump per-frame units, BWEM terrain and FAP fight estimates";
        mainProgram = "sc-extract";
        platforms = ["x86_64-linux"];
      };
    };

    python = pkgs.python3.withPackages (ps:
      with ps; [
        numpy
        pandas
        pyarrow
        scipy
        tqdm
        matplotlib
        torch-bin
        torchvision-bin
        pytest
      ]);
  in {
    packages.${system} = {
      default = scExtract;
      sc-extract = scExtract;
      sc-data = scData;
    };

    apps.${system}.default = {
      type = "app";
      program = pkgs.lib.getExe scExtract;
    };

    devShells.${system}.default = (pkgs.mkShell.override {stdenv = llvm.stdenv;}) {
      packages = [
        python
        scExtract
        pkgs.cmake
        pkgs.ninja
        pkgs.gdb
      ];
      hardeningDisable = ["fortify"];

      # For iterating on extractor/ outside of nix build:
      #   cmake -S extractor -B build -G Ninja && cmake --build build
      STARDUST_DIR = stardustSrc;
      OPENBW_MPQ_PATH = "${scData}/";
      PYTHONPATH = "${toString ./.}";
    };
  };
}
