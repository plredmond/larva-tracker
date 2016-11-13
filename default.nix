let
  pkgs = import <nixpkgs> {};
in

pkgs.stdenv.mkDerivation {
  name = "python-nix";
  version = "0.1.0.0";
  src = ./.;
  buildInputs = [
    pkgs.python27Packages.python
    (pkgs.python27Packages.opencv.override {
      enableGtk2 = true;
      enableFfmpeg = true;
    })
    pkgs.python27Packages.numpy
  ];
}
