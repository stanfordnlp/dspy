{
  description = "DSPy: The framework for programming with foundation models";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";

    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, ... }@inputs: {
    overlays.default = final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (py-final: py-prev: {
          dspy = py-final.callPackage ./default.nix {};
        })
      ];
    };

  } // inputs.utils.lib.eachSystem [
    "x86_64-linux" "x86_64-darwin"
  ] (system:
    let pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
    in {
      # The following creates the development environment for this project.
      devShells.default = let
        python-env = pkgs.python3.withPackages (pyPkgs: with pyPkgs; [
          # Library dependencies.
          backoff
          joblib
          openai
          pandas
          spacy
          regex
          ujson
          tqdm
          datasets

          # Jupyter Lab
          jupyterlab
          ipywidgets
          jupyterlab-widgets
        ]);

        name = "DSPy";
      in pkgs.mkShell {
        inherit name;

        packages = [
          python-env
        ];

        shellHooks = let pythonIcon = "f3e2"; in ''
          export PS1="$(echo -e '\u${pythonIcon}') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} (${name}) \\$ \[$(tput sgr0)\]"
        '';
      };

      packages.default = let pkgs' = import nixpkgs {
        inherit system;
        overlays = [ self.overlays.default ];
      }; in pkgs'.python3Packages.dspy;
    });
}
