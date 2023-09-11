{ lib
, buildPythonPackage
, pythonRelaxDepsHook
, backoff
, joblib
, openai
, pandas
, spacy
, regex
, ujson
, tqdm
, datasets
}:

buildPythonPackage rec {
  pname = "dspy";
  version = "2023.09.10";

  src = ./.;

  propagatedBuildInputs = [
    backoff
    joblib
    openai
    pandas
    spacy
    regex
    ujson
    tqdm
    datasets
  ];

  nativeBuildInputs = [
    pythonRelaxDepsHook
  ];

  pythonRemoveDeps = [
    "jupyter"
  ];

  doCheck = false;

  meta = with lib; {
    homepage = "https://github.com/stanfordnlp/dspy";
    description = "DSPy: The framework for programming with foundation models";
    license = licenses.mit;
    maintainers = with maintainers; [ breakds ];
  };
}
