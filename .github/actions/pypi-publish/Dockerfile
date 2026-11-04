# Temporary eMudhra remediation: remove this overlay once upstream ships
# certifi>=2026.10.19, sigstore>=4.5.1, pip>=26.2.2 in both installation sites,
# and an OS trust store without emSign ECC Root CA - G3.
# Upstream v1.14.3 (7245906fb61fa33a13554f9d5fb856fe61758086), linux/amd64.
FROM ghcr.io/pypa/gh-action-pypi-publish@sha256:411241df99067c98b6d08fd8e26d14e86740a22460599c2af571e6dc26e27ffc

# The upstream image also predates the OS trust-store update. Distrust only the
# compromised root, preserving the unaffected emSign roots.
RUN sed -i 's|^mozilla/emSign_ECC_Root_CA_-_G3.crt$|!mozilla/emSign_ECC_Root_CA_-_G3.crt|' /etc/ca-certificates.conf \
    && update-ca-certificates --fresh

# Verify the exact wheel before either offline installation. Isolated mode
# bypasses upstream's user-site PYTHONPATH when upgrading the global copy.
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl', '/tmp/pip-26.2.2-py3-none-any.whl')" \
    && echo 'fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70  /tmp/pip-26.2.2-py3-none-any.whl' | sha256sum --check --strict \
    && python -m pip install --user --no-index --no-deps /tmp/pip-26.2.2-py3-none-any.whl \
    && python -I -m pip install --no-index --no-deps /tmp/pip-26.2.2-py3-none-any.whl \
    && rm /tmp/pip-26.2.2-py3-none-any.whl
