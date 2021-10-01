# Automated Argo workflows

If you'd like to automate your Jupyter notebooks using Argo, please use these kustomize manifests. If you follow the steps bellow, your application is fully set and ready to be deployed via Argo CD.

For a detailed guide on how to adjust your notebooks etc, please consult [documentation](https://github.com/aicoe-aiops/data-science-workflows/blob/master/Automating%20via%20Argo.md)

1. Replace all `<VARIABLE>` mentions with your project name, respective url or any fitting value
2. Define your automation run structure in the `templates` section of [`cron-workflow.yaml`](./cron-workflow.yml)
3. Set up `sops`:

   1. Install `go` from your distribution repository
   2. Setup `GOPATH`

      ```bash
      echo 'export GOPATH="$HOME/.go"' >> ~/.bashrc
      echo 'export PATH="${GOPATH//://bin:}/bin:$PATH"' >> ~/.bashrc
      source  ~/.bashrc
      ```

   3. Install `sops` from your distribution repository if possible or use [sops GitHub release binaries](https://github.com/mozilla/sops#stable-release)

   4. Import AICoE-SRE's public key [EFDB9AFBD18936D9AB6B2EECBD2C73FF891FBC7E](https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xefdb9afbd18936d9ab6b2eecbd2c73ff891fbc7e):

      ```bash
      gpg --keyserver keyserver.ubuntu.com --recv EFDB9AFBD18936D9AB6B2EECBD2C73FF891FBC7E
      ```

   5. Import tcoufal's ([A76372D361282028A99F9A47590B857E0288997C](https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xa76372d361282028a99f9a47590b857e0288997c)) and mhild's [04DAFCD9470A962A2F272984E5EB0DA32F3372AC](https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x04dafcd9470a962a2f272984e5eb0da32f3372ac) keys (so they can help)

      ```bash
      gpg --keyserver keyserver.ubuntu.com --recv A76372D361282028A99F9A47590B857E0288997C  # tcoufal
      gpg --keyserver keyserver.ubuntu.com --recv 04DAFCD9470A962A2F272984E5EB0DA32F3372AC  # mhild
      ```

   6. If you'd like to be able to build the manifest on your own as well, please list your GPG key in the [`.sops.yaml` file](.sops.yaml), `pgp` section (add to the comma separated list). With your key present there, you can later generate the full manifests using `kustomize` yourself (`ksops` has to be installed, please follow ksops [guide](https://github.com/viaduct-ai/kustomize-sops#0-verify-requirements).

4. Create a secret and encrypt it with `sops`:

   ```bash
   # If you're not already in the `manifest` folder, cd here
   cd manifests
   # Mind that `SECRET_NAME` must match the `SECRET_NAME` used in `cron-workflow.yaml`
   oc create secret generic <SECRET_NAME> \
     --from-literal=path=<BASE_PATH_WITHIN_CEPH_BUCKET> \
     --from-literal=bucket=<BUCKET> \
     --from-literal=access-key-id=<AWS_ACCESS_KEY_ID> \
     --from-literal=secret-access-key=<AWS_SECRET_ACCESS_KEY> \
     --dry-run -o yaml |
   sops --input-type=yaml --output-type=yaml -e /dev/stdin > ceph-creds.yaml
   ```

Note: You can use the S2I image, that was built by [s2i-custom-notebook](https://github.com/AICoE/s2i-custom-notebook) for this automation. This image is expected to be used by default, therefore the `workingDir` is adjusted to `/opt/app-root/backup`. Please change or remove this settings in case you plan on using different image.
