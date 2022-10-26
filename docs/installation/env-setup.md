# Install DeepCV on your local machine

## Clone DeepCV repo or download a tarball to your local machine

For git, clone with SSH is recommended:
```sh
git clone git@gitlab.uzh.ch:lubergroup/deepcv.git
```

You can also use command `git checkout` to switch between `master` and `dev` branches.

## Environment setup & install dependencies

```sh
cd deepcv/
conda env create --file environment.yml
conda activate deepcv
```

This command will create a conda environment named `deepcv` based on parameters defined in `environment.yml` file. You can also change the name using argument `--name NEW_NAME`.
