# Demo assets

This directory contains molecule examples, model checkpoints, and prediction and evaluation outputs.

Large regression datasets matching `checkpoints_s2n_*/eval/plot_data/train_regression_points.csv` use Git LFS. Their filenames and CSV contents are unchanged.

Install [Git LFS](https://git-lfs.com/) on each computer before synchronizing the repository. On Ubuntu:

```bash
sudo apt-get install git-lfs
git lfs install
git pull --ff-only
git lfs pull
```

Use `git lfs ls-files` to check the tracked large files. Commit `.gitattributes` together with any new LFS tracking rules. Existing copies of these CSVs in older commits remain in Git history; this change does not rewrite shared commits.
