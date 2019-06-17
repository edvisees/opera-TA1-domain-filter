# Setup:

Create conda env:

`conda env create -f ./conda_env_domain_filter.yml`

Make sure the resources symlink points to the resources directory:
```bash
ls resources/xiang/domain_filter/
en_best.hdf5  ru_best.hdf5  tfidf0en.pkl  tfidf0.pkl  tfidf0ru.pkl  tfidf0uk.pkl  uk_best.hdf5
```

# Usage:

```bash
source activate domain_filter
python xiang/domain_filter/test.py input_ltf_dir/
source deactivate
```
