# Efficient Fine-Tuning

## Setup

Our code is guaranteed to run on Python 3.10 and CUDA 12.6. With miniconda, you can create a new environment with the following command:

```bash
conda env create -f environment.yml --name IF
```

> (Optional) In some special projection setting, please also install the [`sjlt` library](https://github.com/TRAIS-Lab/sjlt/tree/main) following the installation guide.

## Running Jobs

To run the jobs, you can use the following command:
```bash
nohup bash script.sh > output.log 2>&1 &
```

The `job` directory contains all the jobs we have implemented. You can run them by executing the corresponding script files. The main experiment is controlled by `experiment/job/master.sh`


## Supervised fine-tuning of GPT-2 on Self-Instruct

### Setup

For the credential: we can simply use the temporary credentials obtained in Isengard:

![Temporary Credentials from Isengard](else/credentials.png)

Then the credentials will be automatically handled in `sdgen.py`.