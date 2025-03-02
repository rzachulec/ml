# B&C modelling

This repository contains code used to conduct modelling of data collected as part of a study on building facade temperatures in relation to meteorological data.

## Contents

**BCformatter.py** processes data stored in *./data/data.xslx* and returns a formatted dataset in *./data/training_data.csv*. **BCtrain.py** is responsible for training and contains code for some initial data preprocessing. To visualise model ouputs, **BCgraph.py** is used to render plots. **BCshap** generates explainer value plots.

## How to run

Make sure you have python 3.+ and pip installed. 
Clone the repository. Navigate to the created folder and in terminal, run the following commands (linux/macos): 

    python -m venv .venv
then

    source .venv/bin/activate

and

    pip install -r requirements.txt

to run predictions/training on GPU, install tensorflow extensions; eg to run on Apple Silicon GPU, install *tensorflow-metal*.
