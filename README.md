# A Framework for Extracting and Encoding Features from Object-Centric Event Data

### Jan Niklas Adams, Gyunam Park, Sergej Levich, Daniel Schuster, Wil van der Aalst

To get an up-to-date version of the python library used please visit ocpa.readthedocs.io

Please first use anaconda prompt to create the environment.

``conda env create --file environment.yml``

then unzip the event log in example_logs/mdl/ and put it in the same directory.

Activate the environment in anaconda prompt

``conda activate icsoc``

Go into the repository directory and run 

``python experiments.py``

This will run all the experiments and reproduce the figures. Slight deviations might be due to differently initialized weights in the neural networks.

Note: To use the DGL library, you additionally have to place a config file in 

``~\.dgl\config.json``

containing only the line 
``{"backend":"tensorflow"}``

There might be an error if the environment was installed with a wrong version of a package.
https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly

run 
``pip install protobuf==3.20.*``
in that case.
