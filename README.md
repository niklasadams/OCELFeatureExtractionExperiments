# A Framework for Extracting and Encoding Features from Object-Centric Event Data

### Jan Niklas Adams, Gyunam Park, Sergej Levich, Daniel Schuster, Wil van der Aalst

If you find the provided code or the corresponding paper useful, please consider citing the paper:
``` {.text}
@InProceedings{10.1007/978-3-031-20984-0_3,
author="Adams, Jan Niklas
and Park, Gyunam
and Levich, Sergej
and Schuster, Daniel
and van der Aalst, Wil M. P.",
editor="Troya, Javier
and Medjahed, Brahim
and Piattini, Mario
and Yao, Lina
and Fern{\'a}ndez, Pablo
and Ruiz-Cort{\'e}s, Antonio",
title="A Framework for Extracting and Encoding Features from Object-Centric Event Data",
booktitle="Service-Oriented Computing",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="36--53",
isbn="978-3-031-20984-0"
}
```

The paper is available at: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-20984-0_3)

To get an up-to-date version of the python library used please visit ocpa.readthedocs.io
_____________

### Instructions

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
