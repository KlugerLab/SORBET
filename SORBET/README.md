# SORBET: Spatial 'Omics Reasoning for Binary Label Tasks
This directory implements the code underlying SORBET. The directory structure is as follows:

```text
.
├── data_handling/        # Contains a basic Python class (OmicsGraph) for storing and processing graph-structured omics data. 
├── learning/             # Methods and models for learning from graph-structured omics data.
├── reasoning/            # Methods for underrstanding inferred models. 
```

All relevant methods should be accessible via the namespace. For example, one can import the `data_handling` name space and use the associated methods:
```Python
import SORBET
import SORBET.data_handling as data_handling

data_handling.OmicsGraph(...)
```
