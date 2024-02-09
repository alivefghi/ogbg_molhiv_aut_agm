# OGBG-MolHIV

This is a repository for Algorithm, Graph and Machine learning (AGM) lab at Amirkabir University of Technology (AUT).

Graph: The [ogbg-molhiv](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) and ogbg-molpcba datasets are two molecular property prediction datasets of different sizes: ogbg-molhiv (small) and ogbg-molpcba (medium). They are adopted from the MoleculeNet [1], and are among the largest of the MoleculeNet datasets. All the molecules are pre-processed using RDKit [2]. Each graph represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features are 9-dimensional, containing atomic number and chirality, as well as other additional atom features such as formal charge and whether the atom is in the ring or not. The full description of the features is provided in code. The script to convert the SMILES string [3] to the above graph object can be found here. Note that the script requires RDkit to be installed. The script can be used to pre-process external molecule datasets so that those datasets share the same input feature space as the OGB molecule datasets. This is particularly useful for pre-training graph models, which has great potential to significantly increase generalization performance on the (downstream) OGB datasets [4].For encoding these raw input features, we prepare simple modules called AtomEncoder and BondEncoder. They can be used as follows to embed raw atom and bond features to obtain atom_emb and bond_emb.

Prediction task: The task is to predict the target molecular properties as accurately as possible, where the molecular properties are cast as binary labels, e.g, whether a molecule inhibits HIV virus replication or not. Note that some datasets (e.g., ogbg-molpcba) can have multiple tasks, and can contain nan that indicates the corresponding label is not assigned to the molecule. For evaluation metric, we closely follow [1]. Specifically, for ogbg-molhiv, we use ROC-AUC for evaluation. For ogbg-molpcba, as the class balance is extremely skewed (only 1.4% of data is positive) and the dataset contains multiple classification tasks, we use the Average Precision (AP) averaged over the tasks as the evaluation metric.

Dataset splitting: We adopt the scaffold splitting procedure that splits the molecules based on their two-dimensional structural frameworks. The scaffold splitting attempts to separate structurally different molecules into different subsets, which provides a more realistic estimate of model performance in prospective experimental settings [1].

References
[1] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh SPappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical Science, 9(2):513–530, 2018.
[2] Greg Landrum et al. RDKit: Open-source cheminformatics, 2006.
[3] Eric Anderson, Gilman D. Veith, and David Weininger. SMILES: a line notation and computerized interpreter for chemical structures, 1987.
[4] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec.Strategies for pre-training graph neural networks. In International Conference on Learning Representations (ICLR), 2020.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

To project structured in the following ways:

    .
    ├── code
    │   ├── dataset             # Dataset
    │   ├── utils               # Helper classes and functions
    │   └── notebooks           # Python notebooks
    └── docs                    # Documentation files and Miscellaneous information

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alivefghi/ogbg_molhiv_aut_agm.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Solve the issue regarding the GraKel dependency.

Inside this project, we are using GraKel library for their good implementation of the Graph Kernels. However, there is currently a bug in their library after they updated their codebase. I reported their problem inside this [Print <class 'grakel.graph.Graph'>](https://github.com/ysig/GraKeL/issues/102). To solve the problem for now your project please follow this suggestion fix on the library code.

Update following code to the `get_edges` function at `site-packages/grakel/graph.py`

```python
    def get_edges(self, purpose="adjacency", with_weights=False):
        """Create an iterable of edges as tuples.

        Parameters
        ----------
        purpose : str, valid_values={"adjacency", "dictionary"}, default="adjacency"
            Defines if the edges is given for the "dictionary" format of the
            graph (symbol) to the "adjacency" (index).

        Returns
        -------
        vertices : list
            Returns a list of tuples for edges.

        """
        if purpose not in ["adjacency", "dictionary", "any"]:
            raise ValueError('purpose is either "adjacency" of "dictionary"')

        if purpose == "any":
            if self._format in ['all', 'adjacency']:
                purpose = "adjacency"
            else:
                purpose = "dictionary"

        if purpose == "adjacency":
            self.desired_format("adjacency", warn=True)
            idx_i, idx_j = np.where(self.adjacency_matrix > 0)
            edges = zip(idx_i, idx_j)
            if with_weights:
                return list(zip(edges, self.adjacency_matrix[idx_i, idx_j]))
            else:
                return list(edges)
        if purpose == "dictionary":
            self.desired_format("dictionary", warn=True)
            if with_weights:
                return [(i, j) for i in self.edge_dictionary.keys()
                        for j in self.edge_dictionary[i].keys()]
            else:
                return [((i, j), self.edge_dictionary[i][j])
                        for i in self.edge_dictionary.keys()
                        for j in self.edge_dictionary[i].keys()]
```

## Usage
To run the project, you can run the notebooks.


## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes.
4. Push your branch: `git push origin feature-name`.
5. Create a pull request.


## License
This project is licensed under the [MIT License](LICENSE)