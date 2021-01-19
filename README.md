# PyClee

This is a Python implementation of the **DyClee** (Dynamic clustering for tracking evolving environments) algorithm proposed by Nathalie Barbosa Roa, Louise Travé-Massuyès and Victor H. Grisales-Palacio in https://doi.org/10.1016/j.patcog.2019.05.024.

## Differences compared to the paper
- Density stage: this algorithm was reworked almost entirely based on the text (and on performance considerations) rather than the given Algorithm 2, which seemed to be inconsistent with the text in these ways:
    - Algorithm 2 seems to make no distinction between dense and semi-dense microclusters, which, according to the text, is important in defining the interior vs. the boundary of a cluster
    - Algorithm 2 assigns the `cid` label to any-density second neighbours, but only to dense first neighbours.
- The definition of direct connectedness given in Equation 5 involves a factor of `1/2` which is inconsistent with the stated condition that the hyperboxes of two given microclusters must overlap to be directly connected. An overlap would mean `max_i|c_ij - c_ik| < S_i for all i`, so this condition was used instead.
- In the paper, microclusters may not skip a step when moving between the density partitions (e.g., dense microclusters may not become outliers, and outliers may not become dense). I believe that this creates a risk of trapping microclusters in the dense or outlier groups, especially in cases where the density-based stage is not applied for every sample, so in this implementation these restrictions are lifted.
- Here, the density partitioning step is implemented as a separate step between the distance stage and the density stage, and its frequency is controlled with a separate setting (`partitioning_interval`).
- The paper favoured using a KDTree for spatial indexing. I found this to be too slow as the tree has to be rebuilt from scratch every time a microcluster is added or changed. I've made the R*Tree structure from the `rtree` package the default, as it can be updated with new elements so only has to be created once. This resulted in better performance. (Note: I may not be using the KDTree optimally, this could be looked into.)

I'm looking for input from the original authors to clarify these points, but for now these changes represent my best effort to get sensible results.

Note also that many variable names have been changed to conform to a consistent style.

## Examples

See [`Examples.ipynb`](./Examples.ipynb) for more details.

### Static

```python
# See Examples.ipynb for imports etc.
context = DyCleeContext(2, 0.06, bounds, store_elements=True)
dy = DyClee(context)
clusters = dy.run(X)

MultiPlotter(dy).plot_snapshot(clusters)
```

![blobs-separate](https://user-images.githubusercontent.com/1812261/104946855-af98a780-59b2-11eb-9558-1a03dd7785c0.png)

### Dynamic ("concept drift")

```python
# See Examples.ipynb for imports etc.
context = DyCleeContext(2, 0.06, bounds, forgetting_method=ExponentialForgettingMethod(0.01))
dy = DyClee(context)

fig, ax = plt.subplots()
MultiPlotter(dy, ax, centroids=False).animate(X)
```

![drift-16x](https://user-images.githubusercontent.com/1812261/104943086-22068900-59ad-11eb-9a4f-4b9cd3134acb.gif)


## TODO
- Local-density approach
- Outlier rejection ("`Unclass_accepted`" in the paper)
- Sparse rejection ("`minimum_mc`")
- pip package
- Tests
- Benchmarks

## Contributing
Discussions in issues and pull requests are welcome, in particular in regards to the notes above.

This project uses a [modified version of the Black code style](https://github.com/harenbrs/bleck).