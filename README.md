# PyClee

This is a Python implementation of the **DyClee** (Dynamic clustering for tracking evolving environments) algorithm proposed by Nathalie Barbosa Roa, Louise Travé-Massuyès and Victor H. Grisales-Palacio in https://doi.org/10.1016/j.patcog.2019.05.024.

## Differences compared to the paper
- The definition of direct connectedness given in Equation 5 involves a factor of `1/2` which is inconsistent with the stated condition that the hyperboxes of two given microclusters must overlap to be directly connected. An overlap would mean `max_i|c_ij - c_ik| < S_i for all i`, so this condition was used instead. This was confirmed by the authors to be correct.
- In the paper, microclusters may not skip a step when moving between the density partitions (e.g., dense microclusters may not become outliers, and outliers may not become dense). I believe that in my implementation this would create a risk of trapping and then dropping microclusters, especially in cases where the density-based stage is not applied for every sample, so in this implementation these restrictions are lifted. This will be reconsidered soon.
- In this implementation, the density partitioning step is implemented as a separate step between the distance stage and the density stage, and its frequency is controlled with a separate setting (`partitioning_interval`).
- The paper favoured using a KDTree for spatial indexing. I found this to be too slow as the tree has to be rebuilt from scratch every time a microcluster is added or changed. I've made the R*Tree structure from the `rtree` package the default, as it can be updated with new elements so only has to be created once. This resulted in better performance. (Note: I may not be using the KDTree optimally, this could be looked into.)

Note also that many variable names have been changed to conform to a consistent style.

## Examples

See [`Examples.ipynb`](./Examples.ipynb) for more details.

### Static

```python
# See Examples.ipynb for imports etc.
context = DyCleeContext(2, 0.06, bounds, store_elements=True)
dy = DyClee(context)
clusters = dy.run(X)

MultiPlotter(dy, legend_loc='upper left').plot_snapshot(clusters)
```

![blobs](https://user-images.githubusercontent.com/1812261/105618503-cb9baf00-5ddf-11eb-9fff-2f1adc67d681.png)

### Dynamic ("concept drift")

```python
# See Examples.ipynb for imports etc.
context = DyCleeContext(2, 0.06, bounds, forgetting_method=ExponentialForgettingMethod(0.01))
dy = DyClee(context)

fig, ax = plt.subplots()
MultiPlotter(dy, ax).animate(X)  # Very slow
```

![drift-16x](https://user-images.githubusercontent.com/1812261/105618363-28966580-5dde-11eb-8b7a-162e9715bf38.gif)


## TODO
- Local-density approach
- Outlier rejection ("`Unclass_accepted`" in the paper)
- Sparse rejection ("`minimum_mc`")
- Maintain at least two separate R*Trees (for dense/semi-dense microclusters and for outliers) for performance
- pip package
- Tests
- Benchmarks

## Contributing
Discussions in issues and pull requests are welcome, in particular in regards to the notes above.

This project uses a [modified version of the Black code style](https://github.com/harenbrs/bleck).