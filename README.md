# PyClee

This is a Python implementation of the **DyClee** (Dynamic clustering for tracking evolving environments) algorithm proposed by Nathalie Barbosa Roa, Louise Travé-Massuyès and Victor H. Grisales-Palacio in https://doi.org/10.1016/j.patcog.2019.05.024.

## Differences compared to the paper
- Density stage: this algorithm was reworked almost entirely based on the text (and on performance considerations) rather than the given Algorithm 2, which seemed to be inconsistent with the text in these ways:
    - Algorithm 2 seems to make no distinction between dense and semi-dense microclusters, which, according to the text, is important in defining the interior vs. the boundary of a cluster
    - Algorithm 2 assigns the `cid` label to any-density second neighbours, but only to dense first neighbours.
- The definition of direct connectedness given in Equation 5 involves a factor of `1/2` which is inconsistent with the stated condition that the hyperboxes of two given microclusters must overlap to be directly connected. An overlap would mean `max_i|c_ij - c_ik| < S_i for all i`, so this condition was used instead.
- After each density-based step, cluster labels are removed from microclusters that have not been made part of a cluster during the step. This was not mentioned explicitly in the paper, but the results had a lot of spurious clusters without this change.
- I chose the definition of outlier microclusters to include `density <= median(densities)` (note the closed interval). This is to deal with the fact that in many situations with a lot of widespread noise `min(densities) == median(densities)`, and with the paper's definition this would result in exactly 0 outlier microclusters.
- The paper favoured using a KDTree for spatial indexing. I found this to be too slow as the tree has to be rebuilt from scratch every time a microcluster is added or changed. I've made the R*Tree structure from the `rtree` package the default, as it can be updated with new elements so only has to be created once. This resulted in better performance. (Note: I may not be using the KDTree optimally, this could be looked into.)

I'm looking for input from the original authors to clarify these points, but for now these changes represent my best effort to get sensible results.

Note also that many variable names have been changed to conform to a consistent style.

## TODO
- Local-density approach
- Microcluster elimination (needs clarification about the low density threshold)
- Long-term memory
- Outlier rejection ("`Unclass_accepted`" in the paper)
- Sparse rejection ("`minimum_mc`")
- pip package
- Examples
- Tests
- Benchmarks

## Contributing
Discussions in issues and pull requests are welcome, in particular in regards to the notes above.

This project uses a [modified version of the Black code style](https://github.com/harenbrs/bleck).