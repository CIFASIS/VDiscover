"""
This file is part of VDISCOVER.

VDISCOVER is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

VDISCOVER is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with VDISCOVER. If not, see <http://www.gnu.org/licenses/>.

Copyright 2014 by G.Grieco
"""

import random
import copy


def cluster_sampler(clustered_traces, n_per_cluster):
    #cc = copy.copy(clusters)
    # n_per_cluster = 1#n / len(cc)
    clusters = dict()
    for label, cluster in clustered_traces:
        clusters[cluster] = clusters.get(cluster, []) + [label.split(":")[-1]]

    selected = set()
    tmp = set()

    for (cluster, seeds) in clusters.items():
        n_sample = min(len(seeds), n_per_cluster)
        tmp = set(seeds).intersection(selected)
        if len(tmp) >= n_sample:
            selected.update(set(random.sample(tmp, n_sample)))
        else:
            selected.update(set(random.sample(seeds, n_sample)))

    return selected
