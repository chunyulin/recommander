* Given [ip, time, resource], this code translate it into a mapping table [ip, hid], [resource, rid] and its inverse.
* Then, [hid, rid, 1] tuple can be built and trained to get $M=HR$ by the FunkSVD algorithm.

* $M_{ij}$ represents the user preference of each item. Once a new IP (user) click a dataset, the top-N dataset can be recommended thereafter.
