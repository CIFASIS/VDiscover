
# VDiscovery

VDiscovery is a dataset collected to predict memory corruption vulnerabilities in binary only programs. It was created by analyzing 1039 testcases taken from the Debian Bug Tracker. Every testcase uses a different executable program and they are distributed over 496 packages. They were originally packed along their inputs by the Mayhem team using their symbolic execution tool and submitted to the Debian Bug Tracker. The programs comprised in VDiscovery are quite diverse and included data processing tools from scientific packages, simple games, small desktop programs and even an OCR, among others. This branch contains 
two scripts to download, balance and split this dataset (dynamic_split.py and static_split.py)
Information about features and classes detailed in our technical report:

[Toward large-scale vulnerability discovery using Machine Learning](http://www.vdiscover.org/report.pdf)

To start using this dataset:

    git clone https://github.com/CIFASIS/VDiscover.git -b vdiscovery

It should not be confused with [VDiscover](https://github.com/CIFASIS/VDiscover), our tool for vulnerability discovery. More information is available [here](http://vdiscover.org). 

