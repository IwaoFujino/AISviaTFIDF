# AISviaTFIDF

Data and program source code of TFIDF model to process AIS data.

Here is a list of all files.
1) prestep1-shipdatabymmsi.py: Seperate data by MMSI of ships.
2) prestep2--onesaildatabymmsi.py: Segment ship's data to sails and generate an index key for each sail.
3) prestep3--onesaildatabymmsi-sailcountbyship.py: Count the sails of each ship and generate dataset where its count >= an assigned number.
4) prestep4-onesaildatabymmsi-alltrajectory.py: Draw all trajectory in the dataset.
5) experiment1-onesail-pq4.py: Vector quantization with PQk-means algorithm and generate code documents for all sails in the dataset.
6) experiment2-pq4-docsbysail-tfidf-similarity.py: Calculate similarity between sails in the dataset.
7) experiment3-pq4-docsbyship-tfidf-similarity.py: Calculate similarity between ships in the dataset.
8) experiment4-pq4-docsbysail-tfidf-kmeans.py: Clustering of sails and draw a dendrogram of its result.
9) experiment5-pq4-docsbyship-tfidf-dendrogram.py: Clustering of ships and draw a dendrogram of its result.
10) experiment6-pq4-docsbysail-tfidf-classification.py: Classification of sails and report of evaluation result.

The files and folders are deployed as follows:
tfidfmodel
|
|---aisdata---nari_dynamic.csv <--Download this data file from https://doi.org/10.5281/zenodo.1167595
|---codebook
|---codedocs
|---csvdata
|---onesailtrajectory
|---pickledata
|---picturedata
|---prestep1-shipdatabymmsi.py
|---prestep2--onesaildatabymmsi.py
|---prestep3--onesaildatabymmsi-sailcountbyship.py
|---prestep4-onesaildatabymmsi-alltrajectory.py
|---experiment1-onesail-pq4.py
|---experiment2-pq4-docsbysail-tfidf-similarity.py
|---experiment3-pq4-docsbyship-tfidf-similarity.py
|---experiment4-pq4-docsbysail-tfidf-kmeans.py
|---experiment5-pq4-docsbyship-tfidf-dendrogram.py
|---experiment6-pq4-docsbysail-tfidf-classification.py
