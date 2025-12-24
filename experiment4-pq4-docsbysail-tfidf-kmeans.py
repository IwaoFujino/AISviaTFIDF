# mbdw2020データセットにベクトルモデル（TFIDF）を適用する。

import pickle
import numpy as np
import datetime
import os
import csv
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
# TF-IDFを計算
from sklearn.feature_extraction.text import TfidfVectorizer
# Clutering
from sklearn import cluster

# read pickle data file
def codedocsbysail(codedocsbymmsitimestamp, onesailkeys):
    codedocsbysail={}
    for mmsitimestamp in onesailkeys:
        codedoc = codedocsbymmsitimestamp[mmsitimestamp]
        codedoclist = [str(code) for code in codedoc]
        codedocsbysail[mmsitimestamp] = codedoclist

    docsdata=[]
    docslabels=[]
    for mmsitimestamp in sorted(codedocsbysail.keys()):
        docslabels.append(mmsitimestamp)
        codedocline = " ".join(codedocsbysail[mmsitimestamp]) # for sklearn TFIDF
        print(">>>>", mmsitimestamp)
        #print(codedocsbysail[mmsitimestamp])
        #print(codedocline)
        docsdata.append(codedocline)
    print( "the number of data=", len(docsdata) )
    print("the number of docslabels =", len(docslabels))
    print("total sails =", len(docslabels))

    return docsdata, docslabels

def calculatetfidf(documents):
    # オブジェクト生成
    npdocs=np.array(documents)
    #vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, norm='l2', sublinear_tf=True, smooth_idf=True)
    vecs = vectorizer.fit_transform(npdocs)
    # 単語帳を表示
    terms = vectorizer.get_feature_names_out()
    # TF-IDFを計算
    tfidfs = vecs.toarray()

    return tfidfs, terms


#メイン関数
def main():
    # パラメータ
    nn=8
    codebooklen=nn*nn*nn*nn
    minlength = 500
    print("nn=", nn)
    print("codebooklen=", codebooklen )
    # データセットを読み込む
    print( "check point 0 ..." )
    # read key file
    print("onesailkeysを読み込む...")
    keysfile="./pickledata/newonesailkeys"+str(minlength)+".pickle"
    # prepare keys
    if os.path.exists(keysfile):
        with open(keysfile, mode="rb") as fin1:
            print( keysfile + " EXIST !!!" )
            onesailkeys=pickle.load(fin1)
            print("the number of onesail keys = ", len(onesailkeys))
    else:
        print( keysfile + " NOT Exist !!!" )
        return
    print("コードの文書集合を読み込む...")
    codedocsfile = "./codedocs/codedocsbymmsitimestamp"+str(codebooklen)+"x"+str(minlength)+".pickle"
    if os.path.exists(codedocsfile):
        with open(codedocsfile, mode="rb") as fin2:
            print( codedocsfile + " EXIST !!!" )
            codedocsbymmsitimestamp=pickle.load(fin2)
            print("the number of code documents = ", len(codedocsbymmsitimestamp))
    else:
        print( codedocsfile + " NOT Exist !!!" )
        return
    # prepare data and label
    documents, docslabels = codedocsbysail(codedocsbymmsitimestamp, onesailkeys)
    #for docno, doc in enumerate(documents):
    #    print("docno =", docno, "lendoc =", len(doc))
    # TFIDF
    tfidfs,terms = calculatetfidf(documents)
    data, truelabels = shuffle(tfidfs, docslabels)
    print("shape of original data =", data.shape)
    print("shape of original truelabels =", len(truelabels))


    # Clustering
    model=cluster.KMeans(n_clusters=6)
    model.fit(data)
    labels=model.labels_
    print("Labels=", labels)
    results=[]
    for ll in range(6):
        labelsindex=np.where(labels==ll)
        print("Index of label "+str(ll)+ "=", labelsindex[0])
        for i in labelsindex[0]:
            print("index =", i, end="")
            print("\tmmsi-timestamp =", truelabels[i], end="")
            index=onesailkeys.index(truelabels[i])
            print("\tindex=", index)
            results.append([ll, i, truelabels[i], index])

    with open('./csvdata/voyage-clustering-kmeans.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)

    return

# start from here
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time=end_time-start_time
	print( "start time:" )
	print( start_time.strftime("%Y/%m/%d  %H/%M/%S.") + "%04d" % (start_time.microsecond // 1000) )
	print( "end time:" )
	print( end_time.strftime("%Y/%m/%d  %H/%M/%S.") + "%04d" % (end_time.microsecond // 1000) )
	print( "elapsed_time" )
	print( elapsed_time )
	print( "ALL END !!!" )