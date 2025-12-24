# mbdw2020データセットにベクトルモデル（TFIDF）を適用する。

import pickle
import numpy as np
import datetime
import os
from sklearn.utils import shuffle
from matplotlib import pyplot as plt 
# TF-IDFを計算
from sklearn.feature_extraction.text import TfidfVectorizer
# Clutering
import scipy.cluster.hierarchy as hclst

# read pickle data file
def codedocsbymmsi(codedocsbymmsitimestamp, onesailkeys):
    codedocsbymmsi={}
    for mmsitimestamp in onesailkeys:
        codedoc = codedocsbymmsitimestamp[mmsitimestamp]
        codedoclist = [str(code) for code in codedoc]
        mmsi, timestamp = mmsitimestamp.split("-")
        if mmsi not in codedocsbymmsi.keys():
            codedocsbymmsi[mmsi] = []
        else:
            codedocsbymmsi[mmsi].extend(codedoclist)

    docsdata=[]
    docslabels=[]
    for mmsi in sorted(codedocsbymmsi.keys()):
        docslabels.append(mmsi)
        codedocline = " ".join(codedocsbymmsi[mmsi]) # for sklearn TFIDF
        print(">>>>", mmsi)
        #print(codedocsbymmsi[mmsi])
        #print(codedocline)
        docsdata.append(codedocline)
    print( "the number of data=", len(docsdata) )
    print("docslabels =", docslabels)
    print("total ships =", len(docslabels))

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
    nn=8 #nn=8, 12, 16, 20
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
    documents, docslabels = codedocsbymmsi(codedocsbymmsitimestamp, onesailkeys)
    #for docno, doc in enumerate(documents):
    #    print("docno =", docno, "lendoc =", len(doc))
    # TFIDF
    tfidfs,terms = calculatetfidf(documents)
    data, truelabels = shuffle(tfidfs, docslabels)
    print("shape of original data =", data.shape)
    print("shape of original truelabels =", len(truelabels))

    # Clustering
    methodstr = "single" # single, complete, 'average', ward, If linkage is “ward”, only “euclidean” is accepted
    metricstr = "cosine" # euclidean, jensenshannon, cosine
    # クラスタリングは以下の 1 行だけ。
    results = hclst.linkage(data, method=methodstr,metric=metricstr)

    # デンドログラムで結果を表示
    fig = plt.figure(figsize=(8.0,12.0))
    hclst.dendrogram(results, labels=truelabels, orientation='right', distance_sort='descending')
#    # 各ラベルに色を割り当て
#    colorname=['b', 'g', 'c', 'm', 'y']
#    label_colors =dict(zip(truelabels, colorname))
#    ax = plt.gca()
#    xlbls = ax.get_xmajorticklabels()
#    for lbl in xlbls:
#        lbl.set_color(label_colors[lbl.get_text()])
    #plt.tight_layout()
    plt.ylabel("MMSI", fontsize=16)
    plt.xlabel("1 - cosine similarity", fontsize=16)
    plt.suptitle(f"Dendrogram of Clustering Vessels by their Trajectories (nn={nn})", fontsize=18, fontname="serif", y=0.95)
    plt.xlim((0, 1))
    plt.title("method="+methodstr+"  metric="+metricstr, fontsize=18, fontname='serif')
    pngfile = "./picturedata/dendrogram-shiptfidf-dataset40-nn"+str(nn)+"-"+methodstr+"-"+metricstr+".png"
    plt.savefig(pngfile, dpi=720, facecolor='white', bbox_inches='tight', pad_inches=0.1)
    print("Saved dendrogram to", pngfile)
    #plt.show()
    plt.clf()
    plt.close()

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