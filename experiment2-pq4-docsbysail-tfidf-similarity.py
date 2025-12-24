# mbdw2020データセットにベクトルモデル（TFIDF）を適用する。

import pickle
import numpy as np
import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity
# TF-IDFを計算
from sklearn.feature_extraction.text import TfidfVectorizer

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
        #print(">>>>", mmsitimestamp)
        #print(codedocsbysail[mmsitimestamp])
        #print(codedocline)
        docsdata.append(codedocline)
    #print( "the number of data=", len(docsdata) )
    #print("docslabels =", docslabels)
    print("total labels =", len(docslabels))

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
    # コサイン類似度
    similarity = cosine_similarity(tfidfs)
    # 計算結果を表示
    print("cosine similarity:")
    for n1, simi in enumerate(similarity):
        print(f"idx{n1}    {docslabels[n1]}", end="")
        index = np.argsort(-simi)
        for n2 in index[1:4]:
            print(f"    idx{n2}    {docslabels[n2]}    {simi[n2]:0.4f}\t", end="")
        print()

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