# mbdw2020データセットにベクトルモデル（TFIDF）を適用する。
# SVMを使って、船舶mmsiを判別する。
import pickle
import numpy as np
import datetime
import os
# SVM法による分類
from sklearn import svm
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
# TF-IDFを計算
from sklearn.feature_extraction.text import TfidfVectorizer
#sckitlearnよりCountVectorizerをインポートする
from sklearn.feature_extraction.text import CountVectorizer

# read pickle data file
def readdocsdata(codedocsbymmsitimestamp, onesailkeys):
    docsdata=[]
    for mmsitimestamp in onesailkeys:
        codedoc = codedocsbymmsitimestamp[mmsitimestamp]
        codedocstr = [str(code) for code in codedoc]
        codedocline = " ".join(codedocstr) # for sklearn TFIDF
        docsdata.append(codedocline)
    print( "the number of data=", len(docsdata) )

    return docsdata

def calculatetfidf(documents):
    # オブジェクト生成
    npdocs=np.array(documents)
    #vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, norm='l2', sublinear_tf=True, smooth_idf=True)
    #vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,3), norm='l2', sublinear_tf=True, smooth_idf=True)
    # max_dfは0.5（半分以上の文書に出現する言葉はいらない）を設定
    vecs = vectorizer.fit_transform(npdocs)
    # 単語帳を表示
    terms = vectorizer.get_feature_names_out()
    #for t in terms[0:100]:
    #    print("term =", t)
    # TF-IDFを計算
    tfidfs = vecs.toarray()

    return tfidfs, terms

# 保存済みのデータを読み込む
def readdocslabel(minlength):
    print("辞書とコーパスをロード...")
    print("newonesailkeysを読み込む...")
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
    print(" the number of documents =", len(onesailkeys))
    docslabels=[]
    for timestampmmsi in onesailkeys:
        mmsi, timestamp =timestampmmsi.split("-")
        docslabels.append(mmsi)
    labelname = list(set(docslabels))
    labelname = sorted(labelname)
    print("labelname =", labelname)
    docslabel=np.empty(len(docslabels), dtype=int)
    for i, data in enumerate(docslabels):
        docslabel[i] = labelname.index(data)
    print("docslabel =", docslabel)
    print("total ships =", len(labelname), "total onesails =", sum(docslabel))

    return docslabel, labelname

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
    print("newonesailkeysを読み込む...")
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
    #codedocsfile = "./codedocs/codedocsbytimestampmmsi4096x500.pickle"
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
    documents = readdocsdata(codedocsbymmsitimestamp, onesailkeys)
    #for docno, doc in enumerate(documents):
    #    print("docno =", docno, "lendoc =", len(doc))
    # TFIDF
    tfidfs,terms = calculatetfidf(documents)
    # Label
    docslabel, labelname = readdocslabel(minlength) 
    data, truelabel = shuffle(tfidfs, docslabel)
    print("shape of original data =", data.shape)
    print("shape of original truelabel =", truelabel.shape)

    # SVMモデルを作成
    # ペナルティパラメータ
    C = 20
    # RBFカーネルのパラメータ
    gamma = 0.08
    model = svm.SVC(C=C)
    #　データを分類
    # 層化 kk 分割交差検証
    kk = 4
    skf = StratifiedKFold(n_splits=kk, shuffle=True, random_state=0)
    #交差検証
    print("Stratified K-Fold Cross Validation:")
    nn = 0
    training_accuracy = []
    test_accuracy = []
    for train, test in skf.split(data, truelabel):
        print("nn =", nn)
        # データを用意する
        data_train=data[train]
        label_train=truelabel[train]
        # テストデータはそのまま
        data_test=data[test]
        label_test=truelabel[test]
        # モデルの訓練
        model.fit(data_train, label_train)
        # 訓練データによる分類結果の評価
        """
        predlabel_train = model.predict(data_train)
        print("results by training data:")
        for true, pred in zip(label_train, predlabel_train):
            if(pred==true):
                print("true label =", true, "\t predicted label =", pred)
            else:
                print("true label =", true, "\t predicted label =", pred, "\t not agreed")
        training_acc =  metrics.accuracy_score(label_train, predlabel_train)
        print("training accuracy =", training_acc)
        training_accuracy.append(training_acc)
        print(metrics.classification_report(label_train, predlabel_train, target_names=labelname))
        """
        # テストデータによる分類結果の評価
        predlabel_test = model.predict(data_test)
        print("results by test data:")
        """
        for true, pred in zip(label_test, predlabel_test):
            if(pred==true):
                print("true label =", true, "\t predicted label =", pred)
            else:
                print("true label =", true, "\t predicted label =", pred, "\t not agreed")
        """
        test_acc =  metrics.accuracy_score(label_test, predlabel_test)
        print("test accuracy =", test_acc)
        test_accuracy.append(test_acc)
        print(metrics.classification_report(label_test, predlabel_test, labels= np.arange(len(labelname)), target_names=labelname))
        print("-------------------------------------")
        nn += 1
    #print("average of training accuracy =", np.mean(training_accuracy))
    print("average of test accuracy =", np.mean(test_accuracy))
    print("-------------------------------------")
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