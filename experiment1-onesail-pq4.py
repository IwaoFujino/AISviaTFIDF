# ベクトル量子化(PQKmeans)
#

import datetime
import numpy as np
from os import path
import pqkmeans
import pickle
import codecs

# make codebook
def makepqcodebook(onesailbymmsitimestamp, onesailkeys, codebooklen, minlength, mm):
    # merge all onesail data together
    no=0
    for mmsitimestamp in onesailkeys:
        onesaildata = onesailbymmsitimestamp[mmsitimestamp]
        #print("no=",no)
        if no==0:
            onesaildataall = onesaildata
            #print(onesaildata)
            #print(onesaildataall)
        else:
            onesaildataall = np.vstack((onesaildataall, onesaildata))
            #print(onesaildata)
            #print(onesaildataall)
        no+=1

    # PQコード作成
    # データをクラスタリング
    print( "start making codebook ..." )
    print("positionalldata.shape:\n{}\n".format(onesaildataall.shape))
    #print("positionalldata:\n{}".format(onesaildataall))

    encoder = pqkmeans.encoder.PQEncoder(num_subdim=mm, Ks=256)
    encoder.fit(onesaildataall)
    onesaildataall_pqcode = encoder.transform(onesaildataall)
    kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder, k=codebooklen) # k=2000 OK, k=4096 failed Kはデータ数より少ないとき、終了となる。エラー表示もなし
    kmeans.fit_predict(onesaildataall_pqcode)

    # get code for each sail
    codedocsbymmsitimestamp={}
    ll=0
    for mmsitimestamp in onesailkeys:
        onesaildata = onesailbymmsitimestamp[mmsitimestamp]
        #print("ll=", ll)
        #print("onesaildata length =", len(onesaildata))
        #print(onesaildata[0:10])
        onesaildata_pqcode = encoder.transform(onesaildata)
        onesaildatacode = kmeans.predict(onesaildata_pqcode)
        #print("onesaildatacode length =", len(onesaildatacode))
        #print(onesaildatacode[0:10])
        codedocsbymmsitimestamp[mmsitimestamp]=onesaildatacode
        ll += 1

    return encoder, kmeans, codedocsbymmsitimestamp

# reconstruct　code
def reconstructcode(encoder, kmeans, codebooklen, minlength ):
    # Convert to np.array with the proper dtype
    clustering_centers_numpy = np.array(kmeans.cluster_centers_, dtype=encoder.code_dtype)
    # From PQ-code to vectors
    clustering_centers_codebook = encoder.inverse_transform(clustering_centers_numpy)

    return clustering_centers_codebook

# main procedure
def main():
    # parameter
    minlength=500 # minimum length of tracjectory data (set to 2000, 2024/06/30)
    mm=4
    nn=8 # nn=4, 8, 12, 16, 20, 24
    codebooklen=nn*nn*nn*nn
    print( "nn=", nn )
    print( "codebooklen=", codebooklen )
    # read file
    keysfile="./pickledata/newonesailkeys"+str(minlength)+".pickle"
    # prepare keys
    if path.exists(keysfile):
        with open(keysfile, mode="rb") as fin1:
            print( keysfile + " EXIST !!!" )
            onesailkeys=pickle.load(fin1)
            print("the number of onesail keys = ", len(onesailkeys))
    else:
        print( keysfile + " NOT Exist !!!" )
        return
    datafile="./pickledata/onesailbymmsitimestamp"+str(minlength)+".pickle"
    # prepare data
    if path.exists(datafile):
        with open(datafile, mode="rb") as fin2:
            print( datafile + " EXIST !!!" )
            onesailbymmsitimestamp=pickle.load(fin2)
        print("the number of onesail data = ", len(onesailbymmsitimestamp))
    else:
        print( datafile + " NOT Exist !!!" )
        return

    # make codebook
    encoder, kmeans, codedocsbymmsitimestamp = makepqcodebook(onesailbymmsitimestamp, onesailkeys, codebooklen, minlength, mm)
    # write out code docs
    print( "len(codedocsbymmsitimestamp) =",len(codedocsbymmsitimestamp) )
    codedocsfile="./codedocs/codedocsbymmsitimestamp"+str(codebooklen)+"x"+str(minlength)+".pickle"
    with open(codedocsfile, mode="wb") as fout:
        pickle.dump(codedocsbymmsitimestamp, fout)

    clustering_centers_codebook = reconstructcode(encoder, kmeans, codebooklen, minlength)
    print( "length of real codebook =", len(clustering_centers_codebook) )
    #write out clustering_centers_codebook
    codebookfile = "./codebook/clustering_centers_codebook"+str(codebooklen)+"x"+str(minlength)+".npy"
    np.save(codebookfile, clustering_centers_codebook)
    # write out codebook to text file
    codetextfile="./codebook/codebook"+str(codebooklen)+"x"+str(minlength)+".txt"
    fwcodebook=codecs.open(codetextfile, "w", encoding="utf-8")
    for code, center in enumerate(clustering_centers_codebook):
        fwcodebook.write(str(code)+":"+str(center[0])+","+str(center[1])+","+str(center[2])+","+str(center[3])+"\n")
    fwcodebook.close()

    print( "nn=", nn )
    print( "codebooklen=", codebooklen )

    return

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


