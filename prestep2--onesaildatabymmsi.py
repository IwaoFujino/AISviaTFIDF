# １回の航行データに振り分ける
# １時間以上の間隔があけられたら、新しい航行とする
# ６時間以上の間隔があけられたら、新しい航行とする（>=6*3600 modified 2024.03.16）

import numpy as np
import datetime
import pickle
from os import path

# version x2: if length of onesaildata < minlength then ignore 

def main():
    # main precedure
    # prepare data
    datafile="./pickledata/shipdatabymmsi.pickle"
    if path.exists(datafile):
        print( datafile+" EXIST !!!" )
        with open(datafile, mode="rb") as fin:
            shipdatabymmsi=pickle.load(fin)
    else:
        print( datafile+" NOT Exist !!!" )
        return

    print( "the number of ships in data =", len(shipdatabymmsi) )
    datacntall=0
    for mmsi, oneshipdata in sorted(shipdatabymmsi.items(), key=lambda x:x[0]):
        #print( "mmsi=", mmsi, "------------------------------------" )
        #for onesaildata in oneshipdata:
            #print( "len(onesaildata)=", len(onesaildata) )
        datacntall+=len(oneshipdata)
    print( "the number of total data in shipdatabymmsi =", datacntall )

    # if onesaildata has more than minlength data then save to onesaildatabymmsi, otherwise it will be ignored
    minlength=500 # minimum length of tracjectory data (set to 2000, 2024/06/30)
    onesailcnt=0
    onesaildatabymmsi={}
    for mmsi, oneshipdata in sorted(shipdatabymmsi.items(), key=lambda x:x[0]):
        oldtimestamp=0
        onesaildata=[]
        onesailflag=0
        for data in sorted(oneshipdata, key=lambda x: x[0]):
            timestamp=data[0]
            deltatime=timestamp-oldtimestamp
            # if time interval >= 1 hour (3600 sec) begin a new sail
            # >=6*3600 modified 2024.03.16
            if oldtimestamp != 0 and deltatime >= 6*3600:
                #print( len(onesaildata), deltatime )
                # if onesaildata has more than minlength data then save to onesaildatabymmsi
                if len(onesaildata)>=minlength:
                    if mmsi not in onesaildatabymmsi.keys():
                        onesaildatabymmsi[mmsi]=[onesaildata]
                    else:
                        onesaildatabymmsi[mmsi].append(onesaildata)
                    onesailcnt+=1
                    onesailflag=1
                # reset onesaildata
                onesaildata=[]
            onesaildata.append(data)
            oldtimestamp=timestamp

        if len(onesaildata)>=minlength:
            if onesailflag==0:
                onesaildatabymmsi[mmsi]=[onesaildata]
            else:
                onesaildatabymmsi[mmsi].append(onesaildata)
            onesailcnt+=1
    print("the number of ships in onesaildatabymmsi =", len(onesaildatabymmsi) )
    print("total ships =", len(onesaildatabymmsi), "total onesails =", onesailcnt)
    with open("./pickledata/onesaildatabymmsi"+str(minlength)+".pickle", mode="wb") as fo2:
        pickle.dump(onesaildatabymmsi, fo2)

    datacntall=0
    shipcount={}
    for mmsi, oneshipdata in sorted(onesaildatabymmsi.items(), key=lambda x:x[0]):
        oneshipcount=[]
        for onesaildata in oneshipdata:
            datacntall+=len(onesaildata)
            oneshipcount.append(len(onesaildata))
        shipcount[mmsi] = oneshipcount
    print( "the number of total data record in onesaildatabymmsi =", datacntall )
    print( "the number of onesail in onesaildatabymmsi =", onesailcnt )
    for mmsi, oneshipcount in sorted(shipcount.items(), key=lambda x:x[1], reverse=True):
        print("mmsi=", mmsi, "count=", oneshipcount)
    with open("./pickledata/shipcount"+str(minlength)+".pickle", mode="wb") as fout:
        pickle.dump(shipcount, fout)

    # read pickle data file 
    # make onesail-key file
    alldatalen=0
    onesailbymmsitimestamp={}
    onesailkeys=[]
    for mmsi, mmsidata in sorted(onesaildatabymmsi.items(), key=lambda x:x[0]):
        for onesail in mmsidata:
            timestamp0=onesail[0][0]
            # fomer vq version x4: combine timestamp and mmsi as key
            mmsitimestamp=mmsi + "-" + str(timestamp0)
            #print( "mmsitimestamp=", mmsitimestamp )
            onesailkeys.append(mmsitimestamp)
            onesaildata=[]
            for onedata in onesail:
                onesaildata.append(onedata[1:5])
            onesailbymmsitimestamp[mmsitimestamp]=np.array(onesaildata)
            alldatalen+=len(onesaildata)

    print( "len(onesailbymmsitimestamp) =",len(onesailbymmsitimestamp) )
    print( "length of all data =", alldatalen )

    #for key in onesailkeys[0:100]:
    #    print(key)
    #print("------------------------")
    sortedonesailkeys = sorted(onesailkeys)
    #for key in sortedonesailkeys[0:100]:
    #    print(key)
    #print("------------------------")
    keysfile="./pickledata/onesailkeys"+str(minlength)+".pickle"
    with open(keysfile, mode="wb") as fout1:
        pickle.dump(sortedonesailkeys, fout1)
    print("Saved pickle file " + keysfile +".")
    datafile="./pickledata/onesailbymmsitimestamp"+str(minlength)+".pickle"
    with open(datafile, mode="wb") as fout2:
        pickle.dump(onesailbymmsitimestamp, fout2)
    print("Saved pickle file " + datafile +".")

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
