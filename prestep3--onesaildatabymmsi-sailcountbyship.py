# 全部のOnesailを地図に表示して、PNGファイルに保存する
import datetime
import pickle
import os


def main():
    minlength=500 # minimum length of tracjectory data (set to 2000, 2024/06/30)
    # read keysfile
    keysfile="./pickledata/onesailkeys"+str(minlength)+".pickle"
    # prepare keys
    if os.path.exists(keysfile):
        with open(keysfile, mode="rb") as fin1:
            print( keysfile + " EXIST !!!" )
            onesailkeys=pickle.load(fin1)
            print("the number of onesail keys = ", len(onesailkeys))
    else:
        print( keysfile + " NOT Exist !!!" )
        return

    # 船別にonesailをカウントする
    sailcountbyship = {}
    for mmsitimestamp in onesailkeys:
        mmsi, timestamp = mmsitimestamp.split("-")
        sailcountbyship[mmsi]= sailcountbyship.get(mmsi, 0) + 1

    for mmsi, oneshipcount in sorted(sailcountbyship.items(), key=lambda x:x[1], reverse=True):
        print("mmsi=", mmsi, "count=", oneshipcount)

    countfile="./pickledata/sailcountbyship"+str(minlength)+".pickle"
    with open(countfile, mode="wb") as fout1:
        pickle.dump(sailcountbyship, fout1)

    # 船別でonesailがonesailcount_min回以上のmmsitimestampをnewonekeysに移す
    onesailcount_min = 40
    newonesailmmsi = []
    newonesailkeys = []
    for mmsitimestamp in onesailkeys:
        mmsi, timestamp = mmsitimestamp.split("-")
        if sailcountbyship[mmsi] >= onesailcount_min:
            newonesailkeys.append(mmsitimestamp)
            newonesailmmsi.append(mmsi)
            
    newkeysfile="./pickledata/newonesailkeys"+str(minlength)+".pickle"
    with open(newkeysfile, mode="wb") as fout2:
        pickle.dump(newonesailkeys, fout2)
    onesailcount_min
    print("dataset condition: minimum count of onesail =", onesailcount_min)
    print("number of onesail in onesailkeys =", len(onesailkeys))
    print("number of ship in newonesailkeys =", len(list(set(newonesailmmsi))))
    print("number of onesail in newonesailkeys =", len(newonesailkeys))

    return

# run from here
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
