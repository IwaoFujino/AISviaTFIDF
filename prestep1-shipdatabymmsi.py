# AISデータのクレンジング
# 1. clean data record
# 2. seperate data record by mmsi
# 2025.01.03
# use all data include speed=0 or stoped for clasification

import codecs
import csv
import datetime
import numpy as np
import pickle

# main function
def main():
    # 緯度経度で範囲を指定する
    # smaller coverage
    #west=-7.00 #　経度（西（左））
    #east=-2.00 #　経度（東（右））
    #south=46.50 #　緯度（南（下））　
    #north=49.50 #　緯度（北（上））
    # brest bay details coverage
    west=-6.00 #　経度（西（左））
    east=-4.00 #　経度（東（右））
    south=47.50 #　緯度（南（下））　
    north=49.00 #　緯度（北（上））

    # 設定条件を満たすAISデータレコードを取り出す
    aisdatadir = "./aisdata"
    fcsv=codecs.open(aisdatadir + "/nari_dynamic.csv", encoding="utf-8")
    csv_reader = csv.reader(fcsv)
    csvdata=[]
    for rowno, rowdata in enumerate(csv_reader):
        # for check program
        #if rowno==500000:
        #   break
        if rowno%1000000==0:
            print( "row no. =", rowno )
        if rowno!=0:
            if int(rowdata[5])!=511: # trueheadingは「利用不可」ではない
                if float(rowdata[6])>=west and float(rowdata[6])<=east and float(rowdata[7])>=south and float(rowdata[7])<=north:
                    csvdata.append(rowdata)
    print("the number of all data records=", rowno)
    print( "the number of picked up data records=", len(csvdata) )

    shipdatabymmsi={}
    for data in csvdata:
        mmsi=data[0]
        onedata0=int(data[8]) # timestamp
        onedata1=float(data[6]) #lon
        onedata2=float(data[7]) #lat
        onedata3=(90.0-float(data[5]))*np.pi/180.0 # trueheading
        onedata4=float(data[3]) # speedoverground
        if mmsi not in shipdatabymmsi.keys():
            shipdatabymmsi[mmsi]=[[onedata0,onedata1,onedata2,onedata3,onedata4]]
        else:
            shipdatabymmsi[mmsi].append([onedata0,onedata1,onedata2,onedata3,onedata4])

    with open('./pickledata/shipdatabymmsi.pickle', mode='wb') as fo1:
        pickle.dump(shipdatabymmsi, fo1)
    print( "the number of ships =", len(shipdatabymmsi) )
    datacntall=0
    for mmsi, oneshipdata in sorted(shipdatabymmsi.items(), key=lambda x:x[0]):
        datacntall+=len(oneshipdata)
    print( "the number of total data in shipdatabymmsi =", datacntall )

    return

# run frpm here
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
