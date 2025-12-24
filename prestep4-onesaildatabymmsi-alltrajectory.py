# 全部のOnesailを地図に表示して、PNGファイルに保存する
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#from os import path
import pickle
import os

# brest bay details coverage
west=-6.00 #　経度（西（左））
east=-4.00 #　経度（東（右））
south=47.50 #　緯度（南（下））　
north=49.00 #　緯度（北（上））
lonscale=0.50
latscale=0.25

# draw ship's trajectory (longitude, latitude and heading, speed) on map
def drawonesaildata(outputdir, mmsi, timestamp, fileidx, onesaildata):
	saildatetime=datetime.datetime.fromtimestamp(int(timestamp))
	print("fileidx =", fileidx, "mmsi=", mmsi, "timestamp=",timestamp, "datetime=", saildatetime )
	lons=onesaildata[0,:]
	lats=onesaildata[1,:]
	headings=onesaildata[2,:]
	speeds=onesaildata[3,:]
	r0=5.0e-3
	u10=r0*(speeds+1)*np.cos(headings)
	v10=r0*(speeds+1)*np.sin(headings)

	m = Basemap(projection='merc', llcrnrlat=south, llcrnrlon=west, urcrnrlat=north, urcrnrlon=east, resolution='f')
	m.drawcoastlines(color = '0.15')
	m.fillcontinents(color='#cc9955',lake_color='aqua')
	m.drawcoastlines(linewidth=0.4)
	m.drawmapboundary()
	U, V, X, Y = m.rotate_vector(u10, v10, lons, lats, returnxy=True)

	# draw arrow
	m.quiver(X, Y, U, V, edgecolor='k', facecolor='r', linewidth=.1, scale_units='inches', scale=1)
	strdatetime=str(saildatetime).replace("-","")
	strdatetime=str(strdatetime).replace(" ","-")
	strdatetime=str(strdatetime).replace(":","")
	plt.title("Trajectory of Single Voyage"+" (idx="+str(fileidx)+")\n key="+mmsi+"-"+timestamp,fontsize=16, fontname='serif')			
	#plt.xlabel("Longitude", fontsize=16, fontname='serif') # x軸のタイトル
	#plt.ylabel("Latitude", fontsize=16, fontname='serif') # y軸
    # draw meridians and parallels.
	m.drawmeridians(np.arange(west, east, lonscale),labels=[0,0,0,1],linewidth=0.3)
	m.drawparallels(np.arange(south, north, latscale),labels=[1,1,0,0],linewidth=0.3)
	pngfile = outputdir+"/singlevoyage-trajectory-idx"+str(fileidx)+"-"+mmsi+"-"+timestamp+".png"
	plt.savefig(pngfile,dpi=360)
	plt.clf() # clear figure
	#plt.show()

	return 


def main():
	minlength=500 # minimum length of tracjectory data (set to 2000, 2024/06/30)
	outputdir = "./onesailtrajectory"
	if not os.path.exists(outputdir):
		os.makedirs(outputdir)
	# processing data
	datafile = "./pickledata/onesailbymmsitimestamp"+str(minlength)+".pickle"
	if os.path.exists(datafile):
		print( datafile+" EXIST !!!" )
		with open(datafile, mode="rb") as fi:
				onesailbymmsitimestamp=pickle.load(fi)
	else:
		print( datafile+" NOT Exist !!!" )
		exit()
	# read keysfile
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

	fileidx=0
	for mmsitimestamp in onesailkeys:
		onesaildata = onesailbymmsitimestamp[mmsitimestamp]
		onesaildata=np.transpose(np.array(onesaildata))
		mmsi, timestamp = mmsitimestamp.split("-")
		drawonesaildata(outputdir, mmsi, timestamp, fileidx, onesaildata)
		fileidx += 1

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
