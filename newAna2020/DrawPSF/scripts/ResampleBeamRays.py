#######################################################
# Interpolate the PSF within the simulated ray grid
#
# - author : Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : June 4rd 2020
#
#######################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import pandas as pd
import itertools
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle,Ellipse

import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import cm


# to enlarge the sizes
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (13, 13),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'font.size': 14}
plt.rcParams.update(params)


from scipy import interpolate

from sklearn.neighbors import NearestNeighbors

from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model

from datetime import datetime,date

today = date.today()
string_date=today.strftime("%Y-%m-%d")

# Constants
#-----------

m_to_mm=1000.
mm_to_m=1e-3
mm_to_micr=1e3
mm_to_nm=1e6
inch_to_mm=25.4
mm_to_inch=1./inch_to_mm
micr_to_m=1e-6
micr_to_mm=1e-3
m_to_micr=1./micr_to_m
m_to_cm=100.
m_to_nm=1e9
nm_to_m=1./m_to_nm


arcdeg_to_arcmin=60.
arcmin_to_arcdeg=1./arcdeg_to_arcmin
arcmin_to_arcsec=60.
arcdeg_to_arcsec=arcdeg_to_arcmin*arcmin_to_arcsec
arcsec_to_arcdeg=1./arcdeg_to_arcsec

deg_to_rad=np.pi/180.

rad_to_deg=1./deg_to_rad
rad_to_arcsec=rad_to_deg*arcdeg_to_arcsec
rad_to_arcmin=rad_to_deg*arcdeg_to_arcmin
arcmin_to_rad=1./rad_to_arcmin

# Telescope
#----------

Tel_Focal_Length=20.6 # m : Focal length of the telescope
Tel_Diameter=1.2 # m : Diameter of the telescope
Tel_Fnum=Tel_Focal_Length/Tel_Diameter
pltscale=206265/(Tel_Focal_Length*m_to_mm)  # arcsec per mm

Filt_D=0.200 # m distance of the filter position wrt CCD plane
Filt_size=3*inch_to_mm

Det_xpic=10.0 # microns per pixel
#Det_NbPix=2048 # number of pixels per CCD side
Det_NbPix=4096 # number of pixels per CCD side For 400 only
Det_size=Det_xpic*Det_NbPix*micr_to_mm # CCD size in mm, 5 cm or 2 inch


# Input file

# number of rays
NBEAM_X=11
NBEAM_Y=11
NBEAM=NBEAM_X*NBEAM_Y
NWL=4
NBTOT=NBEAM*NWL

theta_x=0.  # angle in arcmin
theta_y=0.  # angle in arcmin

theta_x_num=int(theta_x*10)
theta_y_num=int(theta_y*10)

if theta_x_num>0:
    theta_nstr='{:0>2}'.format(theta_x_num)
    theta_x_str="p"+theta_nstr
else:
    theta_nstr='{:0>2}'.format(-theta_x_num)
    theta_x_str="m"+theta_nstr

if theta_y_num>0:
    theta_nstr='{:0>2}'.format(theta_y_num)
    theta_y_str="p"+theta_nstr
else:
    theta_nstr='{:0>2}'.format(-theta_y_num)
    theta_y_str="m"+theta_nstr

Beam4_Rayfile="Beam4_Rayfile_{:d}_allwl_{}_{}".format(NBTOT,theta_x_str,theta_y_str)


rayfile_R150="R150_Beam4_Rayfile_484_allwl_m00_m00_RAY_OUT.xlsx"
rayfile_hoe="HOE_Beam4_Rayfile_484_allwl_m00_m00_RAY_OUT.xlsx"


# Read beam files


#input_hoe = pd.ExcelFile(rayfile_hoe)
#df_hoe = input_hoe.parse(index_row=0,header=1)
#df_hoe=df_hoe.iloc[0:NBTOT]

#input_R150 = pd.ExcelFile(rayfile_R150)
#df_R150 = input_R150.parse(index_row=0,header=1)
#df_R150=df_R150.iloc[0:NBTOT]

input_R150 = pd.ExcelFile(rayfile_R150)
df_R150 = input_R150.parse(index_row=0,header=1)
#df_R150=df_R150.iloc[0:NBTOT]
df_R150=df_R150.drop(0)
df_R150 = df_R150.reset_index()

input_hoe = pd.ExcelFile(rayfile_hoe)
df_hoe = input_hoe.parse(index_row=0,header=1)
#df_hoe=df_hoe.iloc[0:NBTOT]
df_hoe=df_hoe.drop(0)

# rename columns otherwise they are not recognize and swap X,Y
df_hoe.columns = ["X0","Y0","Z0","U0","V0","W0","wave","col","X1","Y1","Z1","X2","Y2","Z2","X3","Y3","Z3","Xgoal","Ygoal","Xfinal","Yfinal","Zfinal","Notes","Unnamed"]


# Selection of input

FLAG_R150 = True
FLAG_HOE = False
FLAG_PLOT = False


if FLAG_R150:
    df = df_R150
    disperser_name="Ronchi 150"
    outputdata_csv = "R150_RESAMPLEBEAM_" + Beam4_Rayfile + "_out.csv"
    outputdata_fits = "R150_RESAMPLEBEAM_" + Beam4_Rayfile + "_out.fits"


else:
    df = df_hoe
    disperser_name="Hologram"
    outputdata_csv = "HOE_RESAMPLEBEAM_" + Beam4_Rayfile + "_out.csv"
    outputdata_fits = "HOE_RESAMPLEBEAM_" + Beam4_Rayfile + "_out.fits"

print(df)

# Show Beam
#-------------
X0C=df["X0"].mean()
Y0C=df["Y0"].mean()

RXMAX=np.max(np.abs(df["X0"].values-X0C))
RYMAX=np.max(np.abs(df["Y0"].values-Y0C))
RMAX=np.max(np.array([RXMAX,RYMAX]))


def Select_Beam(row):
    return (row["X0"]-X0C)**2+(row["Y0"]-Y0C)**2 <= RMAX**2

if FLAG_PLOT:
    # Figure 1 : Show input beam
    #-----------------------------

    X0 = np.array(df["X0"].values, dtype='float64')
    Y0 = np.array(df["Y0"].values, dtype='float64')
    V0 = np.array(df["V0"].values, dtype='float64')
    U0 = np.array(df["U0"].values, dtype='float64')

    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    df.plot.scatter(x="X0", y="Y0", c="DarkBlue", marker="o", ax=ax1)
    ax1.set_aspect("equal")
    ax1.grid()

    q = ax2.quiver(X0, Y0, U0, V0, color="red")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X0 (mm)")
    ax2.grid()

    plt.suptitle("Unfiltered square beam")
    plt.show()

###############################################################
# Beam rays resampling
##############################################################
WLSIM=np.array([0.0004,0.0006,0.0008,0.001]) # all simulated wavelength in mm
NSIMRAYS=100000  # number or rays to resample
#NSIMRAYS=20000  # number or rays to resample

SIMRAYS_COORDINATES=(np.random.random((NSIMRAYS,2))-0.5)*2*RMAX
simrays_sel_idx=np.where(SIMRAYS_COORDINATES[:,0]**2+SIMRAYS_COORDINATES[:,1]**2<(RMAX)**2)[0]
SIMRAYSSEL_COORDINATES = SIMRAYS_COORDINATES[simrays_sel_idx]
NBSIMRAYSSEL=len(SIMRAYSSEL_COORDINATES)

if FLAG_PLOT:
    ## Figure for resampling
    f, ax = plt.subplots(figsize=(6,6))
    ax.scatter(SIMRAYSSEL_COORDINATES[:,0],SIMRAYSSEL_COORDINATES[:,1],marker="." ,color="DarkBlue",s=5)
    ax.scatter(df["X0"].values, df["Y0"],marker="o" ,color="red",s=50)
    ax.grid()
    ax.set_aspect("auto")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_title(" Random beam")
    plt.show()



all_Xccd=[]
all_Yccd=[]
#######
#  LOOP on wavelength
########
for WL in WLSIM:
    print("********** WL = {} ********** ".format(WL))
    df_SIM = df.loc[df.wave == WL]

    NBEAMS = len(df_SIM)

    if FLAG_PLOT:
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=NBEAMS)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        all_colors = scalarMap.to_rgba(np.arange(NBEAMS), alpha=1)


        f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5))
        df_SIM.plot.scatter(x="X2", y="Y2", c=all_colors, marker="o", ax=ax1, s=50)
        ax1.set_aspect("equal")
        ax1.grid()
        title1 = "Rays at {}".format(disperser_name)
        ax1.set_title(title1)

        df_SIM.plot.scatter(x="X3", y="Y3", c=all_colors, marker="o", ax=ax2, s=50)
        ax2.set_aspect("equal")
        ax2.grid()
        title2 = "{} : CCD $\\lambda$ = {} nm".format(disperser_name, WL * mm_to_nm)
        ax2.set_title(title2)

        title="{} : $\lambda$ = {}$ nm".format(disperser_name,WL*mm_to_nm)
        plt.suptitle(title, Y=1.1, fontsize=25)
        plt.tight_layout()
        plt.show()



    ############################
    # K nearest neighbourg
    #########################
    X = df_SIM["X0"].values
    Y = df_SIM["Y0"].values
    Z1 = df_SIM["X3"].values
    Z2 = df_SIM["Y3"].values

    refXY = np.dstack((X, Y))[0]

    # Fit KNN
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(refXY)

    # Find the neighbours of simulated resampled
    distances, indices = nbrs.kneighbors(SIMRAYSSEL_COORDINATES)

    # One by one do bilinear interpolation
    Xccd = np.zeros(NBSIMRAYSSEL)
    Yccd = np.zeros(NBSIMRAYSSEL)

    for idx in np.arange(NBSIMRAYSSEL):

        if idx%10000==0:
            print("\t - WL = {} , sim ray beam = {}".format(WL,idx))


        dd = distances[idx]
        indrays = indices[idx]

        # reference rays original coordinates
        theX = df_SIM.iloc[indrays]["X0"].values
        theY = df_SIM.iloc[indrays]["Y0"].values

        theX = theX.astype(float)
        theY = theY.astype(float)

        #print("idx = ",idx, "Xray = ",SIMRAYSSEL_COORDINATES[idx, 0]," Yray = ",SIMRAYSSEL_COORDINATES[idx, 1] )
        #print("idx = ", idx, " theX = ", theX, " theY = ", theY)

        # reference rays final coordinates on CCD
        ZX = df_SIM.iloc[indrays]["X3"].values
        ZY = df_SIM.iloc[indrays]["Y3"].values

        ZX = ZX.astype(float)
        ZY = ZY.astype(float)

        #print("idx = ",idx," ZX = ",ZX," ZY = ",ZY)

        fX = interpolate.interp2d(theX, theY, ZX, kind='linear')
        fY = interpolate.interp2d(theX, theY, ZY, kind='linear')

        Xccd[idx] = fX(SIMRAYSSEL_COORDINATES[idx, 0], SIMRAYSSEL_COORDINATES[idx, 1])
        Yccd[idx] = fY(SIMRAYSSEL_COORDINATES[idx, 0], SIMRAYSSEL_COORDINATES[idx, 1])
    # end loop on rays

    all_Xccd.append(Xccd)
    all_Yccd.append(Yccd)

    if FLAG_PLOT:
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        NBINX = (Xccd.max() - Xccd.min()) / (Det_xpic * micr_to_mm)
        NBINY = (Yccd.max() - Yccd.min()) / (Det_xpic * micr_to_mm)

        ax1.scatter(Xccd, Yccd, marker="o", color="DarkBlue", s=1)
        ax1.scatter(df_SIM["X3"].values, df_SIM["Y3"], marker="o", color="red", s=50)
        ax1.grid()
        ax1.set_aspect("equal")
        ax1.set_xlabel("Xccd mm")
        ax1.set_ylabel("Yccd mm")
        ax1.set_title(" Random beam on CCD")

        ax2.hist2d(Xccd, Yccd, bins=(NBINX, NBINY), cmap=cm.get_cmap('jet', 512))
        ax2.set_aspect("equal")
        ax2.set_xlabel("Xccd mm")
        ax2.set_ylabel("Yccd mm")
        ax2.set_title(" Random beam on CCD")
        plt.show()


# save beam

df_out=pd.DataFrame()

df_out["X0"] = SIMRAYSSEL_COORDINATES[:,0]
df_out["Y0"] = SIMRAYSSEL_COORDINATES[:,1]

df_out["Xccd_400"] = all_Xccd[0]
df_out["Xccd_600"] = all_Xccd[1]
df_out["Xccd_800"] = all_Xccd[2]
df_out["Xccd_1000"] = all_Xccd[3]

df_out["Yccd_400"] = all_Yccd[0]
df_out["Yccd_600"] = all_Yccd[1]
df_out["Yccd_800"] = all_Yccd[2]
df_out["Yccd_1000"] = all_Yccd[3]


print(df_out)

df_out.to_csv(outputdata_csv)





























