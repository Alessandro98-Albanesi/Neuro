import numpy as np
import socket
import struct
import array
import pickle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pyvista as pv
import open3d as o3d
from scipy.spatial import KDTree
import time
import math
import datetime

TumorSavedName="tumorProj"
TumorExpSavedName="tumorProjExp"

units_obj=1 #1 if obj are already in meters, 1000 if they are in mm
pt="Manic3t"#"Circo" #"marra" #"marra" #"albani"
if pt=="marra":
    face = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/Face_MarraG_clean.obj")
    #tumor = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/tumor_MarraG.stl")
    tumor = pv.read(
        "C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/tumor_MarraG_moved.stl")

if pt=="albani":
    face = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/Face_AlbaniE_clean.obj")
    tumor = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/tumor_AlbaniE.stl")

if pt=="slicer":
    face = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/Face_Slicer_clean.obj")
    tumor = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/tumor_Slicer.obj")

if pt=="Circo":
    #face = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/skull_cutCircoInside.stl")
    face = pv.read(
        "C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/ModelloStampatoValutazione/stampato_outsideLayer.stl")

    tumor = pv.read(
        "C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/Fake_Tumor1Circo2.stl")

if pt=="Manic":
    #face = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/skull_cutCircoInside.stl")
    face = pv.read(
        "C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/Acquisiz_Phantom_CT/head_manichi_red_m2.obj")

    tumor = pv.read(
        "C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/Fake_Tumor1Circo2_m.stl")

if pt=="Manic3t":
    #face = pv.read("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/Craniotomia_Automatic/skull_cutCircoInside.stl")
    face = pv.read(
        "C:/Users/Alessandro/Desktop/Neuro/face_3t_mWtextr.obj")

    tumor = pv.read(
        "C:/Users/Alessandro/Desktop/Neuro/tumor1_m2.obj")



tumorDec=1
reduction=0.26
nPt_Tumor=5000
decimTumor=tumor
if tumorDec==1:
    while (len(decimTumor.points) > nPt_Tumor  ):
        decimTumor = tumor.decimate(reduction)
        print("reduction=", reduction, "  pt tumor =", len(decimTumor.points), )
        reduction = reduction + 0.1
tumor=decimTumor
#identify the projection of the tumor on the skin
def computeTumorProj(Avg_nearbyPts,CoM_tumor,tumor, growValue, face, update ):
    dir_COM_skin=Avg_nearbyPts-CoM_tumor
    direction= dir_COM_skin / np.linalg.norm(dir_COM_skin)
    #vector_dir=np.ones((len(tumor.points),3))*direction

    pts_moved=tumor.points+(direction*2)
    #trova punti di intersezione per ogni punto del tumore

    #rateofExpansion to add tot cm= growValue to each point
    bounds=tumor.bounds
    Exp_perc=((bounds[1]-bounds[0])+growValue)/(bounds[1]-bounds[0])

    #pts_expandedMoved=(tumor.points*Exp_perc)+(direction*100)
    #pts_expanded=(tumor.points*Exp_perc)
    pts_expanded=(tumor.points*Exp_perc)
    com_expanded=pv.PolyData(pts_expanded).center_of_mass()
    #recentering the data
    pts_expanded=pv.PolyData(pts_expanded-(com_expanded-CoM_tumor), tumor.faces)
    pts_expandedMoved=pts_expanded.points+(direction*2)
    i=0
    projectedPts=[]
    Prj_expandedPts=[]

    p = pv.Plotter()
    color_Nexp = "aliceblue"
    p.add_mesh(face, color=lightpink2, opacity=0.35, smooth_shading=True)  # face
    p.add_mesh(tumor, color=color_Nexp, opacity=0.95, smooth_shading=True)  # tumor
    p.add_mesh(CoM_tumor, color="black", smooth_shading=True)  # com tumor
    p.add_mesh(Avg_nearbyPts, color="blue", smooth_shading=True )
    p.add_mesh(nearby_points, color=True, smooth_shading=True )
    p.add_mesh(ray, color="dimgrey", line_width=5, label="Ray Segment")  # traj
    p.camera.roll += 0
    p.camera.azimuth += 20
    p.camera.elevation -= 5
    p.set_background("white")
    p.show()

    for pts in tumor.points:
        str=pts.tolist()
        end=pts_moved[i].tolist()
        prjPt, prj_ind = face.ray_trace(str, end)
        # Create geometry to represent ray trace
        projectedPts.append(prjPt)

        # #create the expanded model by 2cm
        expandedPt=pts_expanded.points[i].tolist()
        expandedPt_moved = pts_expandedMoved[i].tolist()
        prjPtExp, prj_indExp = face.ray_trace(expandedPt, expandedPt_moved)
        Prj_expandedPts.append(prjPtExp)
        i += 1

    # for pts in pts_expanded:
    #     str = pts.tolist()
    #     end = pts_expandedMoved[i].tolist()
    #     prjPt, prj_ind = face.ray_trace(str, end)
    #     # Create geometry to represent ray trace
    #     Prj_expandedPts.append(prjPt)
    #     i+=1


    projectedPtsPD=pv.PolyData(np.asarray(projectedPts).squeeze())
    projectedPtsExpPD=pv.PolyData(np.asarray(Prj_expandedPts).squeeze())

    proj_surf = projectedPtsPD.delaunay_2d(alpha=1)
    projExp_surf = projectedPtsExpPD.delaunay_2d(alpha=1)



    #do the same but with a tumor expanded by tot cm about 2cm (20mm)

    color_exp="dodgerblue"
    color_Nexp="aliceblue"
    #color_Nexp="blue"
    show=1
    if show==1:
        p = pv.Plotter()
        #p.add_mesh(face,scalars="distances", opacity=0.95, smooth_shading=True, cmap=my_colormap)
        p.add_mesh(face, color=lightpink2, opacity=0.35, smooth_shading=True)  #face
        p.add_mesh(tumor, color=color_Nexp, opacity=0.95, smooth_shading=True)        #tumor
        p.add_mesh(CoM_tumor, color="black", smooth_shading=True )         #com tumor
        #p.add_mesh(Avg_nearbyPts, color=True, smooth_shading=True )
        #p.add_mesh(nearby_points, color=True, smooth_shading=True )
        p.add_mesh(ray, color="dimgrey", line_width=5, label="Ray Segment")    #traj
        #p.add_mesh(intersection, color="blue", point_size=10, label="Intersection Points")
        #p.add_mesh(pts_moved, color="blue", line_width=5, label="Ray Segment")
        #p.add_mesh(pts_expanded, color=color_exp, opacity=0.45, label="Ray Segment")   #tumore expanded
        #p.add_mesh(projectedPtsPD, color="blue", point_size=2, label="Intersection Points")

        #p.add_mesh(projectedPtsExpPD, color="white", point_size=2, label="Intersection Points")
        p.add_mesh(proj_surf, color="cornflowerblue", opacity=0.99, smooth_shading=True)          #projection tumore
        p.add_mesh(projExp_surf, color="hotpink", opacity=0.99, smooth_shading=True)        #projection expanded tum
        p.camera.roll+= 0
        p.camera.azimuth+=20
        p.camera.elevation-=5
        p.set_background("white")
        p.show()
    #from mm to m
    proj_surf.points=proj_surf.points/units_obj
    projExp_surf.points = projExp_surf.points / units_obj
    decimation=1
    nPt=400
    reduction=0.96
    # proj_surf = proj_surf.decimate(reduction)
    # projExp_surf = projExp_surf.decimate(reduction)
    decimProj_surf = proj_surf
    decimProj_surfExp = projExp_surf

    if decimation:
        while (len(decimProj_surf.points)>nPt and len(decimProj_surfExp.points>nPt)):
            decimProj_surf = proj_surf.decimate(reduction)
            decimProj_surfExp = projExp_surf.decimate(reduction)
            print("reduction=", reduction, "  pt proj exp =", len(decimProj_surf.points), "  pt proj  =",
                  len(decimProj_surfExp.points))
            reduction = reduction + 0.005

    proj_surf=decimProj_surf
    projExp_surf=decimProj_surfExp

    pl = pv.Plotter()
    _ = pl.add_mesh(proj_surf)
    if update==1:
        pl.export_obj('tumorProjUpdate.obj')
    else:
        pl.export_obj('tumorProj.obj')
    pl.close()
    pl = pv.Plotter()
    _ = pl.add_mesh(projExp_surf)
    if update == 1:
        pl.export_obj('tumorProjExpUpdate.obj')
    else:
        pl.export_obj('tumorProjExp.obj')
    pl.close()
    now = datetime.datetime.now()
    # Format the date and time into a string
    date_time_str = now.strftime("%Y-%m-%d_%H-%M")

    proj_surf.save("C:/Users/Alessandro/Desktop/Neuro/"+TumorSavedName+date_time_str+".stl")
    projExp_surf.save("C:/Users/Alessandro/Desktop/Neuro/"+TumorExpSavedName+date_time_str +".stl")

    return proj_surf, projExp_surf, Avg_nearbyPts, CoM_tumor

growValue=0.010*units_obj # intorno di 1cm oltre alla lesione #20 mm

#calcolo la distanza tra il tumore e la faccia
tree = KDTree(tumor.points)
d_kdtree, idx = tree.query(face.points) #d_kdtree ha stessa dim di Face, e ogni punto ha la distanza rispetto al tumore
face["distances"] = d_kdtree
np.mean(d_kdtree)

treshold=face["distances"].min()
pts_belowDist=np.argwhere(face["distances"]<=treshold)
nearbySurf=face.extract_points(face["distances"]<=treshold) #estraggo i punti della faccia che hanno una distanza dal tumore inferiore ad una certa soglia


nearby_points=nearbySurf.points
Avg_nearbyPts=nearbySurf.center_of_mass() #prendo il centro di questi punti vicini
CoM_tumor=tumor.center_of_mass() #centro del tumore


# Define the colors we want to use
blue = np.array([12 / 256, 238 / 256, 246 / 256, 1.0])
green = np.array([0.0, 1.0, 0.0, 1.0])
grey = np.array([189 / 256, 189 / 256, 189 / 256, 1.0])
yellow = np.array([255 / 256, 247 / 256, 0 / 256, 1.0])
red = np.array([1.0, 0.0, 0.0, 1.0])
lightpink2=np.array((247/256, 188/256, 196/256, 0.5))
mapping = np.linspace(face["distances"].min(), face["distances"].max(), 256)
newcolors = np.empty((256, 4))
newcolors[mapping > treshold] = lightpink2
newcolors[mapping <= treshold] = green
my_colormap = ListedColormap(newcolors)

#start and stop punti per tracciare la traiettoria
start = CoM_tumor.tolist()
dist=math.sqrt((Avg_nearbyPts[0]-CoM_tumor[0])*2*units_obj+(Avg_nearbyPts[1]-CoM_tumor[1])*2*units_obj+(Avg_nearbyPts[2]-CoM_tumor[2])*2*units_obj)
Vdir=(Avg_nearbyPts-CoM_tumor)/dist
stop = (Avg_nearbyPts+ Vdir*0.30*units_obj).tolist()

# Perform ray trace between the center of the tumor and the stop point outside
points, ind = face.ray_trace(start, stop)

# Create geometry to represent ray trace
ray = pv.Line(start, stop)
intersection = pv.PolyData(points)

computeProjections=1
if computeProjections==1:
    startTime=time.time()
    update=1
    (proj_surf, projExp_surf, Avg_nearbyPts, CoM_tumor)=computeTumorProj(Avg_nearbyPts,CoM_tumor,tumor, growValue, face, update )
    print("exec time  took %.3f sec.\n" % (time.time() - startTime))
    #salva file non è necessario se poi mando solo vertici e connettività a unity
