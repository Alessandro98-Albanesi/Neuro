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
        pl.export_obj('C:/Users/Alessandro/Desktop/Neuro/tumorProjUpdate.obj')
    else:
        pl.export_obj('C:/Users/Alessandro/Desktop/Neuro/tumorProj.obj')
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

    proj_surf.save("C:/Users/Alessandro/Desktop/Neuro"+TumorSavedName+date_time_str+".stl")
    projExp_surf.save("C:/Users/Alessandro/Desktop/Neuro"+TumorExpSavedName+date_time_str+".stl")

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


exchangeData=1


def ObjLoader(fileName):
    obj=[]
    faces=[]
    vertices=[]
    obj_type=0
    try:
        f = open(fileName)
        for line in f:
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)

                vertex = (float(line[index1:index2])*(-1), float(line[index2:index3]), float(line[index3:-1]))
                # vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                vertices.append(vertex)



            elif line[0] == "f":
                try:
                    string = line.replace("//", "/")
                    ##
                    i = 0
                    # face = []
                    for k in range(3):
                        i = string.find("/", i + 1)
                        face=(int(string[i + 1:string.find(" ", i + 1)])-1)
                        faces.append(face)
                    obj_type=1
                except:
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)
                    face = (int(line[index1:index2])-1 , int(line[index2:index3])-1, int(line[index3:-1])-1)
                    faces.append(face)

        if obj_type==1:
            nFace_tumorPrj = len(faces)
            nPt_tumorPrj = len(vertices)
            obj.extend([nPt_tumorPrj, nFace_tumorPrj])
            obj.extend(np.asarray(vertices).flatten().tolist())
            obj.extend(faces)
        else:
            nFace_tumorPrj = len(faces) * 3
            nPt_tumorPrj = len(vertices)
            nPt_tumorPrj = len(vertices)
            obj.extend([nPt_tumorPrj, nFace_tumorPrj])
            obj.extend(np.asarray(vertices).flatten().tolist())
            obj.extend(np.asarray(faces).flatten().tolist())


        f.close()
        return obj
    except IOError:
        print(".obj file not found.")



fileNameTP="C:/Users/Alessandro/Desktop/Neuro/tumorProj.obj"
fileNameTPE="C:/Users/Alessandro/Desktop/Neuro/tumorProjExp.obj"

fileNameTP_Update="C:/Users/Alessandro/Desktop/Neuro/tumorProjUpdate.obj"
fileNameTPE_Update="C:/Users/Alessandro/Desktop/Neuro/tumorProjExpUpdate.obj"

'''
print("id_button5")
stopUp = [ptRcv[1]*-units_obj, ptRcv[2]*units_obj, ptRcv[3]*units_obj]
startUp = [ptRcv[4]*-units_obj, ptRcv[5]*units_obj, ptRcv[6]*units_obj]
pickle.dump(stopUp, open("stopUp.p", "wb"))
pickle.dump(startUp, open("startUp.p", "wb"))
update=1
(proj_surf, projExp_surf, Avg_nearbyPts, CoM_tumor) = computeTumorProj(np.asarray(stopUp), np.asarray(startUp), tumor, growValue, face,update)
update = 0
'''

if exchangeData==1:
    while True:

        try:

            while True:
                HOST = "192.168.227.213"  # Standard loopback interface address (localhost) #127.0.0.1 #192.168.0.100
                PORT = 1000  # Port to listen on (non-privileged ports are > 1023)
                first = 1
                
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((HOST, PORT))
                    s.listen()
                    print("waiting for a client")
                    conn, addr = s.accept()
                    with conn:
                        print('Connected by', addr)
                        while True:
                            data = conn.recv(400000)
                            try:
                                arr = array.array('f', data)
                                if data:
                                    print(f"Received {data!r}")
                                    ptRcv = arr.tolist()
                                    id_button = int(ptRcv[0])
                                    if id_button==5:
                                        print("id_button5")
                                        stopUp = [ptRcv[1]*-units_obj, ptRcv[2]*units_obj, ptRcv[3]*units_obj]
                                        startUp = [ptRcv[4]*-units_obj, ptRcv[5]*units_obj, ptRcv[6]*units_obj]
                                        pickle.dump(stopUp, open("stopUp.p", "wb"))
                                        pickle.dump(startUp, open("startUp.p", "wb"))
                                        update=1
                                        (proj_surf, projExp_surf, Avg_nearbyPts, CoM_tumor) = computeTumorProj(np.asarray(stopUp), np.asarray(startUp), tumor, growValue, face,update)
                                        update = 0
                                        conn.sendall(b"done")




                            except:
                                if data:
                                    print(f"Received {data!r}") #da 2 in poi str(data)[2:...] i primi due caratteri sono -> b/
                                    #
                                    if str(data)[2:5]=="tpr":
                                        obj = ObjLoader(fileNameTP)
                                        tumorPrj2Send = np.asarray(obj, dtype=float)
                                        info2send=struct.pack('f' * len(tumorPrj2Send), *tumorPrj2Send)
                                        conn.sendall(info2send)
                                        print("sent tpr")
                                    if str(data)[2:5]=="tpe":
                                        obj_TPE = ObjLoader(fileNameTPE)
                                        tumorPrjExp2Send = np.asarray(obj_TPE, dtype=float)
                                        info2send=struct.pack('f' * len(tumorPrjExp2Send), *tumorPrjExp2Send)
                                        conn.sendall(info2send)
                                        print("sent tpe")
                                    if str(data)[2:5]=="trj":
                                        trj2send=[]
                                        trj2send.extend(start)
                                        trj2send.extend(stop)
                                        print("start is", start)
                                        print("stop is", stop)
                                        trj2send = np.asarray(trj2send, dtype=float)
                                        info2send = struct.pack('f' * len(trj2send), *trj2send)
                                        conn.sendall(info2send)
                                    if str(data)[2:5]=="utp": #updated tumor projection
                                        obj = ObjLoader(fileNameTP_Update)
                                        tumorPrj2Send = np.asarray(obj, dtype=float)
                                        info2send = struct.pack('f' * len(tumorPrj2Send), *tumorPrj2Send)
                                        conn.sendall(info2send)

                                    if str(data)[2:5]=="ute": #updated tumor projection expanded
                                        obj_TPE = ObjLoader(fileNameTPE_Update)
                                        tumorPrjExp2Send = np.asarray(obj_TPE, dtype=float)
                                        info2send = struct.pack('f' * len(tumorPrjExp2Send), *tumorPrjExp2Send)
                                        conn.sendall(info2send)

                                    if str(data)[2:5]=="utr":
                                        trj2send=[]
                                        stopUp=pickle.load(open("stopUp.p", "rb"))
                                        startUp=pickle.load(open("startUp.p", "rb"))
                                        trj2send.extend(startUp)
                                        trj2send.extend(stopUp)
                                        print("start Up is", startUp)
                                        print("stop Up is", stopUp)
                                        trj2send = np.asarray(trj2send, dtype=float)
                                        info2send = struct.pack('f' * len(trj2send), *trj2send)
                                        conn.sendall(info2send)



                            if not data:
                                break


        except:
            print("An Exception occurred")
            continue
