import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans"})
fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
C = {"geo":"#E8EEF7","cell":"#DCE8F5","data":"#E3F0E6","gap":"#FCE9D9","lit":"#F2EEE3","dev":"#E9E4F2","opt":"#F7E1E1","exp":"#E6F2F1","out":"#EEEEEE"}
E = {"geo":"#4A6FA5","cell":"#2F5D8A","data":"#3C7A4E","gap":"#B85C14","lit":"#7D6D3F","dev":"#5E4B8B","opt":"#A23B3B","exp":"#2E7D74","out":"#555555"}
W, H = 30.5, 17.0
X = [1, 34.75, 68.5]; Y = [48, 26.5, 1]
def box(c,r,key,title,body):
    x,y=X[c],Y[r]
    ax.add_patch(FancyBboxPatch((x,y),W,H,boxstyle="round,pad=0.3,rounding_size=1.2",fc=C[key],ec=E[key],lw=1.0))
    ax.text(x+W/2,y+H-1.6,title,ha="center",va="top",fontsize=7.4,fontweight="bold",color=E[key])
    ax.text(x+W/2,y+H-5.4,body,ha="center",va="top",fontsize=6.2,color="#222222",linespacing=1.3)
def arrow(p0,p1,rad=0.0,label=None,lx=0,ly=0):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=8,lw=0.9,color="#444444",connectionstyle=f"arc3,rad={rad}",shrinkA=0,shrinkB=0))
    if label: ax.text(lx,ly,label,ha="center",va="center",fontsize=6.0,color="#444444",style="italic")
box(0,0,"geo","1  TPMS cell geometry","gyroid, diamond and blends\nnetwork solid where $\\phi_w \\leq c(w,\\rho^*)$\nperiodic voxels; wall, pore, $a_{sf}$\nfeasible cell size $L \\geq L_{min}(w,\\rho^*)$")
box(1,0,"cell","2  Periodic cell solvers","$k_{eff}$ (finite volumes), $C_{eff}$ (Q1 FEA)\n$K$ (MAC Stokes, MINRES)\ntwo-grid Richardson R(32, 64)\nverified against exact solutions")
box(2,0,"data","3  Closure surrogates","290 cells $\\theta=(w,\\rho^*,a_z)$, Sobol design\nlog-Euclidean tensor targets\nGP and deep ensemble, exact\nsymmetry, recalibrated uncertainty")
box(1,1,"gap","4  Finite-gap strips (RQ1)","N = 1-8 cells between bonded walls\n$\\delta_q = q_{strip}/q_{bulk} - 1$\nfitted correction $1 \\mp a(\\theta)/N$\n(+ graded strips, cut phases)")
box(0,1,"lit","5  Literature closures","Forchheimer coefficient $C_F$\n(Gajetti et al. 2025)\ninterstitial Nu, Reynolds analogy\n(Savoldi et al. 2026) $\\times\\,Pr^{1/3}$")
box(2,1,"dev","6  Homogenized jacket model","annulus unwrapped to $(s,z)$\nDarcy-Forchheimer flow\n3-field non-equilibrium heat model\nsandwich structure; < 1 s per design")
box(2,2,"opt","7  Multi-objective design","min (R, pump power, mass), 3 L/min\n72 B-spline controls of $\\rho^*$, $w$, $L$\nNSGA-II | qNEHVI | Sobol, 5 seeds\nvs baselines B1-B3 (RQ2, RQ3)")
box(1,2,"exp","8  Geometry export","PicoGK implicit in C#\nconformal phase-field grading\nSTL of jacket and 30$^\\circ$ sector\nmin wall, throat, pockets checked")
box(0,2,"out","Outputs","R, $\\Delta p$, pump power, mass, margin\nPareto fronts and baselines\nmanufacturable optimum (STL)\nscripted, reproducible results")
yc=[Y[r]+H/2 for r in range(3)]
arrow((X[0]+W+0.4,yc[0]),(X[1]-0.4,yc[0])); arrow((X[1]+W+0.4,yc[0]),(X[2]-0.4,yc[0]))
arrow((X[1]+W/2,Y[0]-0.4),(X[1]+W/2,Y[1]+H+0.4),label="same solvers on strips",lx=X[1]+W/2+11,ly=Y[0]-2.1)
arrow((X[2]+W/2,Y[0]-0.4),(X[2]+W/2,Y[1]+H+0.4),label="K, k, C, a$_{sf}$, loc.",lx=X[2]+W/2+9,ly=Y[0]-2.1)
arrow((X[1]+W+0.4,yc[1]),(X[2]-0.4,yc[1]),label="$f_K$",lx=X[2]-2.0,ly=yc[1]+1.6)
arrow((X[0]+W/2,Y[1]-0.4),(X[2]+6,Y[1]-0.4),rad=0.13,label="$C_F$, $h_{sf}$",lx=X[1]+W/2,ly=Y[1]-6.2)
arrow((X[2]+W/2,Y[1]-0.4),(X[2]+W/2,Y[2]+H+0.4))
arrow((X[2]-0.4,yc[2]),(X[1]+W+0.4,yc[2])); arrow((X[1]-0.4,yc[2]),(X[0]+W+0.4,yc[2]))
fig.savefig("figs/workflow.png",dpi=300,bbox_inches="tight",facecolor="white")
