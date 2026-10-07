import os
import csv
import random
import sys
from pythonnet import load

# 1. Initialize .NET Runtime
load("coreclr")
import clr
import System
from System import Single, Boolean, String
from System.Threading import ThreadStart
# Add this near your other 'from System...' imports at the top
clr.AddReference("System.Numerics")
from System.Numerics import Vector3

# --- NEW: Import the .NET Native Library Loader ---
from System.Runtime.InteropServices import NativeLibrary

# 2. Set your paths
root_dir = r"D:\nyron 2"
dll_folder = os.path.join(root_dir, "PicoGK_Engine", "bin", "Release", "net9.0")
dll_path = os.path.join(dll_folder, "PicoGK.dll")
engine_path = os.path.join(dll_folder, "picogk.1.7.dll") # The exact path to the muscles

workspace_dir = os.path.join(root_dir, "noyron_workspace")
if not os.path.exists(workspace_dir):
    os.makedirs(workspace_dir)

if not os.path.exists(dll_path):
    print(f"❌ DLL NOT FOUND at: {dll_path}")
    sys.exit()

# 3. Load the C# Dashboard
clr.AddReference(dll_path)
import PicoGK

# ---------------------------------------------------------
# 4. THE MAGIC BULLET: Pre-load the C++ Engine into .NET
# ---------------------------------------------------------
try:
    print("Injecting C++ muscles directly into .NET memory...")
    NativeLibrary.Load(engine_path)
    print("✅ Muscles injected successfully!")
except Exception as e:
    print(f"❌ Failed to inject muscles: {e}")





def noyron_brain_task():
    print("--------------------------------------------------")
    print("🚀 NOYRON 2.0: DATA FACTORY INITIALIZED")
    print("--------------------------------------------------")

    # Constant parameters
    boom_length = 150.0
    outer_radius = 20.0
    hole_radius = 8.0

    # Prepare the CSV Ledger
    csv_path = os.path.join(workspace_dir, "boom_dataset.csv")

    try:
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(["Boom_ID", "Wall_Thickness_mm", "Num_Holes", "STL_Filename"])

            print("Beginning batch generation of 100 variants. This might take a minute...")

            # The Data Factory Loop
            for i in range(1, 101):
                # 1. RANDOMIZE THE DNA
                # Wall thickness between 2.0mm and 6.0mm
                wall_thickness = round(random.uniform(2.0, 6.0), 2)
                # Between 0 and 10 lightening holes
                num_holes = random.randint(0, 10)

                boom_id = f"Boom_{i:03d}"
                stl_name = f"{boom_id}.stl"

                # Print progress every 10 parts
                if i % 10 == 0:
                    print(f"--> Generating {boom_id} / 100...")

                # 2. GENERATE STOCK
                lat_stock = PicoGK.Lattice()
                lat_stock.AddBeam(
                    Vector3(0.0, 0.0, Single(-(boom_length/2))),
                    Vector3(0.0, 0.0, Single((boom_length/2))),
                    Single(outer_radius), Single(outer_radius), Boolean(False)
                )
                vox_part = PicoGK.Voxels(lat_stock)

                # 3. HOLLOW THE CORE
                lat_core = PicoGK.Lattice()
                inner_radius = outer_radius - wall_thickness
                lat_core.AddBeam(
                    Vector3(0.0, 0.0, Single(-(boom_length/2 + 10))),
                    Vector3(0.0, 0.0, Single((boom_length/2 + 10))),
                    Single(inner_radius), Single(inner_radius), Boolean(False)
                )
                vox_part.BoolSubtract(PicoGK.Voxels(lat_core))

                # 4. DRILL DYNAMIC HOLES
                if num_holes > 0:
                    lat_holes = PicoGK.Lattice()
                    spacing = boom_length / (num_holes + 1)
                    start_z = -(boom_length / 2) + spacing

                    for h in range(num_holes):
                        z_pos = start_z + (h * spacing)
                        p1 = Vector3(-40.0, 0.0, Single(z_pos))
                        p2 = Vector3(40.0, 0.0, Single(z_pos))
                        lat_holes.AddBeam(p1, p2, Single(hole_radius), Single(hole_radius), Boolean(False))

                    vox_part.BoolSubtract(PicoGK.Voxels(lat_holes))

                # 5. EXPORT STL
                stl_path = os.path.join(workspace_dir, stl_name)
                part_mesh = PicoGK.Mesh(vox_part)
                try:
                    part_mesh.SaveToStlFile(String(stl_path))
                except:
                    part_mesh.Save(String(stl_path))

                # 6. LOG TO LEDGER
                writer.writerow([boom_id, wall_thickness, num_holes, stl_name])

        print("\n✅ DATA FACTORY COMPLETE!")
        print(f"100 STLs and 'boom_dataset.csv' saved to: {workspace_dir}")
        print("Close the viewer to exit.")

    except Exception as e:
        print(f"❌ Generation Error: {e}")







# 6. Start the Engine
try:
    print("Igniting the engine...")

    v_size      = Single(1.0)
    task        = ThreadStart(noyron_brain_task)
    log_dir     = String(workspace_dir)
    log_name    = String("noyron_log.txt")
    src_dir     = String(workspace_dir)
    lights_file = String("")
    b_end_app   = Boolean(False)

    PicoGK.Library.Go(v_size, task, log_dir, log_name, src_dir, lights_file, b_end_app)

except Exception as e:
    print(f"❌ Engine Crash: {e}")