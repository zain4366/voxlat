import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pythonnet import load
# =========================================================================
# 1. WAKE UP THE GEOMETRY ENGINE (The Muscles)
# =========================================================================
print("Loading C++ Geometry Kernel...")
from pythonnet import load
load("coreclr")
import clr
import System
from System import Single, Boolean, String
from System.Runtime.InteropServices import NativeLibrary

# --- THE MISSING MATH IMPORTS ---
clr.AddReference("System.Numerics")
from System.Numerics import Vector3
# --------------------------------

root_dir = r"D:\nyron 2"
# ... (the rest of your code remains exactly the same)
print("--------------------------------------------------")
print("🚀 NOYRON 2.0: FULL SYSTEM INITIALIZATION")
print("--------------------------------------------------")

# =========================================================================
# 1. WAKE UP THE GEOMETRY ENGINE (The Muscles)
# =========================================================================
print("Loading C++ Geometry Kernel...")
load("coreclr")
import clr
import System
from System import Single, Boolean, String
from System.Runtime.InteropServices import NativeLibrary

root_dir = r"D:\nyron 2"
dll_folder = os.path.join(root_dir, "PicoGK_Engine", "bin", "Release", "net9.0")
dll_path = os.path.join(dll_folder, "PicoGK.dll")
engine_path = os.path.join(dll_folder, "picogk.1.7.dll")
workspace_dir = os.path.join(root_dir, "noyron_workspace")

# Inject C++ into .NET
NativeLibrary.Load(engine_path)
clr.AddReference(dll_path)
import PicoGK

# =========================================================================
# 2. TRAIN THE AI (The Brain)
# =========================================================================
print("Loading Labeled Physics Data...")
csv_path = os.path.join(workspace_dir, "labeled_dataset.csv")
df = pd.read_csv(csv_path)

X = df[['Wall_Thickness_mm', 'Num_Holes']].values
y = df[['Max_Deflection_mm']].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train_t = torch.FloatTensor(X_scaled)
y_train_t = torch.FloatTensor(y_scaled)

class NoyronBrain(nn.Module):
    def __init__(self):
        super(NoyronBrain, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.output(x)

model = NoyronBrain()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training Surrogate Physics Model (500 Epochs)...")
for epoch in range(500):
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =========================================================================
# 3. AI GENERATIVE DESIGN LOOP
# =========================================================================
print("\n--------------------------------------------------")
print("🧠 MISSION: Find lightest boom with < 4.8mm deflection")
print("--------------------------------------------------")

model.eval()
best_design = None
best_mass_score = float('inf')

# Search through hundreds of possibilities in milliseconds
for thickness in np.arange(2.0, 6.1, 0.1):
    for holes in range(0, 11):

        X_eval = scaler_X.transform([[thickness, holes]])
        X_tensor = torch.FloatTensor(X_eval)

        with torch.no_grad():
            pred_scaled = model(X_tensor)
            pred_defl = scaler_y.inverse_transform(pred_scaled.numpy())[0][0]

        if pred_defl <= 4.8:
            # Heuristic: Thinner walls and more holes = lighter
            mass_score = thickness - (holes * 0.15)

            if mass_score < best_mass_score:
                best_mass_score = mass_score
                best_design = (round(thickness, 1), holes, pred_defl)

opt_thick, opt_holes, opt_defl = best_design

print("🏆 OPTIMAL DESIGN FOUND!")
print(f"Wall Thickness : {opt_thick} mm")
print(f"Hole Count     : {opt_holes}")
print(f"Predicted Bend : {opt_defl:.2f} mm")

# =========================================================================
# 4. AUTONOMOUS MANUFACTURE
# =========================================================================
print("\nTransmitting coordinates to C++ Kernel for manufacturing...")
try:
    boom_length = 150.0
    outer_radius = 20.0
    hole_radius = 8.0

    lat_stock = PicoGK.Lattice()
    lat_stock.AddBeam(
        Vector3(0.0, 0.0, Single(-(boom_length/2))),
        Vector3(0.0, 0.0, Single((boom_length/2))),
        Single(outer_radius), Single(outer_radius), Boolean(False)
    )
    vox_part = PicoGK.Voxels(lat_stock)

    lat_core = PicoGK.Lattice()
    inner_radius = outer_radius - opt_thick
    lat_core.AddBeam(
        Vector3(0.0, 0.0, Single(-(boom_length/2 + 10))),
        Vector3(0.0, 0.0, Single((boom_length/2 + 10))),
        Single(inner_radius), Single(inner_radius), Boolean(False)
    )
    vox_part.BoolSubtract(PicoGK.Voxels(lat_core))

    if opt_holes > 0:
        lat_holes = PicoGK.Lattice()
        spacing = boom_length / (opt_holes + 1)
        start_z = -(boom_length / 2) + spacing

        for h in range(opt_holes):
            z_pos = start_z + (h * spacing)
            lat_holes.AddBeam(
                Vector3(-40.0, 0.0, Single(z_pos)),
                Vector3(40.0, 0.0, Single(z_pos)),
                Single(hole_radius), Single(hole_radius), Boolean(False)
            )
        vox_part.BoolSubtract(PicoGK.Voxels(lat_holes))

    final_stl = os.path.join(workspace_dir, "Noyron_AI_Optimized_Boom.stl")
    PicoGK.Mesh(vox_part).SaveToStlFile(String(final_stl))

    print(f"✅ AI Manufacturing Complete! Final STL saved to workspace.")

except Exception as e:
    print(f"❌ Manufacturing Error: {e}")