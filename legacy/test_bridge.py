import clr
import os

# 1. Update this to the FOLDER, not the file
# Change: Remove "\PicoGK.dll" from the end of your string
dll_path = r"D:\nyron 2\PicoGK\bin\Release\net9.0"

# Now this will correctly point to D:\nyron 2\PicoGK\bin\Release\net9.0\PicoGK.dll
clr.AddReference(os.path.join(dll_path, "PicoGK.dll"))

try:
    import PicoGK
    print("✅ Bridge Successful! Noyron 2.0 can now speak to PicoGK.")
except Exception as e:
    print(f"❌ Bridge Failed: {e}")