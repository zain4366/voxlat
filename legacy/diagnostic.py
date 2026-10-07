import ctypes
import os
import sys

print("--- PURE C++ DIAGNOSTIC ---")

# The exact folder where the engine lives
dll_folder = r"D:\nyron 2\PicoGK_Engine\bin\Release\net9.0"
engine_path = os.path.join(dll_folder, "picogk.1.7.dll")

if not os.path.exists(engine_path):
    print(f"❌ Cannot find the C++ engine at: {engine_path}")
    sys.exit()

# Tell Windows to look here for dependencies
if sys.platform == 'win32':
    os.add_dll_directory(dll_folder)

try:
    print(f"Attempting to load {engine_path} directly into memory...")
    # This bypasses C# entirely and directly tests the C++ file
    ctypes.CDLL(engine_path)
    print("✅ SUCCESS: The C++ engine loaded perfectly.")
    print("If you see this, the issue is with .NET, not the C++ file.")

except Exception as e:
    print("\n❌ CRITICAL FAILURE: The C++ engine refused to load.")
    print(f"Error Details: {e}")
    print("\nThis confirms a missing dependency (like a missing DLL file sitting next to it).")