import os, glob
d = os.path.dirname(os.path.dirname(os.path.realpath("legged_gym/__init__.py")))
print("LEGGED_GYM_ROOT_DIR =", d)
print("a1 urdf:", glob.glob(d + "/resources/robots/a1/**/*.urdf", recursive=True)[:2])
print("g1 urdf:", glob.glob(d + "/resources/robots/g1*/**/*.urdf", recursive=True)[:2])
