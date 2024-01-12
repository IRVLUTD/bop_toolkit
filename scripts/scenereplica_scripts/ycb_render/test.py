import CppYCBRenderer
from get_available_devices import *

if __name__ == "__main__":

    width = 640
    height = 480
    gpu_id = 0
    r = CppYCBRenderer.CppYCBRenderer(width, height, get_available_devices()[gpu_id])
    print("hello0")
    r.init()
    print('finished')