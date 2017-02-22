import sys
import nibabel as nib

def getPartitions(imagePath):
    img = nib.load(imagePath)
    imgData = img.get_data()
    Nx, Ny, Nz, Nt = imgData.shape

    xSize = Nx/xPartitions
    ySize = Ny/yPartitions
    zSize = Nz/zPartitions
    partitions = [[[0 for z in range(zPartitions)] for y in range(yPartitions)] for x in range(xPartitions)]
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                if x/xSize < xPartitions and y/ySize < yPartitions and z/zSize < zPartitions:
                    partitions[x/xSize][y/ySize][z/zSize] += imgData[x, y, z, 0]

    return partitions

if len(sys.argv) != 7:
    print "Usage: python {} <images_directory> train|test <imagesNo> <xPartitions> <yPartitions> <zPartitions>".format(sys.argv[0])
    exit(0)

imagesPath = sys.argv[1]
dataset = sys.argv[2]
imagesNo = int(sys.argv[3])
xPartitions = int(sys.argv[4])
yPartitions = int(sys.argv[5])
zPartitions = int(sys.argv[6])

for i in range(1, imagesNo + 1):
    print getPartitions("{}/{}_{}.nii".format(imagesPath, dataset, i))

