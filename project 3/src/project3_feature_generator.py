import sys
import nibabel as nib

def getFeatures(imagePath):
    img = nib.load(imagePath)
    imgData = img.get_data()
    Nx, Ny, Nz, Nt = imgData.shape

    features = [[[[0 for b in range(bucketsNo)] for z in range(zPartitions)] for y in range(yPartitions)] for x in range(xPartitions)]
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                if x >= borders[0] and x <= borders[1] and y >= borders[2] and y <= borders[3] and z >= borders[4] and z <= borders[5]:
                    xPartition = (x - borders[0])/xSize
                    yPartition = (y - borders[2])/ySize
                    zPartition = (z - borders[4])/zSize
                    if xPartition < xPartitions and yPartition < yPartitions and zPartition < zPartitions:
                        value = imgData[x, y, z, 0]
                        bucket = (value - minValue)/bucketSize
                        if value >= minValue and value <= maxValue and bucket < bucketsNo:
                            features[xPartition][yPartition][zPartition][bucket] += 1

    flatFeatures = [value for xPartition in features for yPartition in xPartition for zPartition in yPartition for value in zPartition]
    return flatFeatures


if len(sys.argv) != 12:
    print "Usage: python {} <images_directory> train|test <image_start> <image_end> <borders> <x_partitions> <y_partitions> <z_partitions> <buckets_no> <min_value> <max_value>".format(sys.argv[0])
    exit(0)

imagesPath = sys.argv[1]
dataset = sys.argv[2]
imageStart = int(sys.argv[3])
imagesEnd = int(sys.argv[4])
borders = [int(i) for i in sys.argv[5].split(',')]
xPartitions = int(sys.argv[6])
yPartitions = int(sys.argv[7])
zPartitions = int(sys.argv[8])
bucketsNo = int(sys.argv[9])
minValue = int(sys.argv[10])
maxValue = int(sys.argv[11])

xSize = (borders[1] - borders[0])/xPartitions
ySize = (borders[3] - borders[2])/yPartitions
zSize = (borders[5] - borders[4])/zPartitions
bucketSize = (maxValue - minValue)/bucketsNo

for i in range(imageStart, imagesEnd + 1):
    print getFeatures("{}/{}_{}.nii".format(imagesPath, dataset, i))

