import subprocess

infoDict = {}
exiftool_path = 'exiftool' 
img_path = 'raw_data/1.png'

def print_metadata(img_path):
    # use Exif tool to get the metadata
    process = subprocess.Popen([exiftool_path, img_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) 
    
    # get the tags in dict
    for tag in process.stdout:
        line = tag.strip().split(':')
        infoDict[line[0].strip()] = line[-1].strip()

    for k,v in infoDict.items():
        print(k,':', v)

img_paths = [
    'raw_data/1.png',
    'raw_data/2.png',
    # 'raw_data/3.png',
]

for img_path in img_paths:
    print_metadata(img_path)
    print()