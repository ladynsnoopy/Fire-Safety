import os
import time
#os.system('cmd /c "darknet.exe detector demo data/obj.data.txt yolo-obj.cfg yolo-obj_final.weights videos/Object_Soldering_RGB_17.avi -dont_show > finalstuff/results2.txt"')
videosList = os.listdir("videos")
original = "darknet.exe detector demo data/obj.data.txt yolo-obj.cfg yolo-obj_final.weights videos/name -dont_show > finalstuff/resultsfile"
# for video in videosList[0:31]:
#     print(video)
defaultString = original
counter = 0
for video in videosList[0:31]:
    result = video[:-3] + 'txt'
    defaultString = defaultString.replace('name',str(video))
    defaultString = defaultString.replace('resultsfile',str(result))
    print(str(defaultString))
    os.system('cmd /c"'+defaultString+'"')
    print('counter: '+ str(counter))
    counter += 1
    defaultString = original
    time.sleep(40)
