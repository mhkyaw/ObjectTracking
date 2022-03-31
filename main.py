
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import struct


###############################################################################

#Main function

def main():

    #Check user input
    if len(sys.argv) != 2:
        print("Please type in the following format into the command line: python project1.py video_name.mp4")
        sys.exit(1)

    #If user input correct, load the video and carry out the rest of the program
    else:
        filename = sys.argv[1]
        writevideo(filename, loadvideo(filename))


###############################################################################

#Following code adapted from capture.py to load a video

def loadvideo(filename):

    global capture
    capture = cv2.VideoCapture(filename)

    # Bail if error.
    if not capture or not capture.isOpened():
        print('Error opening video capture!')
        sys.exit(1)

    # Fetch the first frame and bail if none.
    ok, frame = capture.read()

    if not ok or frame is None:
        print('No frames in video')
        sys.exit(1)

    w = frame.shape[1]
    h = frame.shape[0]

    return tmpavg(capture, h, w)

###############################################################################

#Temporal averaging

def tmpavg(capture, h, w):

    #create accumulator variable to sum all frames
    accum = np.zeros((h, w, 3), dtype=np.float32)

    #get number of frames in the video
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    #initialize counter variable
    cnt = 0

    #add all the frames in the video
    for i in range(length-1):
        ok, frame = capture.read()
        if ok:
            accum += frame
            cnt += 1

    #get average of all the frames
    img_ref = np.clip(accum/cnt, 0, 255).astype(np.uint8)

    return img_ref

###############################################################################

#Background subtraction from reference frame

def bgsub(reference, frame):
    #intensity thresholding
    mask = cv2.absdiff(frame, reference).max(axis=2) > 65

    #rescale from [0,1] to [0,255]
    newframe = 255*mask.astype(np.uint8)

    cv2.namedWindow("Before operators")
    cv2.imshow("Before operators", newframe)

    #Perform opening, erosion to get rid of salt noise, then dilation to connect broken pieces of the object

    #Creating kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    newframe = cv2.erode(newframe,kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    newframe = cv2.dilate(newframe,kernel)

    return newframe


###############################################################################

#Following code adapted from capture.py to output video

def writevideo(filename, reference):
    capture = cv2.VideoCapture(filename)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    w = reference.shape[1]
    h = reference.shape[0]

    fps = 30

    fourcc, ext = (cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 'mp4')

    filename = 'captured.'+ext

    ok, frame = capture.read()

    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    if not writer:
        print('Error opening writer')
    else:
        print('Opened', filename, 'for output.')
        writer.write(frame)

    i = 0

    #initialize position lists
    one_x = []
    one_y = []
    two_x = []
    two_y = []

    #Loop movie until the end
    while i<length-1:

        ok, frame = capture.read()

        if ok:
            newframe = bgsub(reference, frame)

        else:
            break

        #All the positions of the objects
        conlist = regions(newframe)

        newframe = cv2.cvtColor(newframe, cv2.COLOR_GRAY2RGB)


        #Tracking one object
        if len(conlist) == 1:
            one_x.append(conlist[0][0])
            one_y.append(conlist[0][1])

            #Label object on video
            cv2.putText(newframe, "Object 1", (one_x[i]-40, one_y[i]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['magenta'], 1)

            #Draw tiny tophat on object 1
            newframe[one_y[i]-5:one_y[i]-2, one_x[i]-12:one_x[i]+15] = (200,0,150)
            newframe[one_y[i]-20:one_y[i]-2, one_x[i]-6:one_x[i]+9] = (200,0,150)

            #Draw eyes on object 1
            newframe[one_y[i]-1:one_y[i]+2, one_x[i]-4:one_x[i]-2] = (0,0,0)
            newframe[one_y[i]-1:one_y[i]+2, one_x[i]+2:one_x[i]+4] = (0,0,0)

        #Tracking two objects
        elif len(conlist)>1:
            #calculate pixel distance from first centroid to previous centroid

            #first iteration, set a reference set of points for object 1 and object 2.
            if i == 0:
                one_x.append(conlist[0][0])
                one_y.append(conlist[0][1])

                two_x.append(conlist[1][0])
                two_y.append(conlist[1][1])

            #subsequent iterations, calculate distance of centroids from previous centroids
            else:
                dist_one = np.sqrt((conlist[0][0]-one_x[i-1])**2 + (conlist[0][1]-one_y[i-1])**2)
                dist_two = np.sqrt((conlist[0][0]-two_x[i-1])**2 + (conlist[0][1]-two_y[i-1])**2)

                #if it's farther away from the second object in the next frame, then this point belongs to the first object
                if dist_one < dist_two:
                    one_x.append(conlist[0][0])
                    one_y.append(conlist[0][1])

                    two_x.append(conlist[1][0])
                    two_y.append(conlist[1][1])
                #if it's farther away from the first object in the next frame, then this point belongs to the second object
                else:
                    two_x.append(conlist[0][0])
                    two_y.append(conlist[0][1])

                    one_x.append(conlist[1][0])
                    one_y.append(conlist[1][1])

                #Label each object on the video
                cv2.putText(newframe, "Object 1", (one_x[i]-40, one_y[i]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['magenta'], 1)
                cv2.putText(newframe, "Object 2", (two_x[i]-40, two_y[i]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)

                #Draw tiny tophat on object 1
                newframe[one_y[i]-5:one_y[i]-2, one_x[i]-12:one_x[i]+15] = (200,0,150)
                newframe[one_y[i]-20:one_y[i]-2, one_x[i]-6:one_x[i]+9] = (200,0,150)

                #Draw eyes on object 1
                newframe[one_y[i]-1:one_y[i]+2, one_x[i]-4:one_x[i]-2] = (0,0,0)
                newframe[one_y[i]-1:one_y[i]+2, one_x[i]+2:one_x[i]+4] = (0,0,0)

                #Draw tiny tophat on object 2
                newframe[two_y[i]-5:two_y[i]-2, two_x[i]-12:two_x[i]+15] = (200,255,0)
                newframe[two_y[i]-20:two_y[i]-2, two_x[i]-6:two_x[i]+9] = (200,255,0)

                #Draw eyes on object 2
                newframe[two_y[i]-1:two_y[i]+2, two_x[i]-4:two_x[i]-2] = (0,0,0)
                newframe[two_y[i]-1:two_y[i]+2, two_x[i]+2:two_x[i]+4] = (0,0,0)



        cv2.namedWindow("After operators")
        cv2.imshow("After operators", newframe)
        cv2.moveWindow('Before operators', 0, 0)
        cv2.moveWindow('After operators', w, 0)
        #Wait 25ms so video doesn't play too fast
        k = cv2.waitKey(25)
        #Check for ESC hit:
        if k & 0xFF == 27:
            break

        i+=1

    #Plot the points onto a graph
    plt.title("Orbit trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(one_x ,one_y,'r-', label = "Object 1 path")

    if len(conlist)>1:
        plt.plot(two_x, two_y, 'b-', label = "Object 2 path")

    #invert the y axis to show position in terms of the video
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

    capture.release()
    cv2.destroyAllWindows()

###############################################################################

#adapted from regions.py
def regions(binary):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conlist = []

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        conlist.append([cX, cY])

    conlist = np.array(conlist)
    return conlist

###############################################################################
#List of colors for annotating
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255),
          'yellow': (0, 255, 255),'magenta': (255, 0, 255),
          'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125),
          'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}


if __name__ == '__main__':
    main()
