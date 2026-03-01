import cv2 #cv2 is OpenCV's Python library needed for webcam, detection, and drawing

def main(): #define main method
    # haarcascades is a pre-trained face detection model based on classical computer vision (not deep learning)
    # cv2.data.haarcascades is the path to the folder containing pre-trained XML models that OPENCV has installed
    # xml file is good for storing and transporting data in code that is both human-readable and machine-readable
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" #xml is a pre-trained model that knows face patterns
    face_cascade = cv2.CascadeClassifier(cascade_path) # creates a detector object, reads xml file and loads a pre-trained face detection model from a file into memory so it can use it to detect faces

    if face_cascade.empty(): #if the detection model isn't loaded correctly, prints an error and exits function
        print("Error: could not load face cascade")
        return 
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #starts the video capture, 0 means it is the default camera
    # cap is the video capture object attached to the camera in usage
    #CAP_DSHOW means DirectShow (a Windows multimedia framework) means, instead of OpenCV choosing a backend automatically, DirectShow API talks to the camera making it more reliable
    

    if not cap.isOpened():
        #checks if webcam is open, if not, prints an error and exits function
        print("Error: could not open webcam")
        return
    
    print("Press 'q' to quit.") #prints a solely instructional message that you can quit by pressing q which will be completed later

    while True:
        ret, frame = cap.read() #cap.read() returns ret (boolean, whether or not frame capture worked) and frame (a NumPy array containing all pixels from the camera at that moment)
        if not ret:
            print("Error: failed to read frame")
            break
            #if red (whether or not frame capture worked) is false, then break and break from loop
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # makes each frame grayscale b/c Haar face detection works faster on grayscale
        faces = face_cascade.detectMultiScale( #detects faces in that frame, looks like this: [(x, y, w, h), (x, y, w, h), ...]
            image=gray, #input image to run detection on
            scaleFactor=1.1, #how much image size changes each time the algorithm rescales the image, the higher -> more accurate, slower
            minNeighbors=5, #how many nearby detections must agree before OpenCV keeps a face box, the higher -> fewer false positives, more detections
            flags=0, #controlled detection modes, don't need to touch this as it is outdated
            minSize=(60, 60), #smallest face size that OpenCV will try to detect, the higher -> more it ignores smaller objects
            maxSize=None #not very important, biggest face size that OpenCV will try to detect, the higher -> more it ignores bigger objects
        )
        for (x, y, w, h) in faces: #gets variables from all elements of each element of faces, x is left coord., y is right coord., w is width, h is height) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            # rectangle = (image, top left corner, bottom right corner, color, line thinkness)
            # (0, 255, 0) is green
            cv2.putText( # used for debugging to confirm detections
                frame, #image
                f"Faces detected: {len(faces)}", #text
                (10, 30), #origin
                cv2.FONT_HERSHEY_SIMPLEX, #font style
                0.9, #font scale (size of text)
                (0, 255, 0), #color (in this case, green)
                thickness=2, #thickness of text lines
                lineType=None, #how smooth the text edges are
                bottomLeftOrigin=None #by defaualt this is false, where false -> origin is top left and true -> origin is bottom left
            )
        cv2.imshow("Face Detection", frame) #imshow shows frames by displaying the video window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            #allows quitting, waitkey(1) waits 1 millisecond before returning keyboard input each loop 
            # 0xFF masks waitKey(1) to only look at lowest 8 bytes
            # and if q is pressed (ord("q") equivalent to 113 as ASCII integer value), it breaks loop
            break
    cap.release() #closes camera to free resources
    cv2.destroyAllWindows() #closes windows to free resources

if __name__ == "__main__": 
    #entry point so script runs
    #without this, main doesn't run and nothing happens
    main()

#Tune if detection is bad:

#Too many false boxes -> increase minNeighbors (e.g. 6 or 7)
#Missing faces -> lower minNeighbors or minSize
#Performance issues -> resize frame smaller before detection
        
            





    



