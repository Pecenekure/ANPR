import cv2 as cv
import compare
import cProfile, pstats, io





if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    cap = cv.VideoCapture("videos/IMG_4432.MP4")
    #cap = cv.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open video file")

    counter = 0
    while(cap.isOpened()):
        counter += 1
        ret, frame = cap.read()
        if ((counter%10)==0): # video z iPhonu ma 30fps takze hodnota 15 znamena kontrolu 2x za sekundu
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            img = frame

            compare.performRecognition(gray,img)

            #cv.imshow('frame', gray)
            #time.sleep(1)
        if(cv.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv.destroyAllWindows()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

