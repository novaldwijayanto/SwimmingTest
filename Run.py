from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
from threading import Thread
import ray
from multiprocessing import Process
import sys
from time import sleep
import asyncio

t0 = time.time()
import concurrent.futures

def run():
    t1 = Process(target=Object_Detection(25, 500,400,750))
    t2 = Process(target=Object_Detection2(25, 500,400,750))
    #ray.get([Object_Detection(25,500).remote(), Object_Detection(25,500).remote()])
    t1.start()
    t2.start()
    # t1.join()
    # t2.join()
    # #ret1, ret2 = ray.get([Object_Detection(25,500), Object_Detection(25,500)])
    #
    # # threads = [
    # #     Thread(target=self.Object_Detection(25, 500)),
    # #     Thread(target=self.Object_Detection2(25, 500))
    # # ]

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     if __name__ == '__main__':
    #         f1 = executor.submit(Object_Detection(25, 500, 400, 750))
    #         f2 = executor.submit(Object_Detection2(25, 500, 400, 750))

    # Object_Detection(25, 500, 400, 750)
    # Object_Detection2(25, 500, 400, 750)

def Object_Detection(Lintasan,ROI,StartXBox,EndXBox):

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", type=str, default="videos/Video1.mp4",
        help="path to optional input video file")
    ap.add_argument("-i2", "--input2", type=str, default="videos/Video2.mp4",
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    # confidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())


    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = VideoStream(src=0).start()
        time.sleep(0.1)

    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])
        vs2 = cv2.VideoCapture(args["input2"])



    writer = None

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    ct2 = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackers2 = []
    trackableObjects = {}
    trackableObjects2 = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    x = []
    empty = []
    empty1 = []
    speed1 = 0
    speed2 = 0
    speed3 = 0
    speed4 = 0
    speed5 = 0
    timeFinished = 0
    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:

        frame = vs.read()



        a = round(time.time()-t0)
        frame = frame[1] if args.get("input", False) else frame



        if args["input"] is not None and frame is None:
            break

        hight, width, _ = frame.shape
        frame = imutils.resize(frame, width=1280)
        frame = frame[0:1000,StartXBox:EndXBox]




        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        if W is None or H is None:
            (H, W) = frame.shape[:2]



        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (W, H), True)

        status = "Waiting"
        rects = []


        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []


            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()


            for i in np.arange(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:

                    idx = int(detections[0, 0, i, 1])

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print("Object detected: ", label)
                    (startX, startY, endX, endY) = box.astype("int");



                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)



        else:

            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom());

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, ROI), (W, ROI), (0, 0, 255), 3)
        #cv2.rectangle(frame, (150, 0), (330, 300),(0, 0, 0), 2)
        cv2.putText(frame,"Finish Line", (0,(ROI + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 100)
        #cv2.putText(frame, "Area", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:

                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)

                to.centroids.append(centroid);cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0,0,0), 2)

                de = endY
                print(de)

                cv2.putText(frame, "time : {:.2f} s".format(timeFinished), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, "speed : {:.2f} m/s".format(speed1), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                if not to.counted:

                    if de > ROI:

                        totalUp += 1
                        speed1 = (Lintasan/round(time.time()-t0))
                        print("finished")
                        timeFinished = round(time.time()-t0)
                        empty.append(totalUp)
                        to.counted = True

            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)


        info2 = [
        (a,"seconds")
        ]


        for (i, (k, v)) in enumerate(info2):
            text = "{} {}".format(k, v)
            cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # Initiate a simple log to save data at end of the day
        if config.Log:
            datetimee = [datetime.datetime.now()]
            d = [datetimee, empty1, empty, x]
            export_data = zip_longest(*d, fillvalue = '')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # show the output frame
        cv2.imshow("Monitor1", frame)
        cv2.imshow("Monitor2", frame)
        cv2.imshow("Monitor3", frame)
        cv2.imshow("Monitor4", frame)
        #sleep(1)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        totalFrames += 1
        fps.update()

        if config.Timer:
            # Automatic timer to stop the live stream. Set to 8 hours (28800s).
            t1 = time.time()
            num_seconds=(t1-t0)
            if num_seconds > 28800:
                break
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("speed : {:.2f} m/s".format(speed1))
    print("time : {:.2f} s".format(timeFinished))
    # if config.Thread:
    #     vs.release()
    # close any open windows
    #cv2.destroyAllWindows()
def Object_Detection2(Lintasan,ROI,StartXBox,EndXBox):

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", type=str, default="videos/Video2.mp4",
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    # confidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())


    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = VideoStream(src=0).start()
        time.sleep(0.5)

    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])



    writer = None

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    ct2 = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackers2 = []
    trackableObjects = {}
    trackableObjects2 = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    x = []
    empty = []
    empty1 = []
    speed1 = 0
    speed2 = 0
    speed3 = 0
    speed4 = 0
    speed5 = 0
    timeFinished = 0
    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:

        frame = vs.read()


        a = round(time.time()-t0)
        frame = frame[1] if args.get("input", False) else frame


        if args["input"] is not None and frame is None:
            break
        hight, width, _ = frame.shape
        frame = imutils.resize(frame, width=1280)
        frame = frame[0:1000,StartXBox:EndXBox]



        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        if W is None or H is None:
            (H, W) = frame.shape[:2]



        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (W, H), True)

        status = "Waiting"
        rects = []


        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []


            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()


            for i in np.arange(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:

                    idx = int(detections[0, 0, i, 1])

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print("Object detected: ", label)
                    (startX, startY, endX, endY) = box.astype("int");



                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)



        else:

            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom());

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, ROI), (W, ROI), (0, 0, 255), 3)
        #cv2.rectangle(frame, (150, 0), (330, 300),(0, 0, 0), 2)
        cv2.putText(frame,"Finish Line", (0,(ROI + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 100)
        #cv2.putText(frame, "Area", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:

                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)

                to.centroids.append(centroid);cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0,0,0), 2)

                de = endY
                print(de)

                cv2.putText(frame, "time : {:.2f} s".format(timeFinished), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, "speed : {:.2f} m/s".format(speed1), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                if not to.counted:

                    if de > ROI:

                        totalUp += 1
                        speed1 = (Lintasan/round(time.time()-t0))
                        print("finished")
                        timeFinished = round(time.time()-t0)
                        empty.append(totalUp)
                        to.counted = True

            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)


        info2 = [
        (a,"seconds")
        ]


        for (i, (k, v)) in enumerate(info2):
            text = "{} {}".format(k, v)
            cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # Initiate a simple log to save data at end of the day
        if config.Log:
            datetimee = [datetime.datetime.now()]
            d = [datetimee, empty1, empty, x]
            export_data = zip_longest(*d, fillvalue = '')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # show the output frame
        cv2.imshow("Monitor2", frame)
        #sleep(0.5)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

        if config.Timer:
            # Automatic timer to stop the live stream. Set to 8 hours (28800s).
            t1 = time.time()
            num_seconds=(t1-t0)
            if num_seconds > 28800:
                break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("speed : {:.2f} m/s".format(speed1))
    print("time : {:.2f} s".format(timeFinished))


    if config.Thread:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()




#learn more about different schedules here: https://pypi.org/project/schedule/
if config.Scheduler:
    ##Runs for every 1 second
    #schedule.every(1).seconds.do(run)
    ##Runs at every day (09:00 am). You can change it.
    schedule.every().day.at("09:00").do(run)

    while 1:
        schedule.run_pending()

else:
    run()
