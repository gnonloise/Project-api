import cv2
from ultralytics import YOLO
import threading

#define the video files for the trackers

video_file01 = 'ultralytics/test.mp4'
video_file02 = 0 #webcam path (index)
#video_file02 = 'wiada.mp4'
#video_file02 = 'video.mp4'

#Load the YOLO8 models 
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n.pt')

def run_tracker_in_thread(filename,model,file_index):
    video = cv2.VideoCapture(filename) #read video file 
    while True :
        ret,frame = video.read()
        if not ret :
            break
        else :
            # track objects in the frame if available
            results = model.track(frame,persist=True)
            res_plotted = results[0].plot()
            cv2.imshow(f'Tracking_{file_index}',res_plotted)
            key = cv2.waitKey(1)
            if key==ord('q'):
                break
    video.release()

tracker1 = threading.Thread(target=run_tracker_in_thread, args=(video_file01, model1, 1), daemon=True)
tracker2 = threading.Thread(target=run_tracker_in_thread, args=(video_file02, model2, 2), daemon=True)

# Start the tracker thread
tracker1.start()

# Start thread that runs video files
tracker2.start()

#Wait for the tracker thread to finish
# tracker thread 1
tracker1.join()
# tracker thread 2
tracker2.join()

# Clean up and close windows
cv2.destroyAllWindows()