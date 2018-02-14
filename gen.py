import cv2
 
#获得视频的格式
videoCapture = cv2.VideoCapture('source.mp4')
  
#获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
   
#指定写视频的格式, I420-avi, MJPG-mp4
videoWriter = cv2.VideoWriter('oto_other.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
   
#读帧
success, frame = videoCapture.read()
    
c=0
while success :
    #cv2.imshow('Video', frame) #显示
    #cv2.waitKey(1000/int(fps)) #延迟
    c+=1
    print (c)
    videoWriter.write(frame) #写视频帧
    success, frame = videoCapture.read() #获取下一帧

print ('done')
