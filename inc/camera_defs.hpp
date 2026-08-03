
cv::VideoCapture cap;
//if io-mode=2 supported (newer camera)
// std::string gst_cam1 = "v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
// std::string gst_cam2 = "v4l2src device=/dev/video1 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
//if io-mode=2 not supported (older/worse camera)
std::string gst_cam1 = "v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
std::string gst_cam2 = "v4l2src device=/dev/video2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";