
//if io-mode=2 supported (newer camera), not resized on input
// std::string gst_cam1 = "v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
// std::string gst_cam2 = "v4l2src device=/dev/video1 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";

//if io-mode=2 not supported (older/worse camera), not resized on input
// inline std::string gst_cam1 = "v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
// inline std::string gst_cam2 = "v4l2src device=/dev/video2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";

//30 fps resized to 960x540
// inline std::string gst_cam1 = "v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, width=960, height=544, format=BGRx ! videorate ! video/x-raw, framerate=16/1 ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
// inline std::string gst_cam2 = "v4l2src device=/dev/video2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, width=960, height=544, format=BGRx ! videorate ! video/x-raw, framerate=16/1 ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";

//10 fps resized to 960x540
// inline std::string gst_cam1 = "v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, width=960, height=544, format=BGRx ! videorate ! video/x-raw, framerate=10/1 ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
// inline std::string gst_cam2 = "v4l2src device=/dev/video2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, width=960, height=544, format=BGRx ! videorate ! video/x-raw, framerate=10/1 ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";

//13 fps resized to 960x540
inline std::string gst_cam1 = "v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, width=960, height=544, format=BGRx ! videorate ! video/x-raw, framerate=13/1 ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
inline std::string gst_cam2 = "v4l2src device=/dev/video2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, width=960, height=544, format=BGRx ! videorate ! video/x-raw, framerate=13/1 ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";


//calibration camera settings, 1920 x 1080
inline std::string gst_calib1 = "v4l2src device=/dev/video0 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";
inline std::string gst_calib2 = "v4l2src device=/dev/video2 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink name=appsink drop=true sync=false";