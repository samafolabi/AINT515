#include "opencv2/opencv.hpp"
#include "opencv2/photo/cuda.hpp"

using namespace cv;
using namespace std;

int main() {
    // Read and open video
    string path = "C:/Users/emper/OneDrive/Desktop/V1.avi";
    VideoCapture cap(path);
    if ( !cap.isOpened() )
    {
        cout << "Cannot open video" << endl;
        return -1;
    }

    // low and high hue values
    int lowH = 0;
    int highH = 179;

    // peak detection values (bins, value)
    int b = 0, b1 = 0, b2= 0, b1c = 0, b2c = 0;

    // boolean for detecting if first frame
    bool first = false;

    // values for counting num of droplets
    int num = 0, prevNum = 0, countdown = 5;

    // pause and first frame for droplet count booleans
    bool firstCount = false, pause = false;


    // for first frame

    // read frame
    Mat img;
    cap.read(img);
    if (img.empty()) {
        cout << "Video has finished" << endl;
        return 0;
    }

    // convert image to HSV
    Mat imgHSV;
    vector<Mat> hsvSplit;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);
    split(imgHSV, hsvSplit);

    // settings for histogram
    int histSize = 90;
    float range[] = {0, 180};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;

    // calculate histogram
    Mat hist;
    calcHist(&hsvSplit[0], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    Mat histImg(400,512, CV_8UC3, Scalar(0,0,0));

    // blur histogram for smoothness
    GaussianBlur(hist, hist, Size(11,11), 0,0,BORDER_REPLICATE);

    //  determine peaks of histogram
    for (int i = 1; i < histSize; i++) {
        // gets the previous, current, and next values
        int prev = cvRound(hist.at<float>(i-1)),
                cur = cvRound(hist.at<float>(i)),
                next = cvRound(hist.at<float>(i+1));

        // if the current value is greater than the previous and next
        if (cur > prev && cur > next) {
            if (cur > b1c) { // if it is the highest overall
                b2 = b1;
                b1 = i;
                b2c = b1c;
                b1c = cur;
            } else if (cur > b2c) { // if it is the second highest
                b2 = i;
                b2c = cur;
            }
        }
    }

    // if bin 2 is close to black, use the other bin
    b = b2 < 15 ? b1 : b2;


    //main loop
    while (true)  {
        if (!first) {
            // read frame
            cap.read(img);
            if (img.empty()) {
                cout << "Video has finished" << endl;
                break;
            }

            // convert image to HSV
            cvtColor(img, imgHSV, COLOR_BGR2HSV);
            split(imgHSV, hsvSplit);
        }

        first = true;


        // DROPLET DETECTION

        // sets range for hue detection
        lowH = b - 10;
        highH = b + 10;

        // equalises value of image
        equalizeHist(hsvSplit[2],hsvSplit[2]);
        merge(hsvSplit,imgHSV);

        // thresholds the image
        Mat imgThresholded;
        inRange(imgHSV, Scalar(lowH, 0,0), Scalar(highH, 255, 255), imgThresholded);

        // clean up noise and perform edge detection
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
        fastNlMeansDenoising(imgThresholded, imgThresholded, 10.0, 21, 21);
        Mat edges;
        erode(imgThresholded, edges, getStructuringElement(MORPH_ELLIPSE,
        Size(5,5)));
        dilate(edges, edges, getStructuringElement(MORPH_ELLIPSE,
        Size(5,5)));
        Canny(edges, edges, 50, 100);

        // create final image and add droplet edges
        Mat final_img;
        img.copyTo(final_img);
        cvtColor(edges, edges, CV_GRAY2BGR);
        final_img += edges;


        // BLOB DETECTION

        // blob detector parameters
        SimpleBlobDetector::Params params;
        params.minThreshold = 0;
        params.maxThreshold = 255;
        params.filterByArea = true;
        params.minArea = 100;
        params.filterByCircularity = true;
        params.minCircularity = 0.1;
        params.filterByConvexity = true;
        params.minConvexity = 0.07;
        params.filterByInertia = true;
        params.minInertiaRatio = 0.01;

        // for blobs
        vector<KeyPoint> keypoints;

        // makes image easier for blob detector
        bitwise_not(imgThresholded, imgThresholded);

        // detect blobs
        #if CV_MAJOR_VERSION < 3 // OpenCV 2
            SimpleBlobDetector detector(params);
            detector.detect( imgThresholded, keypoints);
        #else
            Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
            detector->detect( imgThresholded, keypoints);
        #endif

        // goes through all the detected droplets
        for (uint i = 0; i < keypoints.size(); i++) {
            // draws marker in the centre of mass
            drawMarker(final_img, keypoints[i].pt, Scalar(0,0,255));


            // COUNTING DROPLETS

            // if it is the first frame, it checks if the droplet is left of the needle
            if (keypoints[i].pt.x <= 512 && !firstCount) {
                num += 1;
            }

            // if it is not the first frame, and not paused,
            // it checks if droplet is passing the center of the screen
            // if it is, it increments and waits for the droplet to pass
            if (!pause && firstCount && (keypoints[i].pt.x >= 512 && keypoints[i].pt.x <= 522)) {
                num += 1;
                pause = true;
            }
        }

        // first frame is over
        firstCount = true;

        // counts down for the pause
        if (pause) {
            countdown--;
            if (countdown == 0) {
                pause = false;
                countdown = 5;
            }
        }

        // updates number on screen
        if (num != prevNum) {
            prevNum = num;
        }
        putText(final_img, to_string(num), Point(30, 40), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0,0,255),2);



        // OUTER WEAP

        // converts image to grayscale and applies Hough Circle Transform
        Mat imgGray;
        vector<Vec3f> circles;
        cvtColor(img, imgGray, COLOR_BGR2GRAY);
        HoughCircles(imgGray, circles, HOUGH_GRADIENT, 1,100,100,30,25 ,35);

        Mat img2;
        img.copyTo(img2);
        Mat mask = Mat(img.size(), img.type(), 0.0);

        // creates circles on mask where the outer wrap is
        for (uint i = 0; i < circles.size(); i++) {
            circle(mask, Point(circles[i][0],circles[i][1]), circles[i][2], Scalar(255,255,255),-1);
        }

        // masks out the background
        bitwise_and(img, mask, img2);

        // determines edges of wraps, changes the colour, then adds it to image
        Mat edges2;
        Canny(mask, edges2, 50, 100);
        cvtColor(edges2, edges2, CV_GRAY2BGR);
        edges2 = edges2.mul(cv::Scalar(255, 255, 0), 1);
        final_img += edges2;



        // Show images
        imshow("Original", img);
        imshow("keypoints", final_img );

        // key escape
        char key = (char) waitKey(300);
        if(key == 27)
            break;
      }

    return 0;
}
