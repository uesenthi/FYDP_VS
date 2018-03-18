//objectTrackingTutorial.cpp

//Written by  Kyle Hounslow 2013

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")
//, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//IN THE SOFTWARE.

#include <sstream>
#include <string>
#include <iostream>
#include <map>
#include <queue>
#include <fstream>
#include "opencv/cv.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"


using namespace cv;
using namespace std;
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 42;
int H_MAX = 135;
int S_MIN = 0;
int S_MAX = 205;
int V_MIN = 0;
int V_MAX = 21;
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT * FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

//The following global variables are used for detecting the circles

const std::string windowNameCircles = "Hough Circle Detection Demo";
const std::string cannyThresholdTrackbarName = "Canny threshold";
const std::string accumulatorThresholdTrackbarName = "Accumulator Threshold";
const std::string usage = "Usage : tutorial_HoughCircle_Demo <path_to_input_image>\n";

const std::string circleDisplay = "Circle Display";

const int cannyThresholdInitialValue = 100; //best at 139
const int accumulatorThresholdInitialValue = 50; //best at 36
const int maxAccumulatorThreshold = 200;
const int maxCannyThreshold = 255;

RNG rng(12345);
int thresh = 100;
int max_thresh = 255;

//=====================================The Following functions are used for the colour detect==========================================================//
void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed





}
string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}
void createTrackbars() {
	//create window for trackbars


	cv::namedWindow(trackbarWindowName, 0);

	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf_s(TrackbarName, "H_MIN", H_MIN);
	sprintf_s(TrackbarName, "H_MAX", H_MAX);
	sprintf_s(TrackbarName, "S_MIN", S_MIN);
	sprintf_s(TrackbarName, "S_MAX", S_MAX);
	sprintf_s(TrackbarName, "V_MIN", V_MIN);
	sprintf_s(TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH), 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}
void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25>0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25<FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25>0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25<FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}
void morphOps(Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);

}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects<MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				drawObject(x, y, cameraFeed);
			}

		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}

//======================================The following functions are used for the circle detect===========================================================/

void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold)
{
	// will hold the results of the detection
	std::vector<Vec3f> circles;

	// runs the actual detection
	//HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows / 8, cannyThreshold, accumulatorThreshold, 0, 0);
	HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, 1, cannyThreshold, accumulatorThreshold, 0, 0);
	// clone the colour, input image for displaying purposes
	Mat display = src_display.clone();
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		if (circles[i][2] < 14 && circles[i][2] > 9)
		{
			circle(display, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(display, center, radius, Scalar(0, 0, 255), 3, 8, 0);
			//final_circles->push_back(circles[i]);

			//cout << circles[i] << endl;
		}
	}

	// shows the results
	cv::imshow(windowNameCircles, display);
}


vector<Point> contoursConvexHull(vector<vector<Point> > contours)
{
	vector<Point> result;
	vector<Point> pts;
	for (size_t i = 0; i< contours.size(); i++)
		for (size_t j = 0; j< contours[i].size(); j++)
			pts.push_back(contours[i][j]);
	convexHull(pts, result);
	return result;
}

typedef struct t_color_node {
	cv::Mat       mean;       // The mean of this node
	cv::Mat       cov;
	uchar         classid;    // The class ID

	t_color_node  *left;
	t_color_node  *right;
} t_color_node;

cv::Mat get_dominant_palette(std::vector<cv::Vec3b> colors) {
	const int tile_size = 64;
	cv::Mat ret = cv::Mat(tile_size, tile_size*colors.size(), CV_8UC3, cv::Scalar(0));

	for (int i = 0; i<colors.size(); i++) {
		cv::Rect rect(i*tile_size, 0, tile_size, tile_size);
		cv::rectangle(ret, rect, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]), CV_FILLED);
	}

	return ret;
}

std::vector<t_color_node*> get_leaves(t_color_node *root) {
	std::vector<t_color_node*> ret;
	std::queue<t_color_node*> queue;
	queue.push(root);

	while (queue.size() > 0) {
		t_color_node *current = queue.front();
		queue.pop();

		if (current->left && current->right) {
			queue.push(current->left);
			queue.push(current->right);
			continue;
		}

		ret.push_back(current);
	}

	return ret;
}

std::vector<cv::Vec3b> get_dominant_colors(t_color_node *root) {
	std::vector<t_color_node*> leaves = get_leaves(root);
	std::vector<cv::Vec3b> ret;

	for (int i = 0; i<leaves.size(); i++) {
		cv::Mat mean = leaves[i]->mean;
		ret.push_back(cv::Vec3b(mean.at<double>(0)*255.0f,
			mean.at<double>(1)*255.0f,
			mean.at<double>(2)*255.0f));
	}

	return ret;
}

int get_next_classid(t_color_node *root) {
	int maxid = 0;
	std::queue<t_color_node*> queue;
	queue.push(root);

	while (queue.size() > 0) {
		t_color_node* current = queue.front();
		queue.pop();

		if (current->classid > maxid)
			maxid = current->classid;

		if (current->left != NULL)
			queue.push(current->left);

		if (current->right)
			queue.push(current->right);
	}

	return maxid + 1;
}

void get_class_mean_cov(cv::Mat img, cv::Mat classes, t_color_node *node) {
	const int width = img.cols;
	const int height = img.rows;
	const uchar classid = node->classid;

	cv::Mat mean = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
	cv::Mat cov = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));

	// We start out with the average color
	double pixcount = 0;
	for (int y = 0; y<height; y++) {
		cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
		uchar* ptrClass = classes.ptr<uchar>(y);
		for (int x = 0; x<width; x++) {
			if (ptrClass[x] != classid)
				continue;

			cv::Vec3b color = ptr[x];
			cv::Mat scaled = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
			scaled.at<double>(0) = color[0] / 255.0f;
			scaled.at<double>(1) = color[1] / 255.0f;
			scaled.at<double>(2) = color[2] / 255.0f;

			mean += scaled;
			cov = cov + (scaled * scaled.t());

			pixcount++;
		}
	}

	cov = cov - (mean * mean.t()) / pixcount;
	mean = mean / pixcount;

	// The node mean and covariance
	node->mean = mean.clone();
	node->cov = cov.clone();

	return;
}

void partition_class(cv::Mat img, cv::Mat classes, uchar nextid, t_color_node *node) {
	const int width = img.cols;
	const int height = img.rows;
	const int classid = node->classid;

	const uchar newidleft = nextid;
	const uchar newidright = nextid + 1;

	cv::Mat mean = node->mean;
	cv::Mat cov = node->cov;
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(cov, eigenvalues, eigenvectors);

	cv::Mat eig = eigenvectors.row(0);
	cv::Mat comparison_value = eig * mean;

	node->left = new t_color_node();
	node->right = new t_color_node();

	node->left->classid = newidleft;
	node->right->classid = newidright;

	// We start out with the average color
	for (int y = 0; y<height; y++) {
		cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
		uchar* ptrClass = classes.ptr<uchar>(y);
		for (int x = 0; x<width; x++) {
			if (ptrClass[x] != classid)
				continue;

			cv::Vec3b color = ptr[x];
			cv::Mat scaled = cv::Mat(3, 1,
				CV_64FC1,
				cv::Scalar(0));

			scaled.at<double>(0) = color[0] / 255.0f;
			scaled.at<double>(1) = color[1] / 255.0f;
			scaled.at<double>(2) = color[2] / 255.0f;

			cv::Mat this_value = eig * scaled;

			if (this_value.at<double>(0, 0) <= comparison_value.at<double>(0, 0)) {
				ptrClass[x] = newidleft;
			}
			else {
				ptrClass[x] = newidright;
			}
		}
	}
	return;
}

cv::Mat get_quantized_image(cv::Mat classes, t_color_node *root) {
	std::vector<t_color_node*> leaves = get_leaves(root);

	const int height = classes.rows;
	const int width = classes.cols;
	cv::Mat ret(height, width, CV_8UC3, cv::Scalar(0));

	for (int y = 0; y<height; y++) {
		uchar *ptrClass = classes.ptr<uchar>(y);
		cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
		for (int x = 0; x<width; x++) {
			uchar pixel_class = ptrClass[x];
			for (int i = 0; i<leaves.size(); i++) {
				if (leaves[i]->classid == pixel_class) {
					ptr[x] = cv::Vec3b(leaves[i]->mean.at<double>(0) * 255,
						leaves[i]->mean.at<double>(1) * 255,
						leaves[i]->mean.at<double>(2) * 255);
				}
			}
		}
	}

	return ret;
}

cv::Mat get_viewable_image(cv::Mat classes) {
	const int height = classes.rows;
	const int width = classes.cols;

	const int max_color_count = 12;
	cv::Vec3b *palette = new cv::Vec3b[max_color_count];
	palette[0] = cv::Vec3b(0, 0, 0);
	palette[1] = cv::Vec3b(255, 0, 0);
	palette[2] = cv::Vec3b(0, 255, 0);
	palette[3] = cv::Vec3b(0, 0, 255);
	palette[4] = cv::Vec3b(255, 255, 0);
	palette[5] = cv::Vec3b(0, 255, 255);
	palette[6] = cv::Vec3b(255, 0, 255);
	palette[7] = cv::Vec3b(128, 128, 128);
	palette[8] = cv::Vec3b(128, 255, 128);
	palette[9] = cv::Vec3b(32, 32, 32);
	palette[10] = cv::Vec3b(255, 128, 128);
	palette[11] = cv::Vec3b(128, 128, 255);

	cv::Mat ret = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int y = 0; y<height; y++) {
		cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
		uchar *ptrClass = classes.ptr<uchar>(y);
		for (int x = 0; x<width; x++) {
			int color = ptrClass[x];
			if (color >= max_color_count) {
				printf("You should increase the number of predefined colors!\n");
				continue;
			}
			ptr[x] = palette[color];
		}
	}

	return ret;
}

t_color_node* get_max_eigenvalue_node(t_color_node *current) {
	double max_eigen = -1;
	cv::Mat eigenvalues, eigenvectors;

	std::queue<t_color_node*> queue;
	queue.push(current);

	t_color_node *ret = current;
	if (!current->left && !current->right)
		return current;

	while (queue.size() > 0) {
		t_color_node *node = queue.front();
		queue.pop();

		if (node->left && node->right) {
			queue.push(node->left);
			queue.push(node->right);
			continue;
		}

		cv::eigen(node->cov, eigenvalues, eigenvectors);
		double val = eigenvalues.at<double>(0);
		if (val > max_eigen) {
			max_eigen = val;
			ret = node;
		}
	}

	return ret;
}

std::vector<cv::Vec3b> find_dominant_colors(cv::Mat img, int count) {
	const int width = img.cols;
	const int height = img.rows;

	cv::Mat classes = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));
	t_color_node *root = new t_color_node();

	root->classid = 1;
	root->left = NULL;
	root->right = NULL;

	t_color_node *next = root;
	get_class_mean_cov(img, classes, root);
	for (int i = 0; i<count - 1; i++) {
		next = get_max_eigenvalue_node(root);
		partition_class(img, classes, get_next_classid(root), next);
		get_class_mean_cov(img, classes, next->left);
		get_class_mean_cov(img, classes, next->right);
	}

	std::vector<cv::Vec3b> colors = get_dominant_colors(root);

	cv::Mat quantized = get_quantized_image(classes, root);
	cv::Mat viewable = get_viewable_image(classes);
	cv::Mat dom = get_dominant_palette(colors);

	cv::imwrite("./classification.png", viewable);
	cv::imwrite("./quantized.png", quantized);
	cv::imwrite("./palette.png", dom);

	return colors;
}

map<string, Point2f> assign_ball_colour(vector<Point2f> center, vector<float> radius, Mat display, Mat src)
{
	map<string, vector<int>> code;
	map<string, Vec3b> palette;
	map<string, Point2f>balls;

	
	code["black"] = vector<int>{ 44, 133, 0, 256, 0, 74 };
	//code["red"] = Vec3b(30, 40, 171);
	//palette["green"] = Vec3b(133, 122, 9);
	code["purple"] = vector<int>{ 116, 130, 0, 175, 0, 256 };
	code["blue"] = vector<int>{ 98, 119, 119, 228, 0, 256 };
	//palette["orange"] = Vec3b(115, 148, 194);
	code["white"] = vector<int>{ 0, 102, 0, 38, 0, 256 };
	//palette["maroon"] = Vec3b(51, 36, 128);
	code["yellow"] = vector<int>{ 0, 54, 54, 256, 0, 256 };
	


	
	palette["black"] = Vec3b(73, 73, 31);
	//palette["red"] = Vec3b(0, 0, 255);
	//palette["green"] = Vec3b(0, 255, 0);
	palette["blue"] = Vec3b(136, 106, 48);
	//palette["orange"] = Vec3b(0, 127, 255);
	palette["white"] = Vec3b(116, 119, 105);
	palette["purple"] = Vec3b(125, 96, 53);
	palette["yellow"] = Vec3b(93, 129, 146);
	//palette["indigo"] = Vec3b(128, 0, 0);
	

	string color_name;
	Vec3b nearest_color;
	vector<Vec3b> ball_colour;


	bool trackObjects = true;
	bool useMorphOps = true;
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	//matrix storage for HSV image
	Mat HSV;
	//matrix storage for binary threshold image
	Mat threshold;
	//x and y values for the location of the object
	int x = 0, y = 0;
	//create slider bars for HSV filtering
	createTrackbars();
	bool flag = true;


	//Find the average colour of each circle
	for (int i = 0; i < center.size(); i++)
	{
		//if (radius[i] < 10 || radius[i] > 18)
		if (radius[i] >  7 && radius[i] < 4)
			continue;

		cv::circle(display, center[i], 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		cv::circle(display, center[i], radius[i], Scalar(0, 0, 255), 3, 8, 0);
		imshow("Output", display);

		float min_distance = FLT_MAX;
		try {
			//cv::Mat roi = src(cv::Range(center[i].y - radius[i], center[i].y + radius[i] + 1), cv::Range(center[i].x - radius[i], center[i].x + radius[i] + 1));
			cv::Mat roi = src(cv::Range(center[i].y - radius[i] - 1, center[i].y + radius[i] + 1), cv::Range(center[i].x - radius[i] - 1, center[i].x + radius[i] + 1));

			Mat copy = roi.clone();
			imshow("ROI", roi);

			cvtColor(roi, HSV, COLOR_BGR2HSV);
			
			for (map<string, vector<int>>::iterator pal = code.begin(); pal != code.end(); ++pal)
			{
				x = 0;
				y = 0;
				inRange(HSV, Scalar(pal->second[0], pal->second[2], pal->second[4]), Scalar(pal->second[1], pal->second[3], pal->second[5]), threshold);
				if (useMorphOps)
					morphOps(threshold);
				if (trackObjects)
					trackFilteredObject(x, y, threshold, copy);

				if (x > 0 && y > 0) cout << center[i] << " for " << pal->first << endl;
			}


			ball_colour = find_dominant_colors(roi, 1);
		}
		catch (cv::Exception & e)
		{
			continue;
		}
		
		for (map<string, Vec3b>::iterator pal = palette.begin(); pal != palette.end(); ++pal)
		{
			float dist = norm(pal->second, ball_colour[0]);
			if (dist < min_distance)
			{
				nearest_color = pal->second;
				color_name = pal->first;
				min_distance = dist;
			}
		}
		float th_distance = 1000.f;

		//if (min_distance < th_distance)
		if (min_distance < 10.0f)
		{
		//cout << ball_colour[0] << " is similar to " << color_name << endl;
		balls[color_name] = center[i];

		}
		else
		{
		//cout << ball_colour[0] << " is not in the palette" << endl;
		}
		//cout << "Distance with nearest color " << nearest_color << " is " << min_distance << endl;
		//waitKey();

	}
	return balls;
}

//void scale_table(Mat src, vector<Point2f> center, vector<float> radius, Mat display, Point2f left_corner, Point2f right_corner)
Point2f scale_table(Mat src, Point2f center, string colour)
{
	float scale_x, scale_y;
	float ball_x, ball_y;

	//scale_x = (right_corner.x - left_corner.x) / 54.3;
	//scale_y = (right_corner.y - left_corner.y) / 28.7;

	scale_x = 54.3/(src.size().width);
	scale_y = 28.7/(src.size().height);

	ball_x = center.x * scale_x;
	ball_y = center.y * scale_y;

	cout << "center.x is " << ball_x << " and center.y is " << ball_y << endl;
	return Point2f(ball_x, ball_y);

	/*
	for (int i = 0; i < center.size(); i++)
	{
		//cout << "Left corner at " << left_corner.x << " and " << left_corner.y << endl;
		//cout << "Right corner at " << right_corner.x << " and " << right_corner.y << endl;
		//cout << center[i] << " Compared to " << center[i].x << endl;

		//Filter the circles to only those on the table
		if (center[i].x < right_corner.x && center[i].x > left_corner.x)
		{
			if (center[i].y > left_corner.y && center[i].y < right_corner.y)
			{
				if (radius[i] < 10.0f || radius[i] > 18.0f)
				{
					continue;
				}
				cout << "Circle #" << i << " with radius " << radius[i] << endl;
				cv::circle(display, center[i], 3, Scalar(0, 255, 0), -1, 8, 0);
				// circle outline
				cv::circle(display, center[i], radius[i], Scalar(0, 0, 255), 3, 8, 0);
				imshow("Output", display);
				ball_x = (center[i].x - left_corner.x) / scale_x;
				ball_y = (center[i].y - left_corner.y) / scale_y;

			}
		}
	}
	*/
	
}

Mat3b crop_image_table(Mat src, Point2f &left_corner, Point2f &right_corner)
{
	Mat roi3b = src.clone();
	Mat result = src.clone();

	bool trackObjects = true;
	bool useMorphOps = true;
	Mat HSV;
	Mat threshold;

	cvtColor(roi3b, HSV, COLOR_BGR2HSV);
	int leftx, lefty, rightx, righty;
	Rect r;

	//Find left corner
	inRange(HSV, Scalar(167, 182, 119), Scalar(214, 230, 256), threshold);
	morphOps(threshold);
	trackFilteredObject(leftx, lefty, threshold, roi3b);
	imshow("test", threshold);
	imshow("test1", HSV);
	waitKey();

	//Find right corner
	//inRange(HSV, Scalar(18, 35, 86), Scalar(44, 100, 249), threshold);
	inRange(HSV, Scalar(42, 0, 0), Scalar(135, 205, 21), threshold);
	morphOps(threshold);
	trackFilteredObject(rightx, righty, threshold, roi3b);

	//waitKey();
	/*
	try {
		left_corner = Point2f(leftx - 5, lefty - 5);
		right_corner = Point2f(rightx + 5, righty + 5);
		r = Rect(left_corner, right_corner);
	}
	catch (cv::Exception & e)
	{
		r = Rect(130, 130, 500, 250);
	}
	*/
	
	r = Rect(130, 130, 500, 250);
	//r = Rect(348, 503, 131, 136);
	
	Mat3b crop(result(r));
	return crop;
}

/*
int main(int argc, char** argv) {
//some boolean variables for different functionality within this
//program
bool trackObjects = true;
bool useMorphOps = true;
//Matrix to store each frame of the webcam feed
Mat cameraFeed;
//matrix storage for HSV image
Mat HSV;
//matrix storage for binary threshold image
Mat threshold;
//x and y values for the location of the object
int x = 0, y = 0;
//create slider bars for HSV filtering
createTrackbars();
//video capture object to acquire webcam feed




VideoCapture capture;
//open capture object at location zero (default location for webcam)
capture.open(0);
//set height and width of capture frame
capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);



VideoCapture capture;
capture.open(1);
capture.set(CAP_PROP_AUTOFOCUS, 1);
//capture.set(3, 1024);
//capture.set(4, 576);
capture.set(CAP_PROP_BRIGHTNESS, 100);
capture.set(CAP_PROP_SATURATION, 255);
capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
//src = imread(argv[1], 1);
Point2f left_corner;
Point2f right_corner;
Mat src;
Mat roi3b;

char key = 0;
while (key != 'q' && key != 'Q')
{
	capture.read(roi3b);
	imshow("Test", roi3b);
	key = (char)waitKey(10);
}







//Point2f left_corner;
//Point2f right_corner;


//capture.read(cameraFeed);
//cameraFeed = imread(argv[1], IMREAD_COLOR);
//Mat src = imread(argv[1], IMREAD_COLOR);
//roi3b = crop_image_table(src, left_corner, right_corner);
Mat copy;
//mshow(windowName, roi3b);

//start an infinite loop where webcam feed is copied to cameraFeed matrix
//all of our operations will be performed within this loop

while (1) {

//store image to matrix

copy = roi3b.clone();
//convert frame from BGR to HSV colorspace
cvtColor(roi3b, HSV, COLOR_BGR2HSV);
//filter HSV image between values and store filtered image to
//threshold matrix
inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
//perform morphological operations on thresholded image to eliminate noise
//and emphasize the filtered object(s)w
if (useMorphOps)
morphOps(threshold);
//pass in thresholded frame to our object tracking function
//this function will return the x and y coordinates of the
//filtered object
if (trackObjects)
trackFilteredObject(x, y, threshold, copy);

//show frames
imshow(windowName2, threshold);
imshow(windowName, copy);
//imshow(windowName1, HSV);


//delay 30ms so that screen can refresh.
//image will not appear without this waitKey() command
waitKey(30);
}

return 0;
}
*/


int main(int argc, char** argv)
{
	Mat src, srcGray, srcBlur, srcCanny;

	VideoCapture capture;
	capture.open(1);
	capture.set(CAP_PROP_AUTOFOCUS, 1);
	//capture.set(3, 1024);
	//capture.set(4, 576);
	//capture.set(CAP_PROP_BRIGHTNESS, 100);
	//capture.set(CAP_PROP_SATURATION, 255);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	//src = imread(argv[1], 1);
	Point2f left_corner;
	Point2f right_corner;

	char key = 0;
	while (key != 'q' && key != 'Q')
	{
		capture.read(src);
		imshow("Test", src);
		key = (char)waitKey(10);
	}
	cout << src.size().width << " and " << src.size().height << endl;
	imshow("Test window", src);
	waitKey();
	Mat3b roi3b = crop_image_table(src, left_corner, right_corner);

	imshow("Cropped image", roi3b);
	waitKey();
	//Mat roi3b = src.clone();
	Mat display = roi3b.clone();
	Mat table = roi3b.clone();
	Mat colordetect = roi3b.clone();
	//Mat colordetect = src.clone();
	namedWindow("Output", WINDOW_AUTOSIZE);

	cvtColor(roi3b, srcGray, CV_BGR2GRAY);

	blur(srcGray, srcBlur, Size(3, 3));

	Canny(srcBlur, srcCanny, 0, 100, 3, true);

	vector<vector<Point> > contours;

	findContours(srcCanny, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	Mat drawing = Mat::zeros(srcCanny.size(), CV_8UC3);

	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 255);
		drawContours(drawing, contours, i, color, 2);
	}
	vector<vector<Point>> contours_poly (contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		minEnclosingCircle((Mat)contours[i], center[i], radius[i]);
	}

	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(display, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		//if (radius[i] < 18 && radius[i] > 10)
		if (radius[i] <  7 && radius[i] > 4)
		{
			circle(display, center[i], (int)radius[i], color, 2, 8, 0);
			//cout << radius[i] << endl;
			//waitKey();
		}

		cv::imshow("Output", display);


	}
	cv::imshow("Contours", drawing);
	cout << "Size is " << center.size() << endl;


	map<string, Point2f>finalballs = assign_ball_colour(center, radius, display, colordetect);
	ofstream output;
	output.open("colour_coordinates.txt");
	Point2f coordinates;

	for (map<string, Point2f>::iterator pal = finalballs.begin(); pal != finalballs.end(); ++pal)
	{
		cout << "Found " << pal->first << " at " << pal->second << endl;
		coordinates = scale_table(table, pal->second, pal->first);
		output << pal->first << "," << coordinates.x << "," << coordinates.y << "\n";
	}
	output.close();
	//cout << table.size().height << " and " << table.size().width << endl;

	//scale_table(display, center, radius, display);
	
	waitKey();
	return 0;
}


/*
int main(int argc, char** argv)
{
Mat src, src_gray;

if (argc < 2)
{
std::cerr << "No input image specified\n";
return -1;
}

// Read the image
src = imread(argv[1], IMREAD_COLOR);

if (src.empty())
{
std::cerr << "Invalid input image\n";
return -1;
}
//namedWindow(windowNameCircles, WINDOW_NORMAL);
//namedWindow(circleDisplay, WINDOW_NORMAL);

//imshow("Original", src);
//waitKey();

//Creates a rectangular region of interest
Rect r(450, 200, 1100, 600);
Mat3b roi3b(src(r));
Mat display = roi3b.clone();

Mat1b roiGray;
cvtColor(roi3b, roiGray, COLOR_BGR2GRAY);
GaussianBlur(roiGray, roiGray, Size(9, 9), 7);
//GaussianBlur(roiGray, roiGray, Size(9, 9), 0, 2);
//blur(roiGray, roiGray, Size(9,9));

Mat threshold_output;
//threshold(roiGray, threshold_output, thresh, 255, THRESH_BINARY);
adaptiveThreshold(roiGray, roiGray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);

Mat erodeElement = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

//The bals are recognized here, you can try and make it more accurate
erode(roiGray, roiGray, erodeElement);
erode(roiGray, roiGray, erodeElement);
erode(roiGray, roiGray, erodeElement);
erode(roiGray, roiGray, erodeElement);

dilate(roiGray, roiGray, erodeElement);
dilate(roiGray, roiGray, erodeElement);
dilate(roiGray, roiGray, erodeElement);
dilate(roiGray, roiGray, erodeElement);


//morphologyEx(roiGray, roiGray, MORPH_CLOSE, erodeElement);

//these two vectors needed for output of findContours
vector< vector<Point> > contours;
vector<Vec4i> hierarchy;
//find contours of filtered image using openCV findContours function
//findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
findContours(roiGray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
vector<vector<Point> > contours_poly(contours.size());
vector<Point2f>center(contours.size());
vector<float>radius(contours.size());


for (int i = 0; i < contours.size(); i++)
{
approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
}

Mat drawing = Mat::zeros(roiGray.size(), CV_8UC3);
for (int i = 0; i< contours.size(); i++)
{
Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//drawContours(display, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
if (radius[i] < 18 && radius[i] > 10)
{
circle(display, center[i], (int)radius[i], color, 2, 8, 0);
}
//circle(display, center[i], (int)radius[i], color, 2, 8, 0);
cv::imshow("Contours", display);

waitKey();
}

/// Show in a window
cv::namedWindow("Contours", CV_WINDOW_NORMAL);
cv::imshow("Contours", display);
}
*/

