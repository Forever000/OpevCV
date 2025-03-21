#pragma once

#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
using namespace cv;
class MyQuickOpenCV
{
public:
	MyQuickOpenCV();
	~MyQuickOpenCV();
	void load_image(cv::Mat &src);
	void color_space_demo(cv::Mat &image);
	void mat_creation_demo(cv::Mat &image);
	void pixel_travel_demo(cv::Mat &image);
	void operators_demo(cv::Mat &image);
	void tracking_bar_demo(cv::Mat &image);
	void key_demo(cv::Mat &image);
	void color_style_demo(cv::Mat &image);
	void bitwise_demo(cv::Mat &image);
	void channels_demo(cv::Mat &image);
	void inrange_demo(cv::Mat &image);
	void pixel_statistic_demo(cv::Mat &image);
	void drawing_demo(cv::Mat &image);
	void random_drawing();
	void mouse_drawing_demo(Mat &image);
	void norm_demo(Mat &image);
	void resize_demo(Mat &image);
	void flip_demo(Mat &image);
	void rotate_demo(Mat &image);
	void video_demo(Mat &image);
	void histogram_demo(Mat &image);
	void histogram_2d_demo(Mat &image);
	void histogram_eq_demo(Mat &image);
	void blue_demo(Mat &image);
	void gaussian_demo(Mat &image);
	void bifilter_demo(Mat &image);

private:

};

