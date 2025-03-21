#include"QuickOpenCV.h"

int main() {
	cv::Mat src = cv::imread("./Resources/dream.jpg", cv::IMREAD_UNCHANGED);
	MyQuickOpenCV mqo;
	mqo.load_image(src);
	//mqo.color_space_demo(src);
	//mqo.mat_creation_demo(src);
	//mqo.pixel_travel_demo(src);
	//mqo.operators_demo(src);
	//mqo.tracking_bar_demo(src);
	//mqo.key_demo(src);
	//mqo.color_style_demo(src);
	//mqo.bitwise_demo(src);
	//mqo.channels_demo(src);
	//mqo.inrange_demo(src);
	//mqo.pixel_statistic_demo(src);
	//mqo.drawing_demo(src);
	//mqo.random_drawing();
	//mqo.mouse_drawing_demo(src);
	//mqo.norm_demo(src);
	//mqo.resize_demo(src);
	//mqo.flip_demo(src);
	//mqo.rotate_demo(src);
	//mqo.video_demo(src);
	//mqo.histogram_demo(src);
	//mqo.histogram_eq_demo(src);
	//mqo.blue_demo(src);
	//mqo.gaussian_demo(src);
	mqo.bifilter_demo(src);

	cv::waitKey(10000);
	cv::destroyAllWindows();
	return 0;
}