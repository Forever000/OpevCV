#include"QuickOpenCV.h"

MyQuickOpenCV::MyQuickOpenCV()
{
}

MyQuickOpenCV::~MyQuickOpenCV()
{
}

void MyQuickOpenCV::load_image(cv::Mat &src) {
	//加载文件
	//std::cout << src.depth()<<std::endl;
	if (src.empty()) {
		std::cout << "未能打开文件" << std::endl;
		return;
	}
	//cv::namedWindow("输入窗口", cv::WINDOW_FREERATIO);
	//cv::imshow("输入窗口", src);
}

void MyQuickOpenCV::color_space_demo(cv::Mat &image) {
	//HSV与灰度图转换
	cv::Mat gray, hsv;
	//HSV：H 0-180，S：0-255，V：0-255，V调整亮度，S调整饱和度，H调整颜色
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	cv::imshow("HSV", hsv);
	cv::imshow("灰度", gray);
	//cv::imwrite("./hsv.png", hsv);
	//cv::imwrite("./gray.png", gray);
}

void MyQuickOpenCV::mat_creation_demo(cv::Mat &image) {
	//通过矩阵创建图片
	cv::Mat m1, m2;
	m1 = image.clone();
	image.copyTo(m2);

	//创建空白图像
	cv::Mat m3 = cv::Mat::zeros(Size(512, 512), CV_8UC3);//3通道
	cv::Mat m4 = cv::Mat::ones(Size(8, 8), CV_8UC3);//3通道
	m3 = Scalar(127, 127, 127);//修改像素值
	std::cout << "width:" << m3.cols << " height:" << m3.rows << " channels:" << m3.channels() << std::endl;
	//std::cout << m3 << std::endl;
	imshow("新建图片", m3);
}

void MyQuickOpenCV::pixel_travel_demo(cv::Mat &image) {
	//遍历像素值
	int w = image.cols;
	int h = image.rows;
	int dim = image.channels();
	/*for (int row = 0; row < h; row++){
		for (int col = 0; col < w; col++){
			if (dim == 1){
				//灰度图
				int pv = image.at<uchar>(row, col);
				image.at<uchar>(row, col) = 255 - pv;
			}
			if (dim == 3){
				//彩色图
				Vec3b pv = image.at<Vec3b>(row, col);
				image.at<Vec3b>(row, col)[0] = 255 - pv[0];
				image.at<Vec3b>(row, col)[1] = 255 - pv[1];
				image.at<Vec3b>(row, col)[2] = 255 - pv[2];
			}
		}
	}*/

	for (int row = 0; row < h; row++) {
		uchar *current_row = image.ptr<uchar>(row);
		for (int col = 0; col < w; col++) {
			if (dim == 1) {
				//灰度图
				int pv = *current_row;
				*current_row++ = 255 - *current_row;
			}
			if (dim == 3) {
				//彩色图
				for (int k = 0; k < 3; k++) *current_row++ = saturate_cast<uchar>(*current_row + 200); //saturate_cast保证加法的值不超过255
			}
		}
	}
	imshow("颜色反转", image);
}

void MyQuickOpenCV::operators_demo(cv::Mat &image) {
	//图像的加减法
	cv::Mat dst;
	//dst = image + cv::Scalar(50, 50, 50);//加法操作
	//dst = image - cv::Scalar(50, 50, 50);//减法操作
	//dst = image / cv::Scalar(50, 50, 50);//加法操作
	cv::Mat m = ::Mat::zeros(image.size(), image.type());
	m = Scalar(2, 2, 2);
	//multiply(image, m, dst);//乘法
	//add(image, m, dst);//加法
	//divide(image, m, dst);//除法

	imshow("图像加法", dst);
}

static void trackbarCallBack(int b, void* userdata) {
	//b表示进度条传进来的value，userdata表示回调传进来的图像
	Mat image = *((Mat*)userdata);
	Mat dst = cv::Mat::zeros(image.size(), image.type());
	Mat m = cv::Mat::zeros(image.size(), image.type());
	m = Scalar(b, b, b);
	add(image, m, dst);
	imshow("亮度与对比度调整", dst);
}
static void on_lightness(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = cv::Mat::zeros(image.size(), image.type());
	Mat m = cv::Mat::zeros(image.size(), image.type());
	double alpha = 1.0;
	double beta = 0;
	double gamma = b;
	addWeighted(image, alpha, m, beta, gamma, dst);
	imshow("亮度与对比度调整", dst);
}

static void on_contrast(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = cv::Mat::zeros(image.size(), image.type());
	Mat m = cv::Mat::zeros(image.size(), image.type());
	double alpha = b / 100.0;
	double beta = 0.0;
	double gamma = 0;
	addWeighted(image, alpha, m, beta, gamma, dst);
	imshow("亮度与对比度调整", dst);
}

void MyQuickOpenCV::tracking_bar_demo(cv::Mat &image) {
	namedWindow("亮度与对比度调整", WINDOW_AUTOSIZE);
	int max_value = 100;
	int lightness = 20;
	int contrast = 100;
	//createTrackbar("Tracking Bar:", "亮度与对比度调整", &lightness, max_value,trackbarCallBack,(void*)(&image));
	createTrackbar("Tracking Bar:", "亮度与对比度调整", &lightness, max_value, on_lightness, (void*)(&image));
	createTrackbar("Contrast Bar:", "亮度与对比度调整", &contrast, 200, on_contrast, (void*)(&image));
}

void MyQuickOpenCV::key_demo(cv::Mat &image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	while (true) {
		int c = waitKey(1000);
		if (c == 27) {
			//按键ESC退出
			break;
		}
		if (c == 'b' - 0) {
			cvtColor(image, dst, COLOR_BGR2GRAY);
		}
		if (c == 'h' - 0) {
			cvtColor(image, dst, COLOR_BGR2HSV);
		}
		imshow("键盘响应", dst);
	}
}

void MyQuickOpenCV::color_style_demo(cv::Mat &image) {
	int color_map[] = {
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_CIVIDIS,
		COLORMAP_COOL,
		COLORMAP_DEEPGREEN,
		COLORMAP_HOT,
		COLORMAP_HSV,
		COLORMAP_INFERNO,
		COLORMAP_JET,
		COLORMAP_MAGMA,
		COLORMAP_OCEAN,
		COLORMAP_PARULA,
		COLORMAP_PINK,
		COLORMAP_PLASMA,
		COLORMAP_RAINBOW,
		COLORMAP_SPRING,
		COLORMAP_SUMMER,
		COLORMAP_TURBO,
		COLORMAP_TWILIGHT,
		COLORMAP_TWILIGHT_SHIFTED,
		COLORMAP_VIRIDIS,
		COLORMAP_WINTER
	};

	Mat dst;
	int index = 0;
	while (true) {
		int c = waitKey(500);
		if (c == 27) {
			break;
		}
		applyColorMap(image, dst, color_map[(index++) % 19]);
		imshow("颜色风格", dst);
	}
}

void MyQuickOpenCV::bitwise_demo(cv::Mat &image) {
	Mat m1 = Mat::zeros(image.size(), image.type());
	Mat m2 = Mat::zeros(image.size(), image.type());
	rectangle(m1, Rect(100, 100, 85, 85), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(m2, Rect(150, 150, 85, 85), Scalar(0, 255, 255), -1, LINE_8, 0);
	imshow("m1", m1);
	imshow("m2", m2);
	Mat dst;
	//bitwise_and(m1, m2, dst);
	//bitwise_not(m1, m2, dst);
	//bitwise_or(m1, m2, dst);
	bitwise_xor(m1, m2, dst);
	imshow("像素位操作", dst);
}

void MyQuickOpenCV::channels_demo(cv::Mat &image) {
	std::vector<Mat> mv;
	split(image, mv);
	imshow("蓝色", mv[0]);
	imshow("绿色", mv[1]);
	imshow("红色", mv[2]);

	Mat dst;
	mv[1] = 0;
	mv[2] = 0;
	merge(mv, dst);
	imshow("蓝色", dst);

	int from_to[] = { 0,2,1,1,2,0 };
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	imshow("通道混合", dst);

}

void MyQuickOpenCV::inrange_demo(cv::Mat &image) {
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);

	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(20, 20, 200);
	bitwise_not(mask, mask);
	image.copyTo(redback, mask);//将image拷贝到redback，且只拷贝mask中不为0的像素点
	imshow("redback", redback);
}

void MyQuickOpenCV::pixel_statistic_demo(cv::Mat &image) {
	double minv, maxv;
	Point minp, maxp;
	std::vector<Mat> mv;
	split(image, mv);
	for (int i = 0; i < mv.size(); i++) {
		minMaxLoc(mv[i], &minv, &maxv, &minp, &maxp, Mat());//每个通道的均值
		std::cout << "No.channels: " << i << " min value: " << minv << " max value :" << maxv << std::endl;
	}
	Mat mean, stddev;
	meanStdDev(image, mean, stddev);//方差
	std::cout << "mean: " << mean << std::endl << " stddev: " << stddev << std::endl;
}

void MyQuickOpenCV::drawing_demo(cv::Mat &image) {
	Rect rect;
	rect.x = 130;
	rect.y = 50;
	rect.width = 370;
	rect.height = 480;
	Mat dst, bg = Mat::zeros(image.size(), image.type());
	rectangle(bg, rect, Scalar(0, 0, 255), -1, LINE_8, 0);
	//circle(image, Point(200, 200), 100, Scalar(146, 97, 17), 1, LINE_8, 0);
	line(bg, Point(130, 50), Point(500, 530), Scalar(0, 255, 0), 2, LINE_8, 0);
	RotatedRect rrt;
	rrt.center = Point(200, 200);
	rrt.size = Size(100, 200);
	rrt.angle = 0.0;
	ellipse(bg, rrt, Scalar(0, 255, 255), 2, 8);

	addWeighted(image, 0.7, bg, 0.3, 0, dst);
	imshow("绘制展示", dst);
}

void MyQuickOpenCV::random_drawing() {
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(12345);//随机种子
	while (true) {
		int c = waitKey(1);
		if (c == 27) break;
		int x1 = rng.uniform(0, w);
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		int b = rng.uniform(0, 180), g = rng.uniform(0, 255), r = rng.uniform(0, 255);
		canvas = Scalar(0, 0, 0);
		line(canvas, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 2, LINE_8, 0);
		imshow("canvas", canvas);
	}
}

Point sp, ep;
Mat tem;
static void on_draw(int event, int x, int y, int flags, void *userdata) {
	Mat image = *((Mat*)userdata);
	int imw = image.size().width;
	int imh = image.size().height;
	if (event == EVENT_LBUTTONDOWN) {
		sp.x = x;
		sp.y = y;
		std::cout << "sp: " << sp << std::endl;
	}
	else if (event == EVENT_LBUTTONUP) {
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0) {
			Rect rect(sp.x, sp.y, dx, dy);
			tem.copyTo(image);
			imshow("ROI区域", image(rect));
			rectangle(image, rect, Scalar(0, 255, 255), 2, 8, 0);
			imshow("鼠标绘制", image);
			sp.x = -1;//清除标记
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				Rect rect(sp.x, sp.y, dx, dy);
				tem.copyTo(image);//在移动的过程中，为避免移动过程路径的保留，将原图拷贝到显示的图中，对image进行覆盖
				rectangle(image, rect, Scalar(255, 0, 0), 2, 8, 0);
				imshow("鼠标绘制", image);
			}
		}
	}
}

void MyQuickOpenCV::mouse_drawing_demo(Mat &image) {
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));
	imshow("鼠标绘制", image);
	tem = image.clone();
}

void MyQuickOpenCV::norm_demo(Mat &image) {
	Mat dst;
	std::cout << image.type() << std::endl;
	image.convertTo(image, CV_32F);//将数据转换为浮点数，再进行归一化
	std::cout << image.type() << std::endl;
	normalize(image, dst, 1.0, 0, NORM_MINMAX);
	imshow("图像数据归一化", dst);
}

void MyQuickOpenCV::resize_demo(Mat &image) {
	Mat zoomin, zoomout;
	int h = image.rows;
	int w = image.cols;
	resize(image, zoomin, Size(w / 2, h / 2), 0, 0, INTER_CUBIC);
	imshow("ZoomIn", zoomin);
	resize(image, zoomout, Size(w*1.5, h *1.5), 0, 0, INTER_LINEAR);
	imshow("Zoomout", zoomout);
}

void MyQuickOpenCV::flip_demo(Mat &image) {
	Mat dst;
	flip(image, dst, -1);
	imshow("图像旋转180", dst);
	flip(image, dst, 0);
	imshow("图像倒转", dst);
	flip(image, dst, 1);
	imshow("图像水平翻转", dst);
}

void MyQuickOpenCV::rotate_demo(Mat &image) {
	//仿射变化
	Mat dst, M;
	int h = image.rows;
	int w = image.cols;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45, 1.0);

	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);

	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(255, 255, 0));
	imshow("仿射变化", dst);
}

void MyQuickOpenCV::video_demo(Mat &image) {
	VideoCapture capture(0);
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int count = capture.get(CAP_PROP_FRAME_COUNT);
	double fps = capture.get(CAP_PROP_FPS);
	VideoWriter writer("./test.mp4", CAP_PROP_FOURCC, fps, Size(frame_width, frame_height), true);
	Mat frame;
	while (true) {
		capture.read(frame);
		//createButton("镜像翻转",,frame);
		flip(frame, frame, 1);
		if (frame.empty()) break;
		imshow("frame", frame);
		writer.write(frame);
		color_space_demo(frame);
		int c = waitKey(1);
		if (c == 27) break;
	}
	writer.release();
	capture.release();
}

void MyQuickOpenCV::histogram_demo(Mat &image) {
	//三通道分离
	std::vector<Mat> bgr_plane;
	split(image, bgr_plane);
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	//计算BGR通道直方图
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
	//显示直方图
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	//归一化
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//可视化
	for (int i = 0; i < bins[0]; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*i, hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*i, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*i, hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	Mat bg = Mat::zeros(histImage.size(), histImage.type());
	addWeighted(histImage, 1.0, bg, 0, 50, histImage);
	imshow("Histogram Demo", histImage);
}

void MyQuickOpenCV::histogram_2d_demo(Mat &image) {

}

void MyQuickOpenCV::histogram_eq_demo(Mat &image) {
	Mat gray, dst;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("灰度图", gray);
	equalizeHist(gray, dst);
	imshow("直方图均衡化", dst);
}

void MyQuickOpenCV::blue_demo(Mat &image) {
	//图像卷积
	Mat dst;
	blur(image, dst, Size(3, 3), Point(-1, -1));
	imshow("均值卷积模糊化", dst);
}

void MyQuickOpenCV::gaussian_demo(Mat &image) {
	//高斯模糊
	Mat dst;
	GaussianBlur(image, dst, Size(3, 3), 15);
	imshow("高斯卷积模糊化", dst);
}

void MyQuickOpenCV::bifilter_demo(Mat &image) {
	//高斯双边模糊，保留轮廓部分
	Mat dst;
	bilateralFilter(image, dst, 0, 10, 10);
	imshow("高斯双边模糊化", dst);
}
