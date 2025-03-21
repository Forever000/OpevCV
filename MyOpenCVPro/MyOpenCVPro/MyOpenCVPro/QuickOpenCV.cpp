#include"QuickOpenCV.h"

MyQuickOpenCV::MyQuickOpenCV()
{
}

MyQuickOpenCV::~MyQuickOpenCV()
{
}

void MyQuickOpenCV::load_image(cv::Mat &src) {
	//�����ļ�
	//std::cout << src.depth()<<std::endl;
	if (src.empty()) {
		std::cout << "δ�ܴ��ļ�" << std::endl;
		return;
	}
	//cv::namedWindow("���봰��", cv::WINDOW_FREERATIO);
	//cv::imshow("���봰��", src);
}

void MyQuickOpenCV::color_space_demo(cv::Mat &image) {
	//HSV��Ҷ�ͼת��
	cv::Mat gray, hsv;
	//HSV��H 0-180��S��0-255��V��0-255��V�������ȣ�S�������Ͷȣ�H������ɫ
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	cv::imshow("HSV", hsv);
	cv::imshow("�Ҷ�", gray);
	//cv::imwrite("./hsv.png", hsv);
	//cv::imwrite("./gray.png", gray);
}

void MyQuickOpenCV::mat_creation_demo(cv::Mat &image) {
	//ͨ�����󴴽�ͼƬ
	cv::Mat m1, m2;
	m1 = image.clone();
	image.copyTo(m2);

	//�����հ�ͼ��
	cv::Mat m3 = cv::Mat::zeros(Size(512, 512), CV_8UC3);//3ͨ��
	cv::Mat m4 = cv::Mat::ones(Size(8, 8), CV_8UC3);//3ͨ��
	m3 = Scalar(127, 127, 127);//�޸�����ֵ
	std::cout << "width:" << m3.cols << " height:" << m3.rows << " channels:" << m3.channels() << std::endl;
	//std::cout << m3 << std::endl;
	imshow("�½�ͼƬ", m3);
}

void MyQuickOpenCV::pixel_travel_demo(cv::Mat &image) {
	//��������ֵ
	int w = image.cols;
	int h = image.rows;
	int dim = image.channels();
	/*for (int row = 0; row < h; row++){
		for (int col = 0; col < w; col++){
			if (dim == 1){
				//�Ҷ�ͼ
				int pv = image.at<uchar>(row, col);
				image.at<uchar>(row, col) = 255 - pv;
			}
			if (dim == 3){
				//��ɫͼ
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
				//�Ҷ�ͼ
				int pv = *current_row;
				*current_row++ = 255 - *current_row;
			}
			if (dim == 3) {
				//��ɫͼ
				for (int k = 0; k < 3; k++) *current_row++ = saturate_cast<uchar>(*current_row + 200); //saturate_cast��֤�ӷ���ֵ������255
			}
		}
	}
	imshow("��ɫ��ת", image);
}

void MyQuickOpenCV::operators_demo(cv::Mat &image) {
	//ͼ��ļӼ���
	cv::Mat dst;
	//dst = image + cv::Scalar(50, 50, 50);//�ӷ�����
	//dst = image - cv::Scalar(50, 50, 50);//��������
	//dst = image / cv::Scalar(50, 50, 50);//�ӷ�����
	cv::Mat m = ::Mat::zeros(image.size(), image.type());
	m = Scalar(2, 2, 2);
	//multiply(image, m, dst);//�˷�
	//add(image, m, dst);//�ӷ�
	//divide(image, m, dst);//����

	imshow("ͼ��ӷ�", dst);
}

static void trackbarCallBack(int b, void* userdata) {
	//b��ʾ��������������value��userdata��ʾ�ص���������ͼ��
	Mat image = *((Mat*)userdata);
	Mat dst = cv::Mat::zeros(image.size(), image.type());
	Mat m = cv::Mat::zeros(image.size(), image.type());
	m = Scalar(b, b, b);
	add(image, m, dst);
	imshow("������Աȶȵ���", dst);
}
static void on_lightness(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = cv::Mat::zeros(image.size(), image.type());
	Mat m = cv::Mat::zeros(image.size(), image.type());
	double alpha = 1.0;
	double beta = 0;
	double gamma = b;
	addWeighted(image, alpha, m, beta, gamma, dst);
	imshow("������Աȶȵ���", dst);
}

static void on_contrast(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = cv::Mat::zeros(image.size(), image.type());
	Mat m = cv::Mat::zeros(image.size(), image.type());
	double alpha = b / 100.0;
	double beta = 0.0;
	double gamma = 0;
	addWeighted(image, alpha, m, beta, gamma, dst);
	imshow("������Աȶȵ���", dst);
}

void MyQuickOpenCV::tracking_bar_demo(cv::Mat &image) {
	namedWindow("������Աȶȵ���", WINDOW_AUTOSIZE);
	int max_value = 100;
	int lightness = 20;
	int contrast = 100;
	//createTrackbar("Tracking Bar:", "������Աȶȵ���", &lightness, max_value,trackbarCallBack,(void*)(&image));
	createTrackbar("Tracking Bar:", "������Աȶȵ���", &lightness, max_value, on_lightness, (void*)(&image));
	createTrackbar("Contrast Bar:", "������Աȶȵ���", &contrast, 200, on_contrast, (void*)(&image));
}

void MyQuickOpenCV::key_demo(cv::Mat &image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	while (true) {
		int c = waitKey(1000);
		if (c == 27) {
			//����ESC�˳�
			break;
		}
		if (c == 'b' - 0) {
			cvtColor(image, dst, COLOR_BGR2GRAY);
		}
		if (c == 'h' - 0) {
			cvtColor(image, dst, COLOR_BGR2HSV);
		}
		imshow("������Ӧ", dst);
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
		imshow("��ɫ���", dst);
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
	imshow("����λ����", dst);
}

void MyQuickOpenCV::channels_demo(cv::Mat &image) {
	std::vector<Mat> mv;
	split(image, mv);
	imshow("��ɫ", mv[0]);
	imshow("��ɫ", mv[1]);
	imshow("��ɫ", mv[2]);

	Mat dst;
	mv[1] = 0;
	mv[2] = 0;
	merge(mv, dst);
	imshow("��ɫ", dst);

	int from_to[] = { 0,2,1,1,2,0 };
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	imshow("ͨ�����", dst);

}

void MyQuickOpenCV::inrange_demo(cv::Mat &image) {
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);

	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(20, 20, 200);
	bitwise_not(mask, mask);
	image.copyTo(redback, mask);//��image������redback����ֻ����mask�в�Ϊ0�����ص�
	imshow("redback", redback);
}

void MyQuickOpenCV::pixel_statistic_demo(cv::Mat &image) {
	double minv, maxv;
	Point minp, maxp;
	std::vector<Mat> mv;
	split(image, mv);
	for (int i = 0; i < mv.size(); i++) {
		minMaxLoc(mv[i], &minv, &maxv, &minp, &maxp, Mat());//ÿ��ͨ���ľ�ֵ
		std::cout << "No.channels: " << i << " min value: " << minv << " max value :" << maxv << std::endl;
	}
	Mat mean, stddev;
	meanStdDev(image, mean, stddev);//����
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
	imshow("����չʾ", dst);
}

void MyQuickOpenCV::random_drawing() {
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(12345);//�������
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
			imshow("ROI����", image(rect));
			rectangle(image, rect, Scalar(0, 255, 255), 2, 8, 0);
			imshow("������", image);
			sp.x = -1;//������
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
				tem.copyTo(image);//���ƶ��Ĺ����У�Ϊ�����ƶ�����·���ı�������ԭͼ��������ʾ��ͼ�У���image���и���
				rectangle(image, rect, Scalar(255, 0, 0), 2, 8, 0);
				imshow("������", image);
			}
		}
	}
}

void MyQuickOpenCV::mouse_drawing_demo(Mat &image) {
	namedWindow("������", WINDOW_AUTOSIZE);
	setMouseCallback("������", on_draw, (void*)(&image));
	imshow("������", image);
	tem = image.clone();
}

void MyQuickOpenCV::norm_demo(Mat &image) {
	Mat dst;
	std::cout << image.type() << std::endl;
	image.convertTo(image, CV_32F);//������ת��Ϊ���������ٽ��й�һ��
	std::cout << image.type() << std::endl;
	normalize(image, dst, 1.0, 0, NORM_MINMAX);
	imshow("ͼ�����ݹ�һ��", dst);
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
	imshow("ͼ����ת180", dst);
	flip(image, dst, 0);
	imshow("ͼ��ת", dst);
	flip(image, dst, 1);
	imshow("ͼ��ˮƽ��ת", dst);
}

void MyQuickOpenCV::rotate_demo(Mat &image) {
	//����仯
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
	imshow("����仯", dst);
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
		//createButton("����ת",,frame);
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
	//��ͨ������
	std::vector<Mat> bgr_plane;
	split(image, bgr_plane);
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	//����BGRͨ��ֱ��ͼ
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
	//��ʾֱ��ͼ
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	//��һ��
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//���ӻ�
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
	imshow("�Ҷ�ͼ", gray);
	equalizeHist(gray, dst);
	imshow("ֱ��ͼ���⻯", dst);
}

void MyQuickOpenCV::blue_demo(Mat &image) {
	//ͼ����
	Mat dst;
	blur(image, dst, Size(3, 3), Point(-1, -1));
	imshow("��ֵ���ģ����", dst);
}

void MyQuickOpenCV::gaussian_demo(Mat &image) {
	//��˹ģ��
	Mat dst;
	GaussianBlur(image, dst, Size(3, 3), 15);
	imshow("��˹���ģ����", dst);
}

void MyQuickOpenCV::bifilter_demo(Mat &image) {
	//��˹˫��ģ����������������
	Mat dst;
	bilateralFilter(image, dst, 0, 10, 10);
	imshow("��˹˫��ģ����", dst);
}
