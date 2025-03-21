#include"FD.h"
FD *FD::m_instance = NULL;

FD::FD(){

}

FD::FD(const FD &) {

}

FD::~FD(){

}

FD & FD::operator = (const FD &) {

}
FD *FD::getInstance() {
	if (m_instance == nullptr) {
		m_instance = new FD();
	}
	return m_instance;
}

std::string FD::model() {
	return pb_file_path;
}

std::string FD::config() {
	return pbtxt_file_path;
}

void FD::faceDetect() {
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(FD::getInstance()->model(), FD::getInstance()->config());
	cv::VideoCapture capture(0);
	cv::Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) break;
		cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123), false, false);
		net.setInput(blob);
		cv::Mat probs = net.forward();
		//1*1*N*7
		cv::Mat detectMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
		for (int row = 0; row < detectMat.rows; row++) {
			float conf = detectMat.at<float>(row, 2);//ÖÃÐÅ¶È
			if (conf > 0.5) {
				float x1 = detectMat.at<float>(row, 3) *frame.cols;
				float y1 = detectMat.at<float>(row, 4)*frame.rows;
				float x2 = detectMat.at<float>(row, 5)*frame.cols;
				float y2 = detectMat.at<float>(row, 6)*frame.rows;
				cv::Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, cv::Scalar(0, 255, 0), 2, 8);
			}
		}
		flip(frame, frame, 1);//Í¼Ïñ·­×ª
		imshow("DNNÈËÁ³¼ì²â", frame);
		char c = cv::waitKey(1);
		if (c == 27) break;
	}
	capture.release();
}