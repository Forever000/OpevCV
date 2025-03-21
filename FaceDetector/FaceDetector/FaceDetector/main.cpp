#include"FD.h"

int main() {
	FD::getInstance()->faceDetect();

	cv::waitKey(10000);
	cv::destroyAllWindows();
	return 0;
}