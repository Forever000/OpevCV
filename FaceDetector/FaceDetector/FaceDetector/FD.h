#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>

class FD
{
public:
	static FD *getInstance();
	void faceDetect();
	std::string model();
	std::string config();
private:
	FD();
	FD(const FD &);
	~FD();
	FD & operator = (const FD &);
private:
	std::string pb_file_path = "./Net/face_detector/opencv_face_detector_uint8.pb";
	std::string pbtxt_file_path = "./Net/face_detector/opencv_face_detector.pbtxt";
	static FD *m_instance;
};
