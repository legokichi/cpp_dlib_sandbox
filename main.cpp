#include <dlib/image_processing/object_detector.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using std::cout;
using std::endl;

// http://dlib.net/dnn_mmod_face_detection_ex.cpp.html
// http://dlib.net/files/data/dlib_face_detection_dataset-2016-09-30.tar.gz
// faces_2016_09_30.xml
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

int main(){ try{
  net_type net;
  deserialize("./mmod_human_face_detector.dat") >> net;
  cout << "deserialize" << endl;

  matrix<rgb_pixel> img;
  load_image(img, "./2008_002470.jpg");
  cout << "img" << endl;

  while(img.size() < 1800*1800){ pyramid_up(img); }
  cout << "pyramid_up" << endl;

  const std::vector<mmod_rect> dets = net(img);
  cout << "net" << endl;

  for (auto&& d : dets){
    auto rect = d.rect;
    cout
      << "(" << rect.left() << "," << rect.top() << ")"
      << ","
      << "(" << rect.right() << "," << rect.bottom() << ")"
      << endl;
  }

  return 0;
}catch(std::exception& e){
  cout << "\nexception thrown!" << endl;
  cout << e.what() << endl;
  return 1;
} }


