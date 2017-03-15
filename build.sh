wget http://dlib.net/files/mmod_human_face_detector.dat.bz2
bzip2 -d mmod_human_face_detector.dat.bz2
cp ./dlib/examples/faces/2008_002470.jpg ./


clang++ -Wall -v -std=c++1z -stdlib=libstdc++ -I./dlib/ main.cpp ./dlib/build/dlib/libdlib.a -o main.o
