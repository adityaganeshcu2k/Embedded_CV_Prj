CXX = g++
CXXFLAGS = -O2 -std=c++17 -I/usr/local/include/opencv4
LDFLAGS = -L/usr/lib
LDLIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio

TARGET = rubiks_cube_tracker
SRCS = rubiks_cube_tracker_ver_1.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(TARGET) *.o
