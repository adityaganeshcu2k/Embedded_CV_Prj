CXX       := g++
CXXFLAGS  := -O2 -std=c++17 -I/usr/local/include/opencv4
LDFLAGS   := -L/usr/lib
LDLIBS    := -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio

SRC_DIR   := .
SRCS      := $(SRC_DIR)/rubiks_cube_tracker_ver_3.cpp \
             $(SRC_DIR)/helper_funcs.cpp
OBJS      := $(SRCS:.cpp=.o)

TARGET    := rubiks_cube_tracker

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Compile any .cpp to .o; rebuild if the corresponding .hpp changes
%.o: %.cpp helper_funcs.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
